"""Held-out loss and cached greedy POS evaluation for Llama spatial memory."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from agimaze_predict.data.prepared import PreparedExample

from .data import _ids, collate_examples, parse_example
from .model import LlamaWithSpatialMemory
from .spatial_memory import SpatialMemoryConfig


Collator = Callable[[Sequence[PreparedExample]], dict[str, Tensor]]


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _generation_batch(
    examples: Sequence[PreparedExample], *, tokenizer, config: SpatialMemoryConfig, pad_token_id: int, device: torch.device
) -> tuple[dict[str, Tensor], list[list[int]]]:
    """Prepare prompt-only inputs while retaining map/action tensors from the shared collator."""

    parsed = [parse_example(example) for example in examples]
    prompts = [_ids(tokenizer, item.action_text + "\n") for item in parsed]
    targets = [_ids(tokenizer, item.target) for item in parsed]
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    for target in targets:
        target.append(tokenizer.eos_token_id)
    if any(not prompt for prompt in prompts):
        raise ValueError("each action prompt must produce at least one tokenizer token")
    raw = collate_examples(
        examples,
        tokenizer=tokenizer,
        canvas_height=config.canvas_height,
        canvas_width=config.canvas_width,
        pad_token_id=pad_token_id,
    )
    width = max(map(len, prompts))
    input_ids = [prompt + [pad_token_id] * (width - len(prompt)) for prompt in prompts]
    attention_mask = [[1] * len(prompt) + [0] * (width - len(prompt)) for prompt in prompts]
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "prompt_lengths": torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long, device=device),
        "visual_maps": torch.tensor(raw["visual_maps"], dtype=torch.long, device=device),
        "action_input_ids": torch.tensor(raw["action_input_ids"], dtype=torch.long, device=device),
        "action_attention_mask": torch.tensor(raw["action_attention_mask"], dtype=torch.long, device=device),
    }
    return batch, targets


def _left_pad_embeds(embeds: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Left-pad injected prefixes so every batch member's final real token is at -1."""

    batch, width, hidden = embeds.shape
    result = embeds.new_zeros((batch, width, hidden))
    mask = attention_mask.new_zeros((batch, width))
    for index, length in enumerate(attention_mask.sum(dim=1).tolist()):
        if length <= 0:
            raise ValueError("generation prefix cannot be empty")
        result[index, width - length :] = embeds[index, :length]
        mask[index, width - length :] = 1
    return result, mask


@torch.no_grad()
def greedy_target_token_ids(
    model: LlamaWithSpatialMemory, batch: dict[str, Tensor], *, tokens: int, eos_token_id: int | None
) -> list[list[int]]:
    """Greedily decode ``tokens`` outputs using one cached Llama prefill per batch.

    The wrapper's memory is built once from the map/action history.  Subsequent
    target tokens use the backbone cache, rather than recomputing the full 3B
    parameter decoder for every generated POS token.
    """

    if tokens <= 0:
        raise ValueError("tokens must be positive")
    actions, action_mask = model._action_embeddings(batch["action_input_ids"], batch["action_attention_mask"])
    memory = model.memory_projection(model.workspace(batch["visual_maps"], actions, action_mask))
    token_embeddings = model.llama.get_input_embeddings()(batch["input_ids"])
    memory = memory.to(dtype=token_embeddings.dtype)
    embeds, mask, _ = model._insert_memory(
        token_embeddings, batch["attention_mask"], None, memory, batch["prompt_lengths"]
    )
    embeds, mask = _left_pad_embeds(embeds, mask)
    output = model.llama(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
    generated: list[list[int]] = [[] for _ in range(embeds.shape[0])]
    finished = torch.zeros(embeds.shape[0], dtype=torch.bool, device=embeds.device)
    for _ in range(tokens):
        next_token = output.logits[:, -1].argmax(dim=-1)
        for index, token in enumerate(next_token.tolist()):
            if not finished[index]:
                generated[index].append(token)
        if eos_token_id is not None:
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()):
                break
        mask = torch.cat((mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)), dim=1)
        output = model.llama(
            input_ids=next_token[:, None],
            attention_mask=mask,
            past_key_values=output.past_key_values,
            use_cache=True,
        )
    return generated


@torch.no_grad()
def evaluate_examples(
    model: LlamaWithSpatialMemory,
    examples: Sequence[PreparedExample],
    *,
    tokenizer,
    config: SpatialMemoryConfig,
    collate: Collator,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Compute target-token NLL and full greedy `<POS>...</POS>` exactness."""

    if not examples:
        raise ValueError("cannot evaluate an empty example collection")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    was_training = model.training
    model.eval()
    total_loss, target_tokens = 0.0, 0
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
    for batch in loader:
        batch = _to_device(batch, device)
        output = model(**batch)
        active = int(batch["labels"].ne(-100).sum().item())
        total_loss += float(output.loss.item()) * active
        target_tokens += active

    exact = 0
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch, expected = _generation_batch(
            batch_examples, tokenizer=tokenizer, config=config, pad_token_id=pad_token_id, device=device
        )
        generated = greedy_target_token_ids(
            model, batch, tokens=max(map(len, expected)), eos_token_id=tokenizer.eos_token_id
        )
        exact += sum(actual == wanted for actual, wanted in zip(generated, expected))

    if was_training:
        model.train()
    return {
        "target_token_nll": total_loss / max(target_tokens, 1),
        "greedy_exact_target_accuracy": exact / len(examples),
    }
