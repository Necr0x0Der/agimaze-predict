"""Train Llama with an action-event-aligned 2-D memory on TXT traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import LlamaWithSpatialMemory
from .spatial_memory import SpatialMemoryConfig
from .train import _lora_checkpoint, _maybe_lora, _require_transformers
from .txt_config import resolve_txt_training_arguments
from .txt_data import LlamaTxtRollout, collate_txt_rollouts, load_llama_txt_rollouts


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collator(tokenizer, config: SpatialMemoryConfig):
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Llama tokenizer has neither pad_token_id nor eos_token_id")

    def collate(examples: Sequence[LlamaTxtRollout]) -> dict[str, Tensor]:
        items = collate_txt_rollouts(
            examples, tokenizer=tokenizer, canvas_height=config.canvas_height,
            canvas_width=config.canvas_width, pad_token_id=pad,
        )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in items.items() if key != "target_token_ids"}

    return collate


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _prompt_only_batch(batch: dict[str, Tensor], *, pad_token_id: int) -> dict[str, Tensor]:
    """Remove teacher-forced TXT tokens before an event-memory prefill."""

    lengths = batch["prompt_lengths"].tolist()
    width = max(lengths)
    input_ids = batch["input_ids"].new_full((len(lengths), width), pad_token_id)
    attention_mask = batch["attention_mask"].new_zeros((len(lengths), width))
    for index, length in enumerate(lengths):
        input_ids[index, :length] = batch["input_ids"][index, :length]
        attention_mask[index, :length] = 1
    result = dict(batch)
    result["input_ids"] = input_ids
    result["attention_mask"] = attention_mask
    return result


def _left_pad_embeds(embeds: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Align all final real prefix positions at ``-1`` for batched decoding."""

    batch, width, hidden = embeds.shape
    result = embeds.new_zeros((batch, width, hidden))
    mask = attention_mask.new_zeros((batch, width))
    for index, length in enumerate(attention_mask.sum(dim=1).tolist()):
        if length <= 0:
            raise ValueError("TXT generation prefix cannot be empty")
        result[index, width - length:] = embeds[index, :length]
        mask[index, width - length:] = 1
    return result, mask


@torch.no_grad()
def _greedy_target(model: LlamaWithSpatialMemory, batch: dict[str, Tensor], *, tokens: int, eos_token_id: int | None) -> list[list[int]]:
    """Greedy TXT continuation after an event-memory prefill, using Llama KV cache."""

    inputs = {key: value for key, value in batch.items() if key != "labels"}
    actions, action_mask = model._action_embeddings(inputs["action_input_ids"], inputs["action_attention_mask"])
    memory = model.memory_projection(model.workspace.rollout(inputs["visual_maps"], actions, action_mask))
    token_embeddings = model.llama.get_input_embeddings()(inputs["input_ids"])
    memory = memory.to(dtype=token_embeddings.dtype)
    embeds, mask, _ = model._insert_event_memory(
        token_embeddings, inputs["attention_mask"], None, memory,
        inputs["event_token_positions"], inputs["event_mask"],
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
        output = model.llama(input_ids=next_token[:, None], attention_mask=mask, past_key_values=output.past_key_values, use_cache=True)
    return generated


@torch.no_grad()
def evaluate_txt_rollouts(
    model: LlamaWithSpatialMemory, examples: Sequence[LlamaTxtRollout], *, collate, tokenizer, device: torch.device,
    batch_size: int, greedy_examples: int | None = None,
) -> dict[str, float]:
    if not examples or batch_size <= 0:
        raise ValueError("TXT validation examples and batch_size must be positive")
    was_training = model.training
    model.eval()
    total_loss, total_tokens, token_correct = 0.0, 0, 0
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
    for batch in loader:
        batch = _to_device(batch, device)
        output = model.forward_txt(**{key: value for key, value in batch.items() if key != "prompt_lengths"})
        labels = batch["labels"]
        active = labels[:, 1:].ne(-100)
        total_loss += float(output.loss.item()) * int(active.sum().item())
        total_tokens += int(active.sum().item())
        token_correct += int((output.logits[:, :-1].argmax(dim=-1).eq(labels[:, 1:]) & active).sum().item())
    limit = len(examples) if greedy_examples is None else min(len(examples), greedy_examples)
    exact = 0
    for start in range(0, limit, batch_size):
        group = examples[start:start + batch_size]
        raw = collate(group)
        # Torch collator intentionally omits Python target lists, so rebuild it
        # directly for exact greedy matching.
        target_ids = collate_txt_rollouts(group, tokenizer=tokenizer,
            canvas_height=model.workspace.config.canvas_height, canvas_width=model.workspace.config.canvas_width,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)["target_token_ids"]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        assert pad is not None
        raw = _prompt_only_batch(raw, pad_token_id=pad)
        generated = _greedy_target(model, _to_device(raw, device), tokens=max(map(len, target_ids)), eos_token_id=tokenizer.eos_token_id)
        exact += sum(actual == expected for actual, expected in zip(generated, target_ids))
    if was_training:
        model.train()
    return {
        "txt_target_token_nll": total_loss / max(total_tokens, 1),
        "txt_teacher_forced_token_accuracy": token_correct / max(total_tokens, 1),
        "txt_greedy_exact_accuracy": exact / limit,
        "txt_greedy_examples": float(limit),
    }


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("Llama-3.2-3B TXT training requires a CUDA device; no CUDA device is available")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if args.freeze_backbone:
        backbone.requires_grad_(False)
    backbone = _maybe_lora(backbone, args)
    config = SpatialMemoryConfig(canvas_height=args.canvas_height, canvas_width=args.canvas_width, d_model=args.d_model,
        n_heads=args.n_heads, spatial_layers=args.spatial_layers, memory_tokens=args.memory_tokens)
    model = LlamaWithSpatialMemory(backbone, config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters: enable workspace training or LoRA")
    train_examples = load_llama_txt_rollouts(args.train_datasets, initial_context=args.initial_context, depth=args.depth)
    validation_examples = load_llama_txt_rollouts(args.validation_datasets, initial_context=args.initial_context, depth=args.depth)
    collate = _collator(tokenizer, config)
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"training_rollouts={len(train_examples)} validation_rollouts={len(validation_examples)} trainable_parameters={sum(p.numel() for p in trainable)}", flush=True)
    metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, tokens = 0.0, 0
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_txt(**{key: value for key, value in batch.items() if key != "prompt_lengths"})
            output.loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            active = int(batch["labels"].ne(-100).sum().item())
            total += float(output.loss.item()) * active
            tokens += active
        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            metrics = evaluate_txt_rollouts(model, validation_examples, collate=collate, tokenizer=tokenizer, device=device,
                batch_size=args.batch_size, greedy_examples=args.greedy_evaluate_examples)
            print(
                f"epoch={epoch} train_txt_target_nll={total / max(tokens, 1):.6f} "
                f"val_txt_target_token_nll={metrics['txt_target_token_nll']:.6f} "
                f"val_txt_teacher_forced_token_accuracy={metrics['txt_teacher_forced_token_accuracy']:.4f} "
                f"val_txt_greedy_exact_accuracy={metrics['txt_greedy_exact_accuracy']:.4f} "
                f"val_txt_greedy_examples={metrics['txt_greedy_examples']:.0f}",
                flush=True,
            )
    output = Path(args.output)
    if output.is_dir():
        raise IsADirectoryError(f"checkpoint output must be a file, not a directory: {output}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"checkpoint already exists: {output} (use overwrite=true or --overwrite)")
    checkpoint = {
        "format": "agimaze_predict.llama_memory.txt_event.v1", "base_model": args.base_model,
        "spatial_config": config.to_dict(), "workspace": model.workspace.state_dict(),
        "memory_projection": model.memory_projection.state_dict(), "lora": _lora_checkpoint(model.llama, args),
        "datasets": {"train_paths": [str(path) for path in args.train_datasets], "validation_paths": [str(path) for path in args.validation_datasets]},
        "rollout": {"initial_context": args.initial_context, "depth": args.depth}, "metrics": metrics, "arguments": vars(args),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-dataset", dest="train_datasets", type=Path, action="append")
    parser.add_argument("--validation-dataset", dest="validation_datasets", type=Path, action="append")
    parser.add_argument("--initial-context", choices=("MAP", "START")); parser.add_argument("--depth", type=int)
    parser.add_argument("--output", type=Path); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-model"); parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--lora-rank", type=int); parser.add_argument("--lora-target-modules", nargs="+")
    parser.add_argument("--canvas-height", type=int); parser.add_argument("--canvas-width", type=int); parser.add_argument("--d-model", type=int); parser.add_argument("--n-heads", type=int); parser.add_argument("--spatial-layers", type=int); parser.add_argument("--memory-tokens", type=int)
    parser.add_argument("--seed", type=int); parser.add_argument("--epochs", type=int); parser.add_argument("--evaluate-every", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--learning-rate", type=float); parser.add_argument("--weight-decay", type=float); parser.add_argument("--grad-clip", type=float); parser.add_argument("--greedy-evaluate-examples", type=int); parser.add_argument("--device")
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = resolve_txt_training_arguments(parser)
        if min(args.epochs, args.evaluate_every, args.batch_size, args.depth, args.greedy_evaluate_examples) <= 0:
            parser.error("epochs, evaluate-every, batch-size, depth, and greedy-evaluate-examples must be positive")
        print(json.dumps({"checkpoint": str(train(args))}), flush=True)
    except (FileNotFoundError, FileExistsError, IsADirectoryError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
