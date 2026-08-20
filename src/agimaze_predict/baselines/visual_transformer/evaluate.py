"""Evaluation and greedy POS decoding for the visual-memory model."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from agimaze_predict.data.prepared import PreparedExample

from .model import VisualTransformer, target_cross_entropy
from .tokenizer import collate_visual_examples, serialize_visual_example


def _model_inputs(batch: dict[str, list[list[int]]], device: torch.device) -> dict[str, Tensor]:
    return {
        "visual_maps": torch.tensor(batch["visual_maps"], dtype=torch.long, device=device),
        "event_positions": torch.tensor(batch["event_positions"], dtype=torch.long, device=device),
        "event_counts": torch.tensor(batch["event_counts"], dtype=torch.long, device=device),
    }


@torch.no_grad()
def greedy_target_bytes(model: VisualTransformer, example: PreparedExample) -> bytes:
    item = serialize_visual_example(example)
    device = next(model.parameters()).device
    prefix = item.token_ids[: item.target_start]
    generated = list(prefix)
    expected_length = len(item.target_suffix)
    for _ in range(expected_length):
        # Build a tiny local representation without re-parsing a fake PreparedExample.
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        event_counts = torch.tensor([[sum(event <= pos for event in item.event_positions) for pos in range(len(generated))]], dtype=torch.long, device=device)
        visual_maps = torch.full((1, model.config.canvas_height, model.config.canvas_width), ord(" "), dtype=torch.long, device=device)
        for row, text in enumerate(item.map_rows):
            visual_maps[0, row, : len(text)] = torch.tensor(list(text.encode("utf-8")), dtype=torch.long, device=device)
        events = torch.tensor([item.event_positions], dtype=torch.long, device=device)
        if model.config.pos_readout == "full_text":
            logits = model(input_ids, visual_maps=visual_maps, event_positions=events, event_counts=event_counts)
            next_byte = int(logits[0, -1].argmax().item())
        else:
            # Query-final '>' predicts the first answer byte; later positions
            # are teacher-forced from the bytes generated so far.
            target_input = torch.tensor([[prefix[-1], *generated[item.target_start:-1]]], dtype=torch.long, device=device)
            logits = model(input_ids[:, : item.target_start], visual_maps=visual_maps, event_positions=events, event_counts=event_counts[:, : item.target_start], target_input_ids=target_input)
            next_byte = int(logits[0, -1].argmax().item())
        generated.append(next_byte if next_byte <= 255 else 0)
    return bytes(generated[item.target_start:])


@torch.no_grad()
def evaluate_examples(model: VisualTransformer, examples: Sequence[PreparedExample], *, device: torch.device, batch_size: int = 64) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    target_bytes = 0
    exact = 0
    for start in range(0, len(examples), batch_size):
        subset = examples[start : start + batch_size]
        batch = collate_visual_examples(subset, context_length=model.config.context_length, canvas_height=model.config.canvas_height, canvas_width=model.config.canvas_width)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        kwargs = _model_inputs(batch, device)
        if model.config.pos_readout == "visual_only":
            logits = model(input_ids, **kwargs, target_input_ids=torch.tensor(batch["target_input_ids"], dtype=torch.long, device=device))
            labels = torch.tensor(batch["target_labels"], dtype=torch.long, device=device)
        else:
            logits = model(input_ids, **kwargs)
        active = int(labels.ne(-100).sum().item())
        loss_total += float(target_cross_entropy(logits, labels).item()) * active
        target_bytes += active
        for example in subset:
            expected = serialize_visual_example(example).target_suffix
            exact += int(greedy_target_bytes(model, example) == expected)
    return {"target_byte_nll": loss_total / target_bytes, "greedy_exact_target_accuracy": exact / len(examples)}
