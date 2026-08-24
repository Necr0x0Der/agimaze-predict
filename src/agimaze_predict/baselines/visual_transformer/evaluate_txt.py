"""Teacher-forced TXT rollout evaluation for the visual-memory Transformer."""

from __future__ import annotations

import torch

from .model import VisualTransformer, target_cross_entropy
from .txt_tokenizer import (
    VisualTxtRollout,
    VisualTxtTraceRollout,
    _event_positions,
    collate_visual_txt_rollouts,
    flatten_visual_txt_traces,
)


def _visual_inputs(model: VisualTransformer, rollout: VisualTxtRollout, source: bytes) -> dict[str, torch.Tensor]:
    """Build unpadded event-aligned inputs for one greedy TXT prediction."""

    if len(source) > model.config.context_length:
        raise ValueError("generation source exceeds context_length")
    device = next(model.parameters()).device
    events = _event_positions(source)
    if not events:
        raise ValueError("TXT visual rollout requires one completed ACT block")
    visual_map = torch.full(
        (1, model.config.canvas_height, model.config.canvas_width), ord(" "), dtype=torch.long, device=device
    )
    for row, text in enumerate(rollout.map_rows):
        visual_map[0, row, : len(text)] = torch.tensor(list(text.encode("utf-8")), dtype=torch.long, device=device)
    return {
        "visual_maps": visual_map,
        "event_positions": torch.tensor([events], dtype=torch.long, device=device),
        "event_counts": torch.tensor(
            [[sum(event <= position for event in events) for position in range(len(source))]],
            dtype=torch.long,
            device=device,
        ),
    }


@torch.no_grad()
def greedy_txt_span(model: VisualTransformer, rollout: VisualTxtRollout) -> bytes:
    """Decode one complete TXT span from an ACT-conditioned visual frame."""

    model.eval()
    device = next(model.parameters()).device
    generated: list[int] = []
    for _ in range(len(rollout.target)):
        if model.config.pos_readout == "full_text":
            source = rollout.source + bytes(generated)
            kwargs = _visual_inputs(model, rollout, source)
            input_ids = torch.tensor([source], dtype=torch.long, device=device)
            next_byte = int(model(input_ids, **kwargs)[0, -1].argmax().item())
        else:
            target_input = torch.tensor([[rollout.source[-1], *generated]], dtype=torch.long, device=device)
            # The target decoder reads the final frame after the completed ACT.
            # Its source stays fixed: prior target bytes enter only its own causal decoder.
            source_kwargs = _visual_inputs(model, rollout, rollout.source)
            next_byte = int(
                model(
                    torch.tensor([rollout.source], dtype=torch.long, device=device),
                    **source_kwargs,
                    target_input_ids=target_input,
                )[0, -1].argmax().item()
            )
        generated.append(next_byte if next_byte <= 255 else 0)
    return bytes(generated)


@torch.no_grad()
def evaluate_visual_txt_rollouts(
    model: VisualTransformer,
    traces: list[VisualTxtTraceRollout],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Report TXT-only NLL and teacher-forced greedy span/trace exactness."""

    if not traces:
        raise ValueError("cannot evaluate an empty trace collection")
    model.eval()
    rollouts = flatten_visual_txt_traces(traces)
    total_loss, total_bytes = 0.0, 0
    for start in range(0, len(rollouts), batch_size):
        batch = collate_visual_txt_rollouts(
            rollouts[start : start + batch_size],
            context_length=model.config.context_length,
            canvas_height=model.config.canvas_height,
            canvas_width=model.config.canvas_width,
        )
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        kwargs = {
            key: torch.tensor(batch[key], dtype=torch.long, device=device)
            for key in ("visual_maps", "event_positions", "event_counts")
        }
        if model.config.pos_readout == "full_text":
            logits, labels = model(input_ids, **kwargs), torch.tensor(batch["labels"], dtype=torch.long, device=device)
        else:
            logits = model(input_ids, **kwargs, target_input_ids=torch.tensor(batch["target_input_ids"], dtype=torch.long, device=device))
            labels = torch.tensor(batch["target_labels"], dtype=torch.long, device=device)
        active = int(labels.ne(-100).sum().item())
        total_loss += float(target_cross_entropy(logits, labels).item()) * active
        total_bytes += active

    exact_spans = 0
    exact_traces = 0
    for trace in traces:
        all_exact = True
        for rollout in trace.rollouts:
            is_exact = greedy_txt_span(model, rollout) == rollout.target
            exact_spans += int(is_exact)
            all_exact = all_exact and is_exact
        exact_traces += int(all_exact)
    return {
        "txt_byte_nll": total_loss / total_bytes,
        "txt_span_exact_accuracy": exact_spans / len(rollouts),
        "trace_all_txt_exact_accuracy": exact_traces / len(traces),
    }
