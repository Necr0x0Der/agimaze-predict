"""Teacher-forced rollout evaluation for the TXT trace byte Transformer."""

from __future__ import annotations

import torch

from agimaze_predict.data.txt_trace import TxtRolloutExample

from .model import ByteTransformer, target_cross_entropy
from .txt_tokenizer import collate_txt_rollouts


@torch.no_grad()
def _greedy_bytes(model: ByteTransformer, prefix: bytes, *, length: int) -> bytes:
    if not prefix:
        raise ValueError("generation prefix must contain at least one byte")
    generated = list(prefix)
    for _ in range(length):
        if len(generated) > model.config.context_length:
            raise ValueError("generation prefix exceeds context_length")
        input_ids = torch.tensor([generated], dtype=torch.long, device=next(model.parameters()).device)
        token = int(model(input_ids)[0, -1].argmax().item())
        generated.append(token if token <= 255 else 0)
    return bytes(generated[len(prefix) :])


@torch.no_grad()
def evaluate_txt_rollouts(
    model: ByteTransformer,
    examples: list[TxtRolloutExample],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute TXT-only NLL and greedy spans with ground-truth TXT feedback.

    Each span is generated after the actual initial context and ACT blocks.  Its
    prediction is scored, but the known ground-truth span (not the prediction)
    is appended before the next ACT.  This exactly matches the requested
    teacher-forced rollout contract.
    """

    if not examples:
        raise ValueError("cannot evaluate an empty example collection")
    model.eval()
    total_loss = 0.0
    total_target_bytes = 0
    exact_spans = 0
    spans = 0
    exact_traces = 0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = collate_txt_rollouts(batch_examples, context_length=model.config.context_length)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        logits = model(input_ids)
        active = labels.ne(-100)
        active_bytes = int(active.sum().item())
        total_loss += float(target_cross_entropy(logits, labels).item()) * active_bytes
        total_target_bytes += active_bytes

        for example in batch_examples:
            prefix = b""
            all_exact = True
            for index, (segment, is_target) in enumerate(example.segments):
                encoded = segment.encode("utf-8")
                separator = b"" if index == 0 else b"\n"
                if not is_target:
                    prefix += separator + encoded
                    continue
                prediction = _greedy_bytes(model, prefix + separator, length=len(encoded))
                is_exact = prediction == encoded
                exact_spans += int(is_exact)
                spans += 1
                all_exact = all_exact and is_exact
                # Intentional teacher forcing for the next TXT span.
                prefix += separator + encoded
            exact_traces += int(all_exact)

    return {
        "txt_byte_nll": total_loss / total_target_bytes,
        "txt_span_exact_accuracy": exact_spans / spans,
        "trace_all_txt_exact_accuracy": exact_traces / len(examples),
    }
