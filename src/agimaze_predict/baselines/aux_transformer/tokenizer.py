"""Auxiliary-model serialization and target-only causal batch construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

from agimaze_predict.data.prepared import PreparedExample

BYTE_VOCAB_SIZE = 256
PAD_TOKEN_ID = 256
VOCAB_SIZE = 257
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SerializedAuxExample:
    """One literal byte sequence, the first target offset, and aux source length."""

    token_ids: list[int]
    target_start: int


def serialize_aux_example(example: PreparedExample) -> SerializedAuxExample:
    """Serialize source + newline + target without any in-context state slots."""

    prefix = list(example.input.encode("utf-8"))
    prefix.append(ord("\n"))
    target = list(example.target.encode("utf-8"))
    return SerializedAuxExample(token_ids=[*prefix, *target], target_start=len(prefix))


def collate_aux_examples(
    examples: Sequence[PreparedExample], *, context_length: int
) -> dict[str, list[list[int]]]:
    """Build byte LM tensors and source-only aux lengths.

    ``aux_lengths`` ends immediately before the first target byte. It is passed
    to the model's causal cross-stream masks, preventing target leakage into the
    auxiliary Transformer even under teacher forcing.
    """

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")

    serialized = [serialize_aux_example(example) for example in examples]
    longest = max(len(item.token_ids) for item in serialized)
    if longest > context_length:
        raise ValueError(
            f"serialized example length {longest} exceeds context_length {context_length}; "
            "increase --context-length"
        )

    width = longest - 1
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    aux_lengths: list[int] = []
    for item in serialized:
        ids = item.token_ids
        input_ids.append(ids[:-1] + [PAD_TOKEN_ID] * (width - (len(ids) - 1)))
        row_labels = [IGNORE_INDEX] * width
        for index in range(item.target_start - 1, len(ids) - 1):
            row_labels[index] = ids[index + 1]
        labels.append(row_labels)
        aux_lengths.append(item.target_start)

    return {"input_ids": input_ids, "labels": labels, "aux_lengths": aux_lengths}


def collate_aux_denoising_examples(
    examples: Sequence[PreparedExample],
    *,
    context_length: int,
    mask_rate: float,
    mask_span_length: int,
    generator: "torch.Generator | None" = None,
) -> dict[str, list[list[int]]]:
    """Add source-only span corruption and latent-aligned reconstruction labels.

    The corruption placeholder is PAD, which cannot occur in an unpadded byte
    sequence.  It is applied only before ``aux_lengths``; POS target bytes and
    ordinary target-LM labels are untouched.  Each latent slot clocked to a
    masked source byte receives that byte as its reconstruction label.
    """

    if not 0.0 < mask_rate < 1.0:
        raise ValueError("mask_rate must be strictly between zero and one")
    if mask_span_length <= 0:
        raise ValueError("mask_span_length must be positive")

    # Keep ordinary serialization usable without the optional PyTorch package.
    import torch

    batch = collate_aux_examples(examples, context_length=context_length)
    masked_input_ids = [row.copy() for row in batch["input_ids"]]
    # Record byte-level labels first.  The training collator expands each one
    # to its configured latent-slot count, because that count is model rather
    # than dataset serialization state.
    source_denoise_labels: list[list[int]] = []
    for row, aux_length in zip(masked_input_ids, batch["aux_lengths"], strict=True):
        source_labels = [IGNORE_INDEX] * aux_length
        target_count = max(1, round(aux_length * mask_rate))
        masked_count = 0
        # Draw starts independently and fill consecutive source positions. The
        # loop always progresses: each span wraps within the finite source and
        # stops exactly at the requested number of masked bytes.
        while masked_count < target_count:
            start = int(torch.randint(aux_length, (1,), generator=generator).item())
            for offset in range(mask_span_length):
                position = (start + offset) % aux_length
                if source_labels[position] != IGNORE_INDEX:
                    continue
                source_labels[position] = row[position]
                row[position] = PAD_TOKEN_ID
                masked_count += 1
                if masked_count == target_count:
                    break
        source_denoise_labels.append(source_labels)

    max_aux_length = max(batch["aux_lengths"])
    padded_source_labels = [
        row + [IGNORE_INDEX] * (max_aux_length - len(row)) for row in source_denoise_labels
    ]

    return {
        **batch,
        "denoise_input_ids": masked_input_ids,
        "denoise_source_labels": padded_source_labels,
    }


def expand_denoise_labels_for_latents(
    source_labels: list[list[int]], *, aux_latents_per_token: int
) -> list[list[int]]:
    """Repeat each source reconstruction target for every associated latent."""

    if aux_latents_per_token <= 0:
        raise ValueError("aux_latents_per_token must be positive")
    return [
        [label for label in row for _ in range(aux_latents_per_token)]
        for row in source_labels
    ]
