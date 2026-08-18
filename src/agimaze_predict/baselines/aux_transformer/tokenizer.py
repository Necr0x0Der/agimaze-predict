"""Auxiliary-model serialization and target-only causal batch construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def collate_aux_target_examples(
    examples: Sequence[PreparedExample], *, context_length: int
) -> dict[str, list[list[int]]]:
    """Add teacher-forced target-decoder tensors to an ordinary clean batch.

    The decoder starts from the source newline immediately before ``<POS>``.
    It then receives each prior target byte and predicts the next one.  Its
    source information must therefore arrive through the aux latent stream,
    not through copied source tokens.
    """

    batch = collate_aux_examples(examples, context_length=context_length)
    serialized = [serialize_aux_example(example) for example in examples]
    target_rows = [
        (
            [item.token_ids[item.target_start - 1], *item.token_ids[item.target_start:-1]],
            item.token_ids[item.target_start:],
        )
        for item in serialized
    ]
    width = max(len(labels) for _, labels in target_rows)
    target_input_ids = [
        input_ids + [PAD_TOKEN_ID] * (width - len(input_ids)) for input_ids, _ in target_rows
    ]
    target_labels = [
        labels + [IGNORE_INDEX] * (width - len(labels)) for _, labels in target_rows
    ]
    return {
        **batch,
        "aux_target_input_ids": target_input_ids,
        "aux_target_labels": target_labels,
    }
