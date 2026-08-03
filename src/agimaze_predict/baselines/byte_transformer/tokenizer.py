"""Byte-level serialization and batch construction for the first baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agimaze_predict.data.prepared import PreparedExample

BYTE_VOCAB_SIZE = 256
PAD_TOKEN_ID = 256
VOCAB_SIZE = 257
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SerializedExample:
    """UTF-8 bytes for one causal example and the first target-byte offset."""

    token_ids: list[int]
    target_start: int


def serialize_example(example: PreparedExample) -> SerializedExample:
    """Serialize one record as ``input + newline + target`` UTF-8 bytes.

    The target's first byte is the first byte of ``<POS>``.  No separate BOS/EOS
    tokens are used: the baseline's vocabulary consists of literal byte values,
    plus a single PAD token used only for batching.
    """

    prefix = (example.input + "\n").encode("utf-8")
    target = example.target.encode("utf-8")
    return SerializedExample(token_ids=[*prefix, *target], target_start=len(prefix))


def collate_byte_examples(
    examples: Sequence[PreparedExample],
    *,
    context_length: int,
) -> dict[str, list[list[int]]]:
    """Create padded causal input/label arrays with loss only on target bytes.

    ``input_ids[i]`` predicts ``labels[i]``.  Labels corresponding to context,
    padding, or positions after the target use ``IGNORE_INDEX``.  The first
    target byte is therefore predicted from the last prefix byte, as required by
    causal language modelling.
    """

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")

    serialized = [serialize_example(example) for example in examples]
    lengths = [len(item.token_ids) for item in serialized]
    longest = max(lengths)
    if longest > context_length:
        raise ValueError(
            f"serialized example length {longest} exceeds context_length {context_length}; "
            "increase --context-length"
        )

    # The model consumes every byte except the final byte and predicts every
    # following byte.  Pad to the longest sequence in this batch, not the model
    # context length, to avoid unnecessary compute.
    width = longest - 1
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []

    for item in serialized:
        ids = item.token_ids
        row_input = ids[:-1] + [PAD_TOKEN_ID] * (width - (len(ids) - 1))
        row_labels = [IGNORE_INDEX] * width
        # label index i predicts ids[i + 1], so target begins at target_start - 1.
        for index in range(item.target_start - 1, len(ids) - 1):
            row_labels[index] = ids[index + 1]
        input_ids.append(row_input)
        labels.append(row_labels)

    return {"input_ids": input_ids, "labels": labels}
