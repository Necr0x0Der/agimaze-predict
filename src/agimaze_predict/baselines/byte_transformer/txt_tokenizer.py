"""Byte serialization and sparse TXT-only loss masks for trace rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agimaze_predict.data.txt_trace import TxtRolloutExample

from .tokenizer import IGNORE_INDEX, PAD_TOKEN_ID


@dataclass(frozen=True)
class SerializedTxtRollout:
    token_ids: list[int]
    target_byte_indices: frozenset[int]


def serialize_txt_rollout(example: TxtRolloutExample) -> SerializedTxtRollout:
    """Serialize newline-separated segments and mark every byte in TXT blocks.

    The whole ``<TXT>...</TXT>`` literal block is a target, so the model learns
    both the content and its unambiguous boundaries.  ACT and initial-context
    bytes are always excluded from loss.
    """

    token_ids: list[int] = []
    target_byte_indices: set[int] = set()
    for index, (segment, is_target) in enumerate(example.segments):
        if index:
            token_ids.append(ord("\n"))
        start = len(token_ids)
        token_ids.extend(segment.encode("utf-8"))
        if is_target:
            target_byte_indices.update(range(start, len(token_ids)))
    return SerializedTxtRollout(token_ids=token_ids, target_byte_indices=frozenset(target_byte_indices))


def collate_txt_rollouts(
    examples: Sequence[TxtRolloutExample], *, context_length: int
) -> dict[str, list[list[int]]]:
    """Build causal arrays whose labels are active only for TXT block bytes."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")

    serialized = [serialize_txt_rollout(example) for example in examples]
    longest = max(len(item.token_ids) for item in serialized)
    if longest > context_length:
        raise ValueError(
            f"serialized TXT rollout length {longest} exceeds context_length {context_length}; "
            "increase context_length or decrease depth"
        )

    width = longest - 1
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    for item in serialized:
        ids = item.token_ids
        row_input = ids[:-1] + [PAD_TOKEN_ID] * (width - (len(ids) - 1))
        row_labels = [IGNORE_INDEX] * width
        # Label index i predicts token_ids[i + 1]. The initial '<' of each TXT
        # block is therefore supervised at its preceding ACT/newline byte.
        for target_index in item.target_byte_indices:
            if target_index:
                row_labels[target_index - 1] = ids[target_index]
        input_ids.append(row_input)
        labels.append(row_labels)
    return {"input_ids": input_ids, "labels": labels}
