"""Separate MAP-grid / text-byte serialization for the visual-memory model.

The rendered ``<MAP>`` block never enters the byte Transformer.  It is parsed as
one fixed-size 2-D character canvas, while the text stream receives only the
requested ACT blocks followed by a teacher-forced ``<POS>`` query prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from agimaze_predict.data.prepared import PreparedExample

BYTE_VOCAB_SIZE = 256
PAD_TOKEN_ID = 256
VOCAB_SIZE = 257
IGNORE_INDEX = -100
_MAP_RE = re.compile(r"\A<MAP>(.*?)</MAP>(.*)\Z", re.DOTALL)
_ACT_CLOSE = b"</ACT>"
_POS_OPEN = b"<POS>"


@dataclass(frozen=True)
class SerializedVisualExample:
    """One separated initial canvas and byte target-query sequence."""

    map_rows: tuple[str, ...]
    token_ids: list[int]
    target_start: int
    event_positions: tuple[int, ...]
    target_suffix: bytes


def _map_rows(model_input: str) -> tuple[str, ...]:
    match = _MAP_RE.fullmatch(model_input)
    if match is None:  # Shared loader has already checked this; keep local errors useful.
        raise ValueError("input must start with one <MAP>...</MAP> block")
    rendered = match.group(1)
    if rendered.startswith("\n"):
        rendered = rendered[1:]
    if rendered.endswith("\n"):
        rendered = rendered[:-1]
    rows = tuple(rendered.split("\n"))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("rendered MAP must be a non-empty rectangular character grid")
    return rows


def serialize_visual_example(example: PreparedExample) -> SerializedVisualExample:
    """Detach MAP, append the POS query prompt, and target only its answer.

    The literal opening ``<POS>`` tag is a known query, not a prediction target.
    Thus the target suffix starts with ``("`, while the original target is
    recoverable as ``b'<POS>' + target_suffix``.
    """

    match = _MAP_RE.fullmatch(example.input)
    if match is None:
        raise ValueError("input must start with one <MAP>...</MAP> block")
    actions = match.group(2).lstrip("\r\n")
    if not actions:
        raise ValueError("input has no ACT text after MAP")
    prefix = actions.encode("utf-8") + b"\n" + _POS_OPEN
    target = example.target.encode("utf-8")
    if not target.startswith(_POS_OPEN):
        raise ValueError("target must begin with <POS>")
    target_suffix = target[len(_POS_OPEN) :]
    event_positions: list[int] = []
    offset = 0
    while True:
        close_at = prefix.find(_ACT_CLOSE, offset)
        if close_at < 0:
            break
        event_positions.append(close_at + len(_ACT_CLOSE) - 1)
        offset = close_at + len(_ACT_CLOSE)
    if not event_positions:
        raise ValueError("input needs at least one complete ACT block")
    return SerializedVisualExample(
        map_rows=_map_rows(example.input),
        token_ids=[*prefix, *target_suffix],
        target_start=len(prefix),
        event_positions=tuple(event_positions),
        target_suffix=target_suffix,
    )


def _canvas(rows: tuple[str, ...], *, height: int, width: int, blank_char: int) -> list[list[int]]:
    if height <= 0 or width <= 0:
        raise ValueError("canvas dimensions must be positive")
    if len(rows) > height or len(rows[0]) > width:
        raise ValueError(
            f"rendered MAP {len(rows)}x{len(rows[0])} does not fit visual canvas {height}x{width}"
        )
    canvas = [[blank_char] * width for _ in range(height)]
    for row_index, row in enumerate(rows):
        for column_index, character in enumerate(row):
            char_id = ord(character)
            if char_id >= BYTE_VOCAB_SIZE:
                raise ValueError(
                    f"MAP character {character!r} is outside the current byte-sized visual vocabulary"
                )
            canvas[row_index][column_index] = char_id
    return canvas


def collate_visual_examples(
    examples: Sequence[PreparedExample], *, context_length: int, canvas_height: int, canvas_width: int
) -> dict[str, list[list[int]]]:
    """Build target-only byte LM tensors plus event-aligned visual inputs."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    serialized = [serialize_visual_example(example) for example in examples]
    longest = max(len(item.token_ids) for item in serialized)
    if longest > context_length:
        raise ValueError(
            f"serialized text length {longest} exceeds context_length {context_length}; increase it"
        )
    width = longest - 1
    max_events = max(len(item.event_positions) for item in serialized)
    max_target = max(len(item.target_suffix) for item in serialized)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    event_positions: list[list[int]] = []
    event_counts: list[list[int]] = []
    source_lengths: list[int] = []
    target_input_ids: list[list[int]] = []
    target_labels: list[list[int]] = []
    visual_maps: list[list[list[int]]] = []

    for item in serialized:
        ids = item.token_ids
        text_input = ids[:-1]
        input_ids.append(text_input + [PAD_TOKEN_ID] * (width - len(text_input)))
        row_labels = [IGNORE_INDEX] * width
        # ``target_start - 1`` is the literal '>' closing known <POS>; it predicts '('.
        for index in range(item.target_start - 1, len(ids) - 1):
            row_labels[index] = ids[index + 1]
        labels.append(row_labels)
        event_positions.append([*item.event_positions, *([-1] * (max_events - len(item.event_positions)))])
        counts = []
        for position in range(width):
            counts.append(sum(event <= position for event in item.event_positions))
        event_counts.append(counts)
        source_lengths.append(item.target_start)
        # The visual-only decoder begins with the known query-final '>' and gets
        # teacher-forced prior answer bytes, but no action bytes.
        target_input = [ids[item.target_start - 1], *item.target_suffix[:-1]]
        # Model input positions predict their *next* byte. The known query-final
        # '>' therefore predicts the first answer byte '(' at index zero.
        target_input_ids.append(target_input + [PAD_TOKEN_ID] * (max_target - len(target_input)))
        target_labels.append([*item.target_suffix, *([IGNORE_INDEX] * (max_target - len(item.target_suffix)))])
        visual_maps.append(
            _canvas(item.map_rows, height=canvas_height, width=canvas_width, blank_char=ord(" "))
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "visual_maps": visual_maps,
        "event_positions": event_positions,
        "event_counts": event_counts,
        "source_lengths": source_lengths,
        "target_input_ids": target_input_ids,
        "target_labels": target_labels,
    }
