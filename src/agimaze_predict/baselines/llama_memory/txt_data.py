"""Event-aligned TXT-rollout serialization for Llama spatial memory.

Every completed ACT receives a workspace readout.  The readout is inserted
between that ACT and its TXT observation, so a later teacher-forced TXT remains
ordinary language context but cannot be confused with a memory write event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from agimaze_predict.data.txt_trace import InitialContext, TxtTrace, read_txt_trace_jsonl

from .data import _canvas
from .map_format import map_rows

_IGNORE_INDEX = -100


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> object: ...


def _block(tag: str, value: str) -> str:
    return f"<{tag}>{value}</{tag}>"


def _ids(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = getattr(encoded, "input_ids", None)
    if values is None and isinstance(encoded, dict):
        values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


@dataclass(frozen=True)
class LlamaTxtRollout:
    """One supervised TXT prediction with all completed ACT events in source."""

    map_rows: tuple[str, ...]
    source_segments: tuple[str, ...]
    actions: tuple[str, ...]
    target: str


def txt_rollouts_from_trace(
    trace: TxtTrace, *, initial_context: InitialContext, depth: int
) -> list[LlamaTxtRollout]:
    """Expand one trace into teacher-forced per-transition training records."""

    if initial_context not in {"MAP", "START"}:
        raise ValueError("initial_context must be 'MAP' or 'START'")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    source_segments = [] if initial_context == "MAP" else [_block("START", trace.start_text)]
    actions: list[str] = []
    result: list[LlamaTxtRollout] = []
    rendered_map = map_rows(trace.map_text)
    for transition in trace.transitions[:depth]:
        action_block = _block("ACT", transition.action)
        actions.append(transition.action)
        result.append(LlamaTxtRollout(
            map_rows=rendered_map,
            source_segments=tuple([*source_segments, action_block]),
            actions=tuple(actions),
            target=_block("TXT", transition.text),
        ))
        source_segments.extend((action_block, _block("TXT", transition.text)))
    return result


def load_llama_txt_rollouts(
    paths: Sequence[str], *, initial_context: InitialContext, depth: int
) -> list[LlamaTxtRollout]:
    result: list[LlamaTxtRollout] = []
    for path in paths:
        for trace in read_txt_trace_jsonl(path):
            result.extend(txt_rollouts_from_trace(trace, initial_context=initial_context, depth=depth))
    if not result:
        raise ValueError("no TXT rollouts loaded")
    return result


def collate_txt_rollouts(
    examples: Sequence[LlamaTxtRollout], *, tokenizer: Tokenizer, canvas_height: int, canvas_width: int, pad_token_id: int
) -> dict[str, list]:
    """Tokenize event-segmented Llama TXT records with target-only labels.

    ``event_token_positions`` stores the token position immediately following
    each ACT block.  Model code inserts the corresponding memory sequence at
    that boundary.  The final event is immediately before the target TXT.
    """

    if not examples:
        raise ValueError("cannot collate an empty TXT rollout batch")
    serialized: list[tuple[LlamaTxtRollout, list[int], list[int], list[int], list[list[int]]]] = []
    for item in examples:
        if len(item.source_segments) < 1 or len(item.actions) < 1:
            raise ValueError("TXT rollout requires one completed ACT")
        source: list[int] = []
        event_positions: list[int] = []
        for index, segment in enumerate(item.source_segments):
            segment_ids = _ids(tokenizer, segment + "\n")
            if not segment_ids:
                raise ValueError("TXT source segment must produce tokenizer tokens")
            source.extend(segment_ids)
            if segment.startswith("<ACT>"):
                event_positions.append(len(source))
        target = _ids(tokenizer, item.target)
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        target.append(tokenizer.eos_token_id)
        action_ids = [_ids(tokenizer, action) for action in item.actions]
        if any(not ids for ids in action_ids):
            raise ValueError("each ACT value must produce tokenizer tokens")
        if len(event_positions) != len(action_ids):
            raise ValueError("source ACT boundaries and action history disagree")
        serialized.append((item, source, target, event_positions, action_ids))

    max_total = max(len(source) + len(target) for _, source, target, _, _ in serialized)
    max_events = max(len(events) for _, _, _, events, _ in serialized)
    max_action_width = max(len(ids) for _, _, _, _, actions in serialized for ids in actions)
    result: dict[str, list] = {
        "input_ids": [], "attention_mask": [], "labels": [], "prompt_lengths": [],
        "event_token_positions": [], "event_mask": [], "visual_maps": [],
        "action_input_ids": [], "action_attention_mask": [], "target_token_ids": [],
    }
    for item, source, target, events, actions in serialized:
        ids = [*source, *target]
        padding = max_total - len(ids)
        result["input_ids"].append([*ids, *([pad_token_id] * padding)])
        result["attention_mask"].append([1] * len(ids) + [0] * padding)
        result["labels"].append([_IGNORE_INDEX] * len(source) + target + [_IGNORE_INDEX] * padding)
        result["prompt_lengths"].append(len(source))
        result["event_token_positions"].append([*events, *([-1] * (max_events - len(events)))])
        result["event_mask"].append([1] * len(events) + [0] * (max_events - len(events)))
        action_input = [ids + [pad_token_id] * (max_action_width - len(ids)) for ids in actions]
        action_mask = [[1] * len(ids) + [0] * (max_action_width - len(ids)) for ids in actions]
        action_input += [[pad_token_id] * max_action_width] * (max_events - len(actions))
        action_mask += [[0] * max_action_width] * (max_events - len(actions))
        result["action_input_ids"].append(action_input)
        result["action_attention_mask"].append(action_mask)
        result["visual_maps"].append(_canvas(item.map_rows, canvas_height, canvas_width))
        result["target_token_ids"].append(target)
    return result
