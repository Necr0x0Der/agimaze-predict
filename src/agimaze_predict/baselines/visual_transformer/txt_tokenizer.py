"""TXT-rollout serialization for the event-triggered visual-memory model.

The rendered MAP is exclusively the initial visual frame.  A rollout then
contains text only through one completed ACT; its following full ``<TXT>``
block is the target.  Earlier ground-truth TXT blocks remain in the source
when a later transition is predicted, exactly as in the byte-Transformer TXT
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agimaze_predict.data.txt_trace import InitialContext, TxtTrace

from .tokenizer import IGNORE_INDEX, PAD_TOKEN_ID, _canvas, _map_rows

_ACT_CLOSE = b"</ACT>"


@dataclass(frozen=True)
class VisualTxtRollout:
    """One action-conditioned TXT prediction from a complete trajectory."""

    map_rows: tuple[str, ...]
    source: bytes
    target: bytes


@dataclass(frozen=True)
class VisualTxtTraceRollout:
    """All depth-limited transition targets from one trace."""

    rollouts: tuple[VisualTxtRollout, ...]


def _block(tag: str, value: str) -> str:
    return f"<{tag}>{value}</{tag}>"


def visual_txt_rollouts_from_trace(
    trace: TxtTrace, *, initial_context: InitialContext, depth: int
) -> VisualTxtTraceRollout:
    """Expand a trace into one target per action, retaining prior TXT feedback.

    ``initial_context='MAP'`` means MAP is supplied only as the visual canvas;
    ``START`` additionally passes the textual START block to the action stream.
    In both modes the true rendered MAP initializes visual memory, so the
    comparison changes only the textual initial context as it does for the byte
    baseline.
    """

    if initial_context not in {"MAP", "START"}:
        raise ValueError("initial_context must be 'MAP' or 'START'")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    source_segments: list[str] = []
    if initial_context == "START":
        source_segments.append(_block("START", trace.start_text))
    rollouts: list[VisualTxtRollout] = []
    for transition in trace.transitions[:depth]:
        source = "\n".join([*source_segments, _block("ACT", transition.action)]).encode("utf-8")
        rollouts.append(
            VisualTxtRollout(
                map_rows=_map_rows(_block("MAP", trace.map_text)),
                source=source,
                target=_block("TXT", transition.text).encode("utf-8"),
            )
        )
        source_segments.extend((_block("ACT", transition.action), _block("TXT", transition.text)))
    return VisualTxtTraceRollout(rollouts=tuple(rollouts))


def load_visual_txt_traces(
    paths: Sequence[str], *, initial_context: InitialContext, depth: int
) -> list[VisualTxtTraceRollout]:
    """Load and expand paths without changing the shared TXT trace parser."""

    from agimaze_predict.data.txt_trace import read_txt_trace_jsonl

    traces: list[VisualTxtTraceRollout] = []
    for path in paths:
        traces.extend(
            visual_txt_rollouts_from_trace(trace, initial_context=initial_context, depth=depth)
            for trace in read_txt_trace_jsonl(path)
        )
    return traces


def flatten_visual_txt_traces(traces: Sequence[VisualTxtTraceRollout]) -> list[VisualTxtRollout]:
    return [rollout for trace in traces for rollout in trace.rollouts]


def _event_positions(source: bytes) -> tuple[int, ...]:
    positions: list[int] = []
    offset = 0
    while True:
        close_at = source.find(_ACT_CLOSE, offset)
        if close_at < 0:
            return tuple(positions)
        positions.append(close_at + len(_ACT_CLOSE) - 1)
        offset = close_at + len(_ACT_CLOSE)


def collate_visual_txt_rollouts(
    examples: Sequence[VisualTxtRollout], *, context_length: int, canvas_height: int, canvas_width: int
) -> dict[str, list[list[int]]]:
    """Create TXT-only targets plus event-indexed visual inputs.

    The visual-only decoder starts on the final byte of ``</ACT>`` and predicts
    the complete ``<TXT>...</TXT>`` block.  Thus it has no direct action-token
    input while still learning the same byte target as the full-text control.
    """

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    serialized: list[tuple[VisualTxtRollout, tuple[int, ...], list[int]]] = []
    for item in examples:
        events = _event_positions(item.source)
        if not events:
            raise ValueError("TXT visual rollout requires one completed ACT block")
        ids = [*item.source, *item.target]
        if len(ids) > context_length:
            raise ValueError(
                f"serialized TXT visual rollout length {len(ids)} exceeds context_length {context_length}; "
                "increase context_length or decrease depth"
            )
        serialized.append((item, events, ids))

    longest = max(len(ids) for _, _, ids in serialized)
    width = longest - 1
    max_events = max(len(events) for _, events, _ in serialized)
    max_target = max(len(item.target) for item, _, _ in serialized)
    result: dict[str, list[list[int]]] = {
        "input_ids": [], "labels": [], "visual_maps": [], "event_positions": [],
        "event_counts": [], "target_input_ids": [], "target_labels": [],
    }
    for item, events, ids in serialized:
        source_length = len(item.source)
        text_input = ids[:-1]
        result["input_ids"].append(text_input + [PAD_TOKEN_ID] * (width - len(text_input)))
        labels = [IGNORE_INDEX] * width
        for index in range(source_length - 1, len(ids) - 1):
            labels[index] = ids[index + 1]
        result["labels"].append(labels)
        result["event_positions"].append([*events, *([-1] * (max_events - len(events)))])
        result["event_counts"].append([sum(event <= position for event in events) for position in range(width)])
        target_input = [item.source[-1], *item.target[:-1]]
        result["target_input_ids"].append(target_input + [PAD_TOKEN_ID] * (max_target - len(target_input)))
        result["target_labels"].append([*item.target, *([IGNORE_INDEX] * (max_target - len(item.target)))])
        result["visual_maps"].append(
            _canvas(item.map_rows, height=canvas_height, width=canvas_width, blank_char=ord(" "))
        )
    return result
