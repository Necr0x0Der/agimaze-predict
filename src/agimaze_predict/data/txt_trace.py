"""Validated MAP/START + (ACT, TXT)+ trace records for the TXT baseline.

Unlike :mod:`agimaze_predict.data.prepared`, this module deliberately owns the
raw composed TXT-trace contract.  Keeping it separate prevents the established
MAP + ACT+ -> POS loaders used by per-step and seq experiments from changing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

InitialContext = Literal["MAP", "START"]
_TAG_RE = re.compile(r"<(MAP|START|ACT|TXT|END)>(.*?)</\1>", re.DOTALL)


class TxtTraceFormatError(ValueError):
    """A composed TXT trace JSONL record violates the trace contract."""


@dataclass(frozen=True)
class TxtTransition:
    action: str
    text: str


@dataclass(frozen=True)
class TxtTrace:
    """One complete trajectory, without collection metadata or terminal END."""

    map_text: str
    start_text: str
    transitions: tuple[TxtTransition, ...]


@dataclass(frozen=True)
class TxtRolloutExample:
    """A depth-limited, teacher-forced rollout with target TXT blocks marked."""

    segments: tuple[tuple[str, bool], ...]

    @property
    def text(self) -> str:
        return "\n".join(segment for segment, _ in self.segments)


def _error(path: Path, line_number: int, message: str) -> TxtTraceFormatError:
    return TxtTraceFormatError(f"{path}:{line_number}: {message}")


def _block(tag: str, value: str) -> str:
    return f"<{tag}>{value}</{tag}>"


def parse_txt_trace(path: Path, line_number: int, trace: str) -> TxtTrace:
    """Parse exactly ``MAP, START, (ACT, TXT)+, [END]`` from one trace."""

    cursor = 0
    blocks: list[tuple[str, str]] = []
    for match in _TAG_RE.finditer(trace):
        if trace[cursor : match.start()].strip():
            raise _error(path, line_number, "contains text outside recognised blocks")
        blocks.append((match.group(1), match.group(2)))
        cursor = match.end()
    if trace[cursor:].strip():
        raise _error(path, line_number, "contains text outside recognised blocks")
    if not blocks:
        raise _error(path, line_number, "trace contains no recognised blocks")

    expected: str | None = "MAP"
    map_text: str | None = None
    start_text: str | None = None
    transitions: list[TxtTransition] = []
    pending_action: str | None = None
    saw_end = False

    for tag, value in blocks:
        if saw_end:
            raise _error(path, line_number, f"{tag} appears after END")
        if expected == "MAP":
            if tag != "MAP":
                raise _error(path, line_number, "trace must begin with exactly one MAP block")
            if not value.strip():
                raise _error(path, line_number, "MAP is empty")
            map_text = value
            expected = "START"
            continue
        if expected == "START":
            if tag != "START":
                raise _error(path, line_number, "START must immediately follow MAP")
            if not value.strip():
                raise _error(path, line_number, "START is empty")
            start_text = value
            expected = "ACT"
            continue
        if expected == "ACT":
            if tag == "END":
                if not transitions:
                    raise _error(path, line_number, "END appears before a complete ACT/TXT transition")
                saw_end = True
                continue
            if tag != "ACT":
                raise _error(path, line_number, "TXT must be preceded by ACT")
            if not value.strip():
                raise _error(path, line_number, "ACT is empty")
            pending_action = value
            expected = "TXT"
            continue

        # expected == "TXT"
        if tag != "TXT":
            raise _error(path, line_number, "ACT must be followed by TXT")
        if not value.strip():
            raise _error(path, line_number, "TXT is empty")
        assert pending_action is not None
        transitions.append(TxtTransition(action=pending_action, text=value))
        pending_action = None
        expected = "ACT"

    if expected == "TXT":
        raise _error(path, line_number, "final ACT has no TXT block")
    if map_text is None or start_text is None or not transitions:
        raise _error(path, line_number, "trace needs MAP, START, and at least one ACT/TXT transition")
    return TxtTrace(map_text=map_text, start_text=start_text, transitions=tuple(transitions))


def rollout_from_trace(
    trace: TxtTrace, *, initial_context: InitialContext, depth: int
) -> TxtRolloutExample:
    """Select a MAP or START prefix and the first ``depth`` teacher-forced pairs."""

    if initial_context not in {"MAP", "START"}:
        raise ValueError("initial_context must be 'MAP' or 'START'")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    initial = _block("MAP", trace.map_text) if initial_context == "MAP" else _block("START", trace.start_text)
    segments: list[tuple[str, bool]] = [(initial, False)]
    for transition in trace.transitions[:depth]:
        segments.append((_block("ACT", transition.action), False))
        segments.append((_block("TXT", transition.text), True))
    return TxtRolloutExample(segments=tuple(segments))


def read_txt_trace_jsonl(path: str | Path) -> list[TxtTrace]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TXT trace dataset does not exist: {dataset_path}")
    traces: list[TxtTrace] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error(dataset_path, line_number, f"invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict) or set(record) != {"trace"}:
                raise _error(dataset_path, line_number, "record must have exactly the key 'trace'")
            if not isinstance(record["trace"], str):
                raise _error(dataset_path, line_number, "'trace' must be a string")
            traces.append(parse_txt_trace(dataset_path, line_number, record["trace"]))
    if not traces:
        raise TxtTraceFormatError(f"{dataset_path}: dataset contains no records")
    return traces


class TxtTraceDataset(Sequence[TxtRolloutExample]):
    """Depth-limited TXT rollouts ready for a TXT-specific byte collator."""

    def __init__(self, path: str | Path, *, initial_context: InitialContext, depth: int) -> None:
        self.path = Path(path)
        self._examples = [
            rollout_from_trace(trace, initial_context=initial_context, depth=depth)
            for trace in read_txt_trace_jsonl(self.path)
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TxtRolloutExample:
        return self._examples[index]

    def __iter__(self) -> Iterator[TxtRolloutExample]:
        return iter(self._examples)
