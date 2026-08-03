"""Loader for prepared AGI Maze MAP + ACT+ -> POS JSONL datasets.

This module owns the common prepared-data contract used by the baselines.  It is
independent of raw trace collection and model/tokenizer implementations.  A
record has one rendered initial map, one or more requested actions, and one
position target.  ``per_step`` is the one-action special case; fixed-horizon
``seq`` examples use two or more actions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

_INPUT_RE = re.compile(r"\A<MAP>.*?</MAP>(?:\n<ACT>\S(?:.*?\S)?</ACT>)+\Z", re.DOTALL)
_TARGET_RE = re.compile(r"\A<POS>\([^\r\n<>]+\)</POS>\Z")


class PreparedDatasetFormatError(ValueError):
    """A prepared MAP + ACT+ -> POS JSONL record violates the common contract."""


@dataclass(frozen=True)
class PreparedExample:
    """One map-conditioned, action-sequence position-prediction example."""

    input: str
    target: str


def _format_error(path: Path, line_number: int, message: str) -> PreparedDatasetFormatError:
    return PreparedDatasetFormatError(f"{path}:{line_number}: {message}")


def _parse_record(path: Path, line_number: int, line: str) -> PreparedExample:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise _format_error(path, line_number, f"invalid JSON: {exc.msg}") from exc

    if not isinstance(record, dict) or set(record) != {"input", "target"}:
        raise _format_error(path, line_number, "record must have exactly the keys 'input' and 'target'")

    model_input, target = record["input"], record["target"]
    if not isinstance(model_input, str) or not isinstance(target, str):
        raise _format_error(path, line_number, "'input' and 'target' must both be strings")
    if not _INPUT_RE.fullmatch(model_input):
        raise _format_error(
            path,
            line_number,
            "'input' must be one <MAP>...</MAP> block followed by one or more "
            "newline-separated <ACT>...</ACT> blocks",
        )
    if not _TARGET_RE.fullmatch(target):
        raise _format_error(path, line_number, "'target' must be one <POS>(row, col)</POS> block")

    return PreparedExample(input=model_input, target=target)


def read_prepared_jsonl(path: str | Path) -> list[PreparedExample]:
    """Read and validate every non-empty record in a prepared JSONL file."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset does not exist: {dataset_path}")

    examples: list[PreparedExample] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if line:
                examples.append(_parse_record(dataset_path, line_number, line))

    if not examples:
        raise PreparedDatasetFormatError(f"{dataset_path}: dataset contains no records")
    return examples


class PreparedMapActionsToPosDataset(Sequence[PreparedExample]):
    """Indexable common prepared dataset suitable for PyTorch's ``DataLoader``.

    It returns text-level records.  Models own their tokenization and batching,
    so no model-specific representation leaks into the shared data layer.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._examples = read_prepared_jsonl(self.path)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> PreparedExample:
        return self._examples[index]

    def __iter__(self) -> Iterator[PreparedExample]:
        return iter(self._examples)


def collate_prepared_examples(examples: Sequence[PreparedExample]) -> dict[str, list[str]]:
    """Minimal model-neutral DataLoader collator preserving text records."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    return {
        "inputs": [example.input for example in examples],
        "targets": [example.target for example in examples],
    }
