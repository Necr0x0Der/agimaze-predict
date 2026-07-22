"""Loader for prepared per-step MAP + ACT -> POS JSONL datasets.

This module owns the public prepared-data contract. It intentionally does not
know how raw traces are collected, and it is independent of any model/tokenizer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

_INPUT_RE = re.compile(r"\A<MAP>.*?</MAP>\n<ACT>\S(?:.*?\S)?</ACT>\Z", re.DOTALL)
_TARGET_RE = re.compile(r"\A<POS>\([^\r\n<>]+\)</POS>\Z")


class PerStepDatasetFormatError(ValueError):
    """A prepared per-step JSONL record does not satisfy the dataset contract."""


@dataclass(frozen=True)
class PerStepExample:
    """One action-conditioned transition prediction example."""

    input: str
    target: str


def _format_error(path: Path, line_number: int, message: str) -> PerStepDatasetFormatError:
    return PerStepDatasetFormatError(f"{path}:{line_number}: {message}")


def _parse_record(path: Path, line_number: int, line: str) -> PerStepExample:
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
        raise _format_error(path, line_number, "'input' must be <MAP>...</MAP> followed by '\\n<ACT>...</ACT>'")
    if not _TARGET_RE.fullmatch(target):
        raise _format_error(path, line_number, "'target' must be one <POS>(row, col)</POS> block")

    return PerStepExample(input=model_input, target=target)


def read_per_step_jsonl(path: str | Path) -> list[PerStepExample]:
    """Read and validate every non-empty record in a prepared per-step JSONL file."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared per-step dataset does not exist: {dataset_path}")

    examples: list[PerStepExample] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if line:
                examples.append(_parse_record(dataset_path, line_number, line))

    if not examples:
        raise PerStepDatasetFormatError(f"{dataset_path}: dataset contains no records")
    return examples


class PerStepMapActToPosDataset(Sequence[PerStepExample]):
    """Indexable prepared dataset suitable for PyTorch's ``DataLoader``.

    It intentionally returns text-level ``PerStepExample`` instances. Each model
    owns its tokenizer and batching policy, so no model representation leaks into
    the shared data layer.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._examples = read_per_step_jsonl(self.path)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> PerStepExample:
        return self._examples[index]

    def __iter__(self) -> Iterator[PerStepExample]:
        return iter(self._examples)


def collate_per_step_examples(examples: Sequence[PerStepExample]) -> dict[str, list[str]]:
    """Minimal model-neutral DataLoader collator preserving text records."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    return {
        "inputs": [example.input for example in examples],
        "targets": [example.target for example in examples],
    }
