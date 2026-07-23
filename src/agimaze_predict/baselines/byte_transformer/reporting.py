"""Human-readable reports for byte-Transformer predictions.

This module deliberately has no PyTorch dependency so its formatting logic stays
unit-testable in minimal installations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agimaze_predict.data.per_step import PerStepExample


@dataclass(frozen=True)
class GreedyPrediction:
    """One fully decoded target, retained for human-readable error reports."""

    example: PerStepExample
    predicted: bytes

    @property
    def expected(self) -> bytes:
        return self.example.target.encode("utf-8")

    @property
    def is_correct(self) -> bool:
        return self.predicted == self.expected


def _display_bytes(value: bytes) -> str:
    """Render malformed UTF-8 unambiguously, without losing valid text."""

    return value.decode("utf-8", errors="backslashreplace")


def _first_difference(expected: bytes, predicted: bytes) -> int | None:
    for index, (expected_byte, predicted_byte) in enumerate(zip(expected, predicted)):
        if expected_byte != predicted_byte:
            return index
    if len(expected) != len(predicted):
        return min(len(expected), len(predicted))
    return None


def write_greedy_error_log(path: str | Path, predictions: Sequence[GreedyPrediction]) -> int:
    """Write every incorrect greedy decode as a readable, self-contained record.

    Each record preserves the model input MAP and ACT, then shows expected and
    generated POS spans. This is intentionally plain UTF-8 text, so a person can
    inspect errors with ordinary editor/search tools.
    """

    output = Path(path)
    errors = [prediction for prediction in predictions if not prediction.is_correct]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# greedy POS errors: {len(errors)} / {len(predictions)} examples\n")
        for number, prediction in enumerate(errors, start=1):
            expected = prediction.expected
            generated = prediction.predicted
            first_difference = _first_difference(expected, generated)
            stream.write(f"\n{'=' * 80}\n")
            stream.write(f"ERROR {number}\n")
            stream.write(f"first_different_byte: {first_difference}\n")
            stream.write("<INPUT>\n")
            stream.write(prediction.example.input)
            stream.write("\n</INPUT>\n")
            stream.write(f"expected:  {_display_bytes(expected)}\n")
            stream.write(f"predicted: {_display_bytes(generated)}\n")
    return len(errors)
