"""Backward-compatible aliases for the common prepared-data contract.

New code should import :mod:`agimaze_predict.data.prepared`.  A per-step record
is exactly the one-action instance of that common MAP + ACT+ -> POS contract.
These aliases keep existing imports and checkpoints' surrounding scripts stable.
"""

from .prepared import (
    PreparedDatasetFormatError,
    PreparedExample,
    PreparedMapActionsToPosDataset,
    collate_prepared_examples,
    read_prepared_jsonl,
)

PerStepDatasetFormatError = PreparedDatasetFormatError
PerStepExample = PreparedExample
PerStepMapActToPosDataset = PreparedMapActionsToPosDataset
collate_per_step_examples = collate_prepared_examples
read_per_step_jsonl = read_prepared_jsonl

__all__ = [
    "PerStepDatasetFormatError",
    "PerStepExample",
    "PerStepMapActToPosDataset",
    "collate_per_step_examples",
    "read_per_step_jsonl",
]
