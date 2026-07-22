from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agimaze_predict.data.per_step import (
    PerStepDatasetFormatError,
    PerStepMapActToPosDataset,
    collate_per_step_examples,
    read_per_step_jsonl,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_per_step.jsonl"


class PerStepLoaderTest(unittest.TestCase):
    def test_reads_prepared_jsonl_and_exposes_sequence_dataset(self) -> None:
        examples = read_per_step_jsonl(FIXTURE)
        dataset = PerStepMapActToPosDataset(FIXTURE)

        self.assertEqual(len(examples), 2)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], examples[0])
        self.assertTrue(dataset[0].input.endswith("<ACT>right</ACT>"))
        self.assertEqual(dataset[1].target, "<POS>(0, 1)</POS>")

    def test_model_neutral_collator(self) -> None:
        batch = collate_per_step_examples(read_per_step_jsonl(FIXTURE))
        self.assertEqual(
            batch,
            {
                "inputs": [
                    "<MAP>\n+---+\n| S |\n+---+\n</MAP>\n<ACT>right</ACT>",
                    "<MAP>\n+---+---+\n| S     |\n+---+---+\n</MAP>\n<ACT>right</ACT>",
                ],
                "targets": ["<POS>(0, 0)</POS>", "<POS>(0, 1)</POS>"],
            },
        )

    def test_rejects_invalid_records(self) -> None:
        cases = [
            ({"input": "<MAP>x</MAP>\n<ACT>right</ACT>", "target": "(0, 1)"}, "target"),
            ({"input": "<MAP>x</MAP><ACT>right</ACT>", "target": "<POS>(0, 1)</POS>"}, "input"),
            ({"input": "x", "target": "y", "extra": "z"}, "exactly"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.jsonl"
            for record, message in cases:
                with self.subTest(record=record):
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(PerStepDatasetFormatError, message):
                        read_per_step_jsonl(path)


if __name__ == "__main__":
    unittest.main()
