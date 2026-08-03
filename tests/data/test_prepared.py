from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agimaze_predict.data.prepared import (
    PreparedDatasetFormatError,
    PreparedMapActionsToPosDataset,
    collate_prepared_examples,
    read_prepared_jsonl,
)


class PreparedLoaderTest(unittest.TestCase):
    def test_reads_fixed_horizon_sequence_example(self) -> None:
        record = {
            "input": "<MAP>initial map</MAP>\n<ACT>right</ACT>\n<ACT>down</ACT>",
            "target": "<POS>(1, 1)</POS>",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "seq.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            examples = read_prepared_jsonl(path)
            dataset = PreparedMapActionsToPosDataset(path)

        self.assertEqual(examples[0].input.count("<ACT>"), 2)
        self.assertEqual(dataset[0], examples[0])
        self.assertEqual(
            collate_prepared_examples(examples),
            {"inputs": [record["input"]], "targets": [record["target"]]},
        )

    def test_rejects_missing_action_or_non_position_target(self) -> None:
        cases = [
            ({"input": "<MAP>map</MAP>", "target": "<POS>(0, 0)</POS>"}, "one or more"),
            ({"input": "<MAP>map</MAP>\n<ACT>right</ACT>", "target": "<POS>(0, 0)</POS>\n<POS>(0, 1)</POS>"}, "target"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.jsonl"
            for record, message in cases:
                with self.subTest(record=record):
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(PreparedDatasetFormatError, message):
                        read_prepared_jsonl(path)


if __name__ == "__main__":
    unittest.main()
