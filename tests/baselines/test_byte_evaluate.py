from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agimaze_predict.baselines.byte_transformer.reporting import (
    GreedyPrediction,
    write_greedy_error_log,
)
from agimaze_predict.data.per_step import PerStepExample


class GreedyErrorLogTest(unittest.TestCase):
    def test_writes_only_wrong_predictions_with_input_and_targets(self) -> None:
        correct = GreedyPrediction(
            example=PerStepExample(
                input="<MAP>correct-map</MAP>\n<ACT>right</ACT>",
                target="<POS>(0, 1)</POS>",
            ),
            predicted=b"<POS>(0, 1)</POS>",
        )
        incorrect = GreedyPrediction(
            example=PerStepExample(
                input="<MAP>wrong-map</MAP>\n<ACT>left</ACT>",
                target="<POS>(2, 0)</POS>",
            ),
            predicted=b"<POS>(2, 1)</POS>",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "errors.txt"
            errors = write_greedy_error_log(path, [correct, incorrect])
            report = path.read_text(encoding="utf-8")

        self.assertEqual(errors, 1)
        self.assertIn("# greedy POS errors: 1 / 2 examples", report)
        self.assertIn("<MAP>wrong-map</MAP>\n<ACT>left</ACT>", report)
        self.assertIn("expected:  <POS>(2, 0)</POS>", report)
        self.assertIn("predicted: <POS>(2, 1)</POS>", report)
        self.assertNotIn("correct-map", report)


if __name__ == "__main__":
    unittest.main()
