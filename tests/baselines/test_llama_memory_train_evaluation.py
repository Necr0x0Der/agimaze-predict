from __future__ import annotations

import ast
from pathlib import Path
import unittest


TRAIN_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "agimaze_predict"
    / "baselines"
    / "llama_memory"
    / "train.py"
)
EVALUATE_PATH = TRAIN_PATH.with_name("evaluate.py")


class LlamaMemoryTrainingEvaluationDefinitionTest(unittest.TestCase):
    def test_train_uses_validation_datasets_and_reports_greedy_exactness(self) -> None:
        source = TRAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("from .evaluate import evaluate_examples", source)
        self.assertIn("_load_examples(args.validation_datasets)", source)
        self.assertIn("val_target_token_nll", source)
        self.assertIn("val_greedy_exact_target_accuracy", source)
        self.assertIn('"metrics": validation', source)
        self.assertIn('"format": "agimaze_predict.llama_memory.v1"', source)
        self.assertIn("output.parent.mkdir(parents=True, exist_ok=True)", source)
        self.assertNotIn("output.mkdir(parents=True, exist_ok=True)", source)
        ast.parse(EVALUATE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
