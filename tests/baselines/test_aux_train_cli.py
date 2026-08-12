from __future__ import annotations

import ast
from pathlib import Path
import unittest


TRAIN_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "agimaze_predict"
    / "baselines"
    / "aux_transformer"
    / "train.py"
)


class AuxTrainCliDefinitionTest(unittest.TestCase):
    def test_supports_optional_config_and_dataset_shards(self) -> None:
        """Inspect the parser definition without importing optional PyTorch."""

        tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
        build_parser = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_parser"
        )
        option_calls = [
            node
            for node in ast.walk(build_parser)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ]
        options = {
            value.value
            for call in option_calls
            for value in call.args
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        }

        self.assertIn("--config", options)
        self.assertIn("--train-dataset", options)
        self.assertIn("--validation-dataset", options)
        self.assertIn("--test-dataset", options)
        self.assertIn("--no-overwrite", options)
        self.assertNotIn("--state-tokens", options)
        self.assertIn("--aux-latents-per-token", options)
        self.assertIn("--aux-gate-mode", options)
        self.assertIn("--aux-scale", options)
        self.assertIn("--aux-gate-init", options)
        self.assertNotIn("--dataset", options)
        self.assertNotIn("--validation-fraction", options)


if __name__ == "__main__":
    unittest.main()
