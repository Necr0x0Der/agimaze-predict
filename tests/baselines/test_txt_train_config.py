from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agimaze_predict.baselines.byte_transformer.txt_config import load_txt_training_config


class TxtTrainConfigTest(unittest.TestCase):
    def write_config(self, root: Path, content: str) -> Path:
        path = root / "experiments" / "txt.toml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_map_context_depth_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_config(
                Path(temp_dir),
                """
[data]
train_files = ["data/train.jsonl"]
validation_files = ["data/valid.jsonl"]
initial_context = "MAP"
depth = 4

[run]
output = "runs/model.pt"
""",
            )
            values = load_txt_training_config(path)
        self.assertEqual(values["train_datasets"], [path.parent / "data/train.jsonl"])
        self.assertEqual(values["validation_datasets"], [path.parent / "data/valid.jsonl"])
        self.assertEqual(values["output"], path.parent / "runs/model.pt")
        self.assertEqual(values["initial_context"], "MAP")
        self.assertEqual(values["depth"], 4)

    def test_rejects_invalid_initial_context_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            initial = self.write_config(root, "[data]\ninitial_context = \"map\"\n")
            with self.assertRaisesRegex(ValueError, "exactly 'MAP' or 'START'"):
                load_txt_training_config(initial)
            depth = self.write_config(root, "[data]\ndepth = 0\n")
            with self.assertRaisesRegex(ValueError, "depth must be a positive integer"):
                load_txt_training_config(depth)


if __name__ == "__main__":
    unittest.main()
