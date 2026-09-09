from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from agimaze_predict.baselines.llama_memory.txt_config import resolve_txt_training_arguments


class LlamaMemoryTxtConfigTest(unittest.TestCase):
    def test_config_paths_resolve_relative_to_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiments" / "txt.toml"
            config.parent.mkdir()
            config.write_text(
                """[data]
train_files = ["../data/train.jsonl"]
validation_files = ["../data/valid.jsonl"]
initial_context = "START"
depth = 2

[run]
output = "../runs/model.pt"
""",
                encoding="utf-8",
            )
            parser = __import__("argparse").ArgumentParser(argument_default=__import__("argparse").SUPPRESS)
            parser.add_argument("--config", type=Path)
            arguments = resolve_txt_training_arguments(parser, ["--config", str(config)])

            self.assertEqual(arguments.train_datasets, [(root / "data" / "train.jsonl").resolve()])
            self.assertEqual(arguments.validation_datasets, [(root / "data" / "valid.jsonl").resolve()])
            self.assertEqual(arguments.output, (root / "runs" / "model.pt").resolve())
            self.assertEqual(arguments.depth, 2)


if __name__ == "__main__":
    unittest.main()
