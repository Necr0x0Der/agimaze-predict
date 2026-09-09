from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from agimaze_predict.baselines.llama_memory.config import resolve_training_arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-dataset", dest="train_datasets", type=Path, action="append")
    parser.add_argument("--validation-dataset", dest="validation_datasets", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    return parser


class LlamaMemoryConfigTest(unittest.TestCase):
    def test_config_resolves_output_relative_to_toml(self) -> None:
        with self.subTest("temporary config"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "experiments" / "llama-memory" / "seq" / "run.toml"
                config.parent.mkdir(parents=True)
                train = root / "train.jsonl"
                valid = root / "valid.jsonl"
                record = '{"input":"<MAP>x</MAP>\\n<ACT>up</ACT>","target":"<POS>(0, 0)</POS>"}\n'
                train.write_text(record)
                valid.write_text(record)
                config.write_text(
                    "[data]\n"
                    'train_files = ["../../../train.jsonl"]\n'
                    'validation_files = ["../../../valid.jsonl"]\n'
                    "[run]\n"
                    'output = "../../../runs/model.pt"\n'
                )

                args = resolve_training_arguments(_parser(), ["--config", str(config)])

                self.assertEqual(args.output, (root / "runs" / "model.pt").resolve())

    def test_explicit_output_retains_shell_relative_semantics(self) -> None:
        with self.subTest("CLI output"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "run.toml"
                train = root / "train.jsonl"
                valid = root / "valid.jsonl"
                record = '{"input":"<MAP>x</MAP>\\n<ACT>up</ACT>","target":"<POS>(0, 0)</POS>"}\n'
                train.write_text(record)
                valid.write_text(record)
                config.write_text(
                    "[data]\n"
                    'train_files = ["train.jsonl"]\n'
                    'validation_files = ["valid.jsonl"]\n'
                    "[run]\n"
                    'output = "from-config.pt"\n'
                )

                args = resolve_training_arguments(
                    _parser(), ["--config", str(config), "--output", "from-cli.pt"]
                )

                self.assertEqual(args.output, Path("from-cli.pt"))


if __name__ == "__main__":
    unittest.main()
