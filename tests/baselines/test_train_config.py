from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agimaze_predict.baselines.byte_transformer.config import (
    load_training_config,
    resolve_training_arguments,
)


class TrainConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        # Importing the trainer needs optional PyTorch, so these tests use a
        # minimal parser with the same suppressed-default behavior.
        import argparse

        self.parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        self.parser.add_argument("--config", type=Path)
        self.parser.add_argument("--train-dataset", dest="train_datasets", type=Path, action="append")
        self.parser.add_argument("--validation-dataset", dest="validation_datasets", type=Path, action="append")
        self.parser.add_argument("--output", type=Path)
        self.parser.add_argument("--epochs", type=int)
        self.parser.add_argument("--d-model", type=int)
        self.parser.add_argument("--state-tokens", type=int)
        overwrite_group = self.parser.add_mutually_exclusive_group()
        overwrite_group.add_argument("--overwrite", action="store_true")
        overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false")

    def write_config(self, root: Path, content: str) -> Path:
        config = root / "experiments" / "training.toml"
        config.parent.mkdir()
        config.write_text(content, encoding="utf-8")
        return config

    def test_loads_shards_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.write_config(
                root,
                """
[data]
train_files = ["data/train-a.jsonl", "data/train-b.jsonl"]
test_files = ["data/test.jsonl"]

[model]
d_model = 96
n_heads = 3
state_tokens = 4

[training]
epochs = 12

[run]
output = "runs/model.pt"
""",
            )

            values = load_training_config(config)

            base = config.parent
            self.assertEqual(values["train_datasets"], [base / "data/train-a.jsonl", base / "data/train-b.jsonl"])
            self.assertEqual(values["validation_datasets"], [base / "data/test.jsonl"])
            self.assertEqual(values["output"], base / "runs/model.pt")
            self.assertEqual(values["d_model"], 96)
            self.assertEqual(values["n_heads"], 3)
            self.assertEqual(values["state_tokens"], 4)
            self.assertEqual(values["epochs"], 12)

    def test_explicit_cli_options_override_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.write_config(
                root,
                """
[data]
train_datasets = ["data/config-train.jsonl"]
validation_datasets = ["data/config-validation.jsonl"]

[training]
epochs = 12

[model]
d_model = 96
state_tokens = 4

[run]
output = "runs/config.pt"
overwrite = true
""",
            )

            args = resolve_training_arguments(
                self.parser,
                [
                    "--config",
                    str(config),
                    "--train-dataset",
                    "cli-train.jsonl",
                    "--validation-dataset",
                    "cli-validation.jsonl",
                    "--output",
                    "cli.pt",
                    "--epochs",
                    "3",
                    "--d-model",
                    "64",
                    "--state-tokens",
                    "1",
                    "--no-overwrite",
                ],
            )

            self.assertEqual(args.train_datasets, [Path("cli-train.jsonl")])
            self.assertEqual(args.validation_datasets, [Path("cli-validation.jsonl")])
            self.assertEqual(args.output, Path("cli.pt"))
            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.d_model, 64)
            self.assertEqual(args.state_tokens, 1)
            self.assertFalse(args.overwrite)

    def test_rejects_negative_or_boolean_state_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            negative = self.write_config(Path(temp_dir), "[model]\nstate_tokens = -1\n")
            with self.assertRaisesRegex(ValueError, "state_tokens must be a non-negative integer"):
                load_training_config(negative)

        with tempfile.TemporaryDirectory() as temp_dir:
            boolean = self.write_config(Path(temp_dir), "[model]\nstate_tokens = true\n")
            with self.assertRaisesRegex(ValueError, "state_tokens must be a non-negative integer"):
                load_training_config(boolean)

    def test_rejects_unknown_configuration_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir), "[model]\nunknown = 1\n")
            with self.assertRaisesRegex(ValueError, "unsupported keys: unknown"):
                load_training_config(config)

    def test_rejects_ambiguous_validation_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                """
[data]
validation_datasets = ["validation.jsonl"]
test_files = ["test.jsonl"]
""",
            )
            with self.assertRaisesRegex(ValueError, r"cannot set both validation_\* and test_\* dataset lists"):
                load_training_config(config)


if __name__ == "__main__":
    unittest.main()
