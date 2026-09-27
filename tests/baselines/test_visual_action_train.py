"""Checkpoint selection and dual logging for visual action training."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from agimaze_predict.baselines.visual_transformer.config import DEFAULT_TRAINING_ARGUMENTS
from agimaze_predict.baselines.visual_transformer.train import make_logger, train


class VisualActionTrainTest(unittest.TestCase):
    def test_visual_only_combined_loss_saves_checkpoint(self) -> None:
        from agimaze_predict.baselines.visual_transformer.model import VisualTransformer, VisualTransformerConfig

        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_per_step.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "visual_only.pt"
            args = Namespace(**(DEFAULT_TRAINING_ARGUMENTS | {
                "train_datasets": [fixture], "validation_datasets": [fixture],
                "output": output, "predict_actions": True, "pos_readout": "visual_only",
                "epochs": 1, "evaluate_every": 1, "batch_size": 2,
                "context_length": 64, "d_model": 16, "visual_d_model": 16,
                "n_layers": 1, "n_heads": 4, "visual_spatial_layers": 1,
                "visual_temporal_layers": 1, "canvas_height": 3, "canvas_width": 9,
            }))
            result = train(args, make_logger(output))
            self.assertEqual(result["epoch"], 1)
            self.assertEqual(result["metrics"]["total_bytes"],
                             result["metrics"]["act_bytes"] + result["metrics"]["pos_bytes"])
            self.assertGreater(result["metrics"]["act_bytes"], 0)
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            model = VisualTransformer(VisualTransformerConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])

    def test_saves_only_improved_validation_and_logs_each_epoch(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_per_step.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "best.pt"
            args = Namespace(**(DEFAULT_TRAINING_ARGUMENTS | {
                "train_datasets": [fixture], "validation_datasets": [fixture],
                "output": output, "predict_actions": True,
                "epochs": 3, "evaluate_every": 1, "batch_size": 2,
                "context_length": 64, "d_model": 16, "visual_d_model": 16,
                "n_layers": 1, "n_heads": 4, "visual_spatial_layers": 1,
                "visual_temporal_layers": 1, "canvas_height": 3, "canvas_width": 9,
            }))
            logger = make_logger(output)
            losses = [5.0, 4.0, 6.0]
            def metrics(*_args, **_kwargs):
                score = losses.pop(0)
                return {"total_byte_nll": score, "act_byte_nll": score,
                        "pos_byte_nll": score, "total_bytes": 2,
                        "act_bytes": 1, "pos_bytes": 1}

            original_save = torch.save
            saved_epochs = []
            def save(checkpoint, path):
                saved_epochs.append(checkpoint["epoch"])
                original_save(checkpoint, path)

            with patch("agimaze_predict.baselines.visual_transformer.train.evaluate_action_losses", side_effect=metrics), \
                 patch("agimaze_predict.baselines.visual_transformer.train.torch.save", side_effect=save):
                result = train(args, logger)

            self.assertEqual(saved_epochs, [1, 2])
            self.assertEqual(result["epoch"], 2)
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["metrics"]["total_byte_nll"], 4.0)
            text = logger.path.read_text(encoding="utf-8")
            self.assertIn("val_act_byte_nll=4.000000", text)
            self.assertIn("Best validation loss: 4.000000; epoch: 2", text)
            self.assertEqual(text.count("Checkpoint saved (epoch="), 2)
            with self.assertRaises(FileExistsError):
                train(args, make_logger(output))


if __name__ == "__main__":
    unittest.main()
