from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from agimaze_predict.baselines.visual_transformer.txt_config import load_visual_txt_training_config
from agimaze_predict.baselines.visual_transformer.txt_tokenizer import (
    collate_visual_txt_rollouts,
    visual_txt_rollouts_from_trace,
)
from agimaze_predict.data.txt_trace import TxtTrace, TxtTransition

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from agimaze_predict.baselines.visual_transformer.model import (
        VisualTransformer,
        VisualTransformerConfig,
        target_cross_entropy,
    )


def trace() -> TxtTrace:
    return TxtTrace(
        map_text="ab\ncd",
        start_text="at (0, 0)",
        transitions=(TxtTransition("right", "at (0, 1)"), TxtTransition("down", "at (1, 1)")),
    )


class VisualTxtTokenizerTest(unittest.TestCase):
    def test_each_target_sees_its_action_and_ground_truth_prior_txt(self) -> None:
        rollouts = visual_txt_rollouts_from_trace(trace(), initial_context="MAP", depth=2).rollouts
        self.assertEqual(rollouts[0].source, b"<ACT>right</ACT>")
        self.assertEqual(
            rollouts[1].source,
            b"<ACT>right</ACT>\n<TXT>at (0, 1)</TXT>\n<ACT>down</ACT>",
        )
        self.assertEqual(rollouts[0].target, b"<TXT>at (0, 1)</TXT>")

        batch = collate_visual_txt_rollouts(rollouts, context_length=128, canvas_height=3, canvas_width=4)
        first_target_start = len(rollouts[0].source) - 1
        self.assertEqual(batch["labels"][0][first_target_start], ord("<"))
        self.assertEqual(batch["event_positions"][0], [len(b"<ACT>right</ACT>") - 1, -1])
        self.assertEqual(
            batch["event_positions"][1],
            [len(b"<ACT>right</ACT>") - 1, len(rollouts[1].source) - 1],
        )
        self.assertEqual(batch["target_input_ids"][0][0], ord(">"))
        self.assertEqual(batch["target_labels"][0][0], ord("<"))

    def test_start_context_is_textual_but_map_stays_visual(self) -> None:
        rollout = visual_txt_rollouts_from_trace(trace(), initial_context="START", depth=1).rollouts[0]
        self.assertEqual(rollout.map_rows, ("ab", "cd"))
        self.assertEqual(rollout.source, b"<START>at (0, 0)</START>\n<ACT>right</ACT>")


class VisualTxtConfigTest(unittest.TestCase):
    def test_loads_txt_settings_and_uses_visual_gate_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment.toml"
            path.write_text(
                """[data]
train_files = ["train.jsonl"]
validation_files = ["valid.jsonl"]
initial_context = "MAP"
depth = 2

[run]
output = "run.pt"
""",
                encoding="utf-8",
            )
            values = load_visual_txt_training_config(path)
        self.assertEqual(values["train_datasets"], [path.parent / "train.jsonl"])
        self.assertEqual(values["initial_context"], "MAP")
        self.assertEqual(values["depth"], 2)


@unittest.skipUnless(TORCH_AVAILABLE, "optional dependency 'torch' is not installed")
class VisualTxtModelTest(unittest.TestCase):
    def test_full_text_and_visual_only_txt_batches_backpropagate(self) -> None:
        rollout = visual_txt_rollouts_from_trace(trace(), initial_context="MAP", depth=1).rollouts[0]
        batch = collate_visual_txt_rollouts([rollout], context_length=128, canvas_height=3, canvas_width=4)
        kwargs = {
            key: torch.tensor(batch[key])
            for key in ("visual_maps", "event_positions", "event_counts")
        }
        common = dict(
            context_length=128, d_model=32, visual_d_model=32, n_heads=4, n_layers=1,
            visual_spatial_layers=1, visual_temporal_layers=1, canvas_height=3, canvas_width=4,
            visual_gate_init=0.99,
        )
        for mode, label_key in (("full_text", "labels"), ("visual_only", "target_labels")):
            with self.subTest(mode=mode):
                model = VisualTransformer(VisualTransformerConfig(**common, pos_readout=mode))
                extra = {} if mode == "full_text" else {"target_input_ids": torch.tensor(batch["target_input_ids"])}
                logits = model(torch.tensor(batch["input_ids"]), **kwargs, **extra)
                labels = torch.tensor(batch[label_key])
                loss = target_cross_entropy(logits, labels)
                loss.backward()
                self.assertEqual(logits.shape[:2], labels.shape)
                self.assertTrue(torch.isfinite(loss).item())
                self.assertIsNotNone(model.visual_memory.text_to_visual.weight.grad)


if __name__ == "__main__":
    unittest.main()
