from __future__ import annotations

import importlib.util
import unittest

from agimaze_predict.baselines.visual_transformer.tokenizer import collate_visual_examples, serialize_visual_example
from agimaze_predict.data.per_step import PerStepExample

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from agimaze_predict.baselines.visual_transformer.model import VisualTransformer, VisualTransformerConfig, target_cross_entropy


class VisualTokenizerTest(unittest.TestCase):
    def test_removes_map_from_text_and_uses_pos_as_query(self) -> None:
        example = PerStepExample(input="<MAP>ab\ncd</MAP>\n<ACT>left</ACT>", target="<POS>(0, 1)</POS>")
        item = serialize_visual_example(example)
        self.assertEqual(item.map_rows, ("ab", "cd"))
        self.assertEqual(bytes(item.token_ids[: item.target_start]), b"<ACT>left</ACT>\n<POS>")
        self.assertEqual(item.target_suffix, b"(0, 1)</POS>")
        batch = collate_visual_examples([example], context_length=64, canvas_height=3, canvas_width=4)
        self.assertEqual(batch["visual_maps"][0][0][:2], [ord("a"), ord("b")])
        self.assertEqual(batch["event_positions"][0], [len(b"<ACT>left</ACT>") - 1])
        self.assertEqual(batch["labels"][0][item.target_start - 1], ord("("))


@unittest.skipUnless(TORCH_AVAILABLE, "optional dependency 'torch' is not installed")
class VisualTransformerTest(unittest.TestCase):
    def test_full_text_forward_is_finite_and_visual_path_gets_gradients(self) -> None:
        example = PerStepExample(input="<MAP>ab\ncd</MAP>\n<ACT>left</ACT>", target="<POS>(0, 1)</POS>")
        batch = collate_visual_examples([example], context_length=64, canvas_height=3, canvas_width=4)
        model = VisualTransformer(VisualTransformerConfig(context_length=64, d_model=32, visual_d_model=32, n_heads=4, n_layers=2, visual_spatial_layers=1, visual_temporal_layers=1, canvas_height=3, canvas_width=4))
        logits = model(torch.tensor(batch["input_ids"]), visual_maps=torch.tensor(batch["visual_maps"]), event_positions=torch.tensor(batch["event_positions"]), event_counts=torch.tensor(batch["event_counts"]))
        labels = torch.tensor(batch["labels"])
        loss = target_cross_entropy(logits, labels)
        loss.backward()
        self.assertEqual(logits.shape[:2], labels.shape)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.visual_memory.character_embedding.weight.grad)
        self.assertIsNotNone(model.visual_memory.text_to_visual.weight.grad)

    def test_visual_only_target_decoder_has_no_action_byte_input(self) -> None:
        example = PerStepExample(input="<MAP>ab\ncd</MAP>\n<ACT>left</ACT>", target="<POS>(0, 1)</POS>")
        batch = collate_visual_examples([example], context_length=64, canvas_height=3, canvas_width=4)
        model = VisualTransformer(VisualTransformerConfig(context_length=64, d_model=32, visual_d_model=32, n_heads=4, n_layers=1, visual_spatial_layers=1, visual_temporal_layers=1, canvas_height=3, canvas_width=4, pos_readout="visual_only"))
        logits = model(torch.tensor(batch["input_ids"]), visual_maps=torch.tensor(batch["visual_maps"]), event_positions=torch.tensor(batch["event_positions"]), event_counts=torch.tensor(batch["event_counts"]), target_input_ids=torch.tensor(batch["target_input_ids"]))
        labels = torch.tensor(batch["target_labels"])
        target_cross_entropy(logits, labels).backward()
        self.assertEqual(logits.shape[:2], labels.shape)
        self.assertIsNotNone(model.target_blocks[0].cross_attention.query.weight.grad)


if __name__ == "__main__":
    unittest.main()
