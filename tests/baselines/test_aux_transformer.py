from __future__ import annotations

import importlib.util
import math
import unittest

from agimaze_predict.baselines.aux_transformer.tokenizer import (
    collate_aux_examples,
    collate_aux_target_examples,
)
from agimaze_predict.data.per_step import PerStepExample

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from agimaze_predict.baselines.aux_transformer.model import (
        AuxTransformer,
        AuxTransformerConfig,
        target_cross_entropy,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "optional dependency 'torch' is not installed")
class AuxTransformerTest(unittest.TestCase):
    def test_parallel_aux_stream_forward_loss_and_diagnostics(self) -> None:
        example = PerStepExample(
            input="<MAP>x</MAP>\n<ACT>right</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_aux_examples([example], context_length=128)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        labels = torch.tensor(batch["labels"], dtype=torch.long)
        aux_lengths = torch.tensor(batch["aux_lengths"], dtype=torch.long)
        model = AuxTransformer(
            AuxTransformerConfig(
                context_length=128,
                d_model=32,
                n_heads=4,
                n_layers=2,
                aux_latents_per_token=2,
                aux_gate_mode="learned",
                aux_gate_init=0.05,
            )
        )

        logits, diagnostics = model(
            input_ids, aux_lengths=aux_lengths, return_aux_diagnostics=True
        )
        loss = target_cross_entropy(logits, labels)
        loss.backward()

        self.assertEqual(logits.shape, (1, input_ids.shape[1], 257))
        self.assertAlmostEqual(diagnostics["gate/layer_0"], 0.05, places=5)
        self.assertIn("cross_residual_ratio/layer_1", diagnostics)
        self.assertIsNotNone(model.aux_blocks[0].cross_attention.query.weight.grad)
        self.assertTrue(torch.isfinite(model.aux_blocks[0].cross_attention.query.weight.grad).all().item())

    def test_auxiliary_mask_stops_before_target_and_main_cannot_read_future_latents(self) -> None:
        config = AuxTransformerConfig(
            context_length=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            aux_latents_per_token=2,
        )
        model = AuxTransformer(config)
        aux_lengths = torch.tensor([3, 5], dtype=torch.long)
        aux_to_main, main_to_aux, auxiliary_length = model._auxiliary_masks(
            sequence_length=8, aux_lengths=aux_lengths
        )

        self.assertEqual(auxiliary_length, 10)
        self.assertTrue((aux_to_main[0] <= 2).all().item())
        self.assertTrue((aux_to_main[1] <= 4).all().item())
        self.assertEqual(main_to_aux[1, 0].item(), 1)
        self.assertEqual(main_to_aux[1, 4].item(), 9)

    def test_off_gate_matches_explicit_aux_ablation(self) -> None:
        example = PerStepExample(
            input="<MAP>x</MAP>\n<ACT>right</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_aux_examples([example], context_length=128)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        aux_lengths = torch.tensor(batch["aux_lengths"], dtype=torch.long)
        model = AuxTransformer(
            AuxTransformerConfig(
                context_length=128,
                d_model=32,
                n_heads=4,
                n_layers=2,
                aux_latents_per_token=1,
                aux_gate_mode="off",
            )
        )
        model.eval()
        logits = model(input_ids, aux_lengths=aux_lengths)
        ablated_logits = model(input_ids, aux_lengths=aux_lengths, disable_aux=True)
        self.assertTrue(torch.equal(logits, ablated_logits))

    def test_target_aligned_decoder_is_teacher_forced_and_reaches_aux(self) -> None:
        example = PerStepExample(
            input="<MAP>xyz</MAP>\n<ACT>right</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_aux_target_examples([example], context_length=128)
        target_input_ids = torch.tensor(batch["aux_target_input_ids"], dtype=torch.long)
        target_labels = torch.tensor(batch["aux_target_labels"], dtype=torch.long)
        self.assertEqual(target_input_ids.shape, target_labels.shape)
        self.assertEqual(target_input_ids[0, 0].item(), ord("\n"))
        self.assertEqual(target_labels[0, 0].item(), ord("<"))

        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        aux_lengths = torch.tensor(batch["aux_lengths"], dtype=torch.long)
        model = AuxTransformer(
            AuxTransformerConfig(
                context_length=128,
                d_model=32,
                n_heads=4,
                n_layers=2,
                aux_latents_per_token=2,
                aux_target_weight=0.1,
            )
        )
        main_logits, auxiliary_states = model(
            input_ids, aux_lengths=aux_lengths, return_auxiliary_states=True
        )
        main_loss = target_cross_entropy(main_logits, torch.tensor(batch["labels"]))
        aux_target_logits = model.auxiliary_target_logits(
            target_input_ids, auxiliary_states=auxiliary_states, aux_lengths=aux_lengths
        )
        aux_target_loss = target_cross_entropy(aux_target_logits, target_labels)
        (main_loss + 0.1 * aux_target_loss).backward()
        self.assertEqual(aux_target_logits.shape[:2], target_labels.shape)
        self.assertIsNotNone(model.aux_target_output.weight.grad)
        self.assertIsNotNone(model.aux_blocks[0].attention.qkv.weight.grad)



if __name__ == "__main__":
    unittest.main()
