from __future__ import annotations

import importlib.util
import math
import unittest

from agimaze_predict.baselines.aux_transformer.tokenizer import (
    collate_aux_examples,
    collate_aux_denoising_examples,
    expand_denoise_labels_for_latents,
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

    def test_masked_denoising_is_source_only_and_reaches_aux(self) -> None:
        example = PerStepExample(
            input="<MAP>xyz</MAP>\n<ACT>right</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_aux_denoising_examples(
            [example],
            context_length=128,
            mask_rate=0.2,
            mask_span_length=3,
            generator=torch.Generator().manual_seed(7),
        )
        source_labels = batch["denoise_source_labels"][0]
        aux_length = batch["aux_lengths"][0]
        self.assertEqual(len(source_labels), aux_length)
        self.assertGreater(sum(label != -100 for label in source_labels), 0)
        self.assertEqual(batch["denoise_input_ids"][0][aux_length:], batch["input_ids"][0][aux_length:])

        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        denoise_input_ids = torch.tensor(batch["denoise_input_ids"], dtype=torch.long)
        aux_lengths = torch.tensor(batch["aux_lengths"], dtype=torch.long)
        denoise_labels = torch.tensor(
            expand_denoise_labels_for_latents(source_labels=[source_labels], aux_latents_per_token=2),
            dtype=torch.long,
        )
        model = AuxTransformer(
            AuxTransformerConfig(
                context_length=128,
                d_model=32,
                n_heads=4,
                n_layers=2,
                aux_latents_per_token=2,
                aux_denoise_weight=0.1,
                aux_mask_rate=0.2,
                aux_mask_span_length=3,
            )
        )
        # The main path stays clean; the auxiliary reconstruction path is separate.
        main_loss = target_cross_entropy(model(input_ids, aux_lengths=aux_lengths), torch.tensor(batch["labels"]))
        denoise_logits = model.auxiliary_denoising_logits(denoise_input_ids, aux_lengths=aux_lengths)
        denoise_loss = target_cross_entropy(denoise_logits, denoise_labels)
        (main_loss + 0.1 * denoise_loss).backward()
        self.assertEqual(denoise_logits.shape[:2], denoise_labels.shape)
        self.assertIsNotNone(model.aux_denoise_head.weight.grad)
        self.assertIsNotNone(model.aux_blocks[0].attention.qkv.weight.grad)



if __name__ == "__main__":
    unittest.main()
