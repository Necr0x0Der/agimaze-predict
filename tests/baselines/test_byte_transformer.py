from __future__ import annotations

import importlib.util
import math
import unittest

from agimaze_predict.baselines.byte_transformer.tokenizer import (
    VOCAB_SIZE_WITH_STATE,
    collate_byte_examples,
)
from agimaze_predict.data.per_step import PerStepExample

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from agimaze_predict.baselines.byte_transformer.model import (
        ByteTransformer,
        ByteTransformerConfig,
        target_cross_entropy,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "optional dependency 'torch' is not installed")
class ByteTransformerTest(unittest.TestCase):
    def test_forward_shape_and_target_only_loss(self) -> None:
        examples = [
            PerStepExample(
                input="<MAP>\n| S |\n</MAP>\n<ACT>right</ACT>",
                target="<POS>(0, 0)</POS>",
            ),
            PerStepExample(
                input="<MAP>x</MAP>\n<ACT>up</ACT>",
                target="<POS>(0, 0)</POS>",
            ),
        ]
        batch = collate_byte_examples(examples, context_length=128)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        labels = torch.tensor(batch["labels"], dtype=torch.long)
        model = ByteTransformer(
            ByteTransformerConfig(context_length=128, d_model=32, n_heads=4, n_layers=2)
        )

        logits = model(input_ids)
        loss = target_cross_entropy(logits, labels)
        loss.backward()

        self.assertEqual(logits.shape, (2, input_ids.shape[1], 257))
        self.assertTrue(math.isfinite(float(loss.item())))
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all().item())

    def test_parallel_aux_stream_forward_loss_and_diagnostics(self) -> None:
        example = PerStepExample(
            input="<MAP>x</MAP>\n<ACT>right</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_byte_examples([example], context_length=128)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        labels = torch.tensor(batch["labels"], dtype=torch.long)
        aux_lengths = torch.tensor(batch["aux_lengths"], dtype=torch.long)
        model = ByteTransformer(
            ByteTransformerConfig(
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
        config = ByteTransformerConfig(
            context_length=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            aux_latents_per_token=2,
        )
        model = ByteTransformer(config)
        aux_lengths = torch.tensor([3, 5], dtype=torch.long)
        aux_to_main, main_to_aux, auxiliary_length = model._auxiliary_masks(
            sequence_length=8, aux_lengths=aux_lengths
        )

        self.assertEqual(auxiliary_length, 10)
        # Example zero has only three source positions: all auxiliary rows
        # clamp at its final permitted source byte, never target index 3+.
        self.assertTrue((aux_to_main[0] <= 2).all().item())
        self.assertTrue((aux_to_main[1] <= 4).all().item())
        # Main position i reads at most latent slots assigned to source i.
        self.assertEqual(main_to_aux[1, 0].item(), 1)
        self.assertEqual(main_to_aux[1, 4].item(), 9)

    def test_off_gate_matches_explicit_aux_ablation(self) -> None:
        example = PerStepExample(
            input="<MAP>x</MAP>\n<ACT>right</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_byte_examples([example], context_length=128)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        aux_lengths = torch.tensor(batch["aux_lengths"], dtype=torch.long)
        model = ByteTransformer(
            ByteTransformerConfig(
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

    def test_state_slot_model_forward_and_loss(self) -> None:
        example = PerStepExample(
            input="<MAP>x</MAP>\n<ACT>right</ACT>\n<ACT>up</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        batch = collate_byte_examples([example], context_length=128, state_tokens=2)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        labels = torch.tensor(batch["labels"], dtype=torch.long)
        model = ByteTransformer(
            ByteTransformerConfig(
                context_length=128,
                d_model=32,
                n_heads=4,
                n_layers=2,
                state_tokens=2,
            )
        )

        logits = model(input_ids)
        loss = target_cross_entropy(logits, labels)
        loss.backward()

        self.assertEqual(model.config.vocab_size, VOCAB_SIZE_WITH_STATE)
        # The latent-slot ID is input-only, never a class competing with bytes
        # in the output softmax.
        self.assertEqual(logits.shape[-1], VOCAB_SIZE_WITH_STATE - 1)
        self.assertTrue(math.isfinite(float(loss.item())))


if __name__ == "__main__":
    unittest.main()
