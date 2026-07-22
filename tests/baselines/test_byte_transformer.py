from __future__ import annotations

import importlib.util
import math
import unittest

from agimaze_predict.baselines.byte_transformer.tokenizer import collate_byte_examples
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


if __name__ == "__main__":
    unittest.main()
