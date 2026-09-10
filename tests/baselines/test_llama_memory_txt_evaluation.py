from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    torch = None


@unittest.skipIf(torch is None, "requires the optional torch dependency")
class LlamaMemoryTxtEvaluationTest(unittest.TestCase):
    def test_accuracy_aligns_targets_after_per_event_memory_tokens(self) -> None:
        from agimaze_predict.baselines.llama_memory.train_txt import _txt_accuracy_counts

        memory_tokens = 3
        # Original labels contain two source/target pairs padded to common width.
        # Their target spans start after prompts of 4 and 5 tokens respectively.
        batch = {
            "labels": torch.tensor([
                [-100, -100, -100, -100, 7, 8, -100, -100],
                [-100, -100, -100, -100, -100, 9, 10, 11],
            ]),
            "prompt_lengths": torch.tensor([4, 5]),
            "event_mask": torch.tensor([[1, 0], [1, 1]]),
        }
        # Memory insertion expands those prefixes to 7 and 11 tokens. Make the
        # logits predict the original targets at their expanded causal positions.
        logits = torch.zeros((2, 14, 16))
        logits[0, 6, 7] = 1
        logits[0, 7, 8] = 1
        logits[1, 10, 9] = 1
        logits[1, 11, 10] = 1
        logits[1, 12, 11] = 1

        correct, active = _txt_accuracy_counts(logits, batch, memory_tokens=memory_tokens)

        self.assertEqual(active, 5)
        self.assertEqual(correct, 5)


if __name__ == "__main__":
    unittest.main()
