from __future__ import annotations

import unittest

from agimaze_predict.baselines.byte_transformer.tokenizer import IGNORE_INDEX
from agimaze_predict.baselines.byte_transformer.txt_tokenizer import (
    collate_txt_rollouts,
    serialize_txt_rollout,
)
from agimaze_predict.data.txt_trace import TxtRolloutExample


class TxtTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example = TxtRolloutExample(
            segments=(
                ("<MAP>m</MAP>", False),
                ("<ACT>right</ACT>", False),
                ("<TXT>first</TXT>", True),
                ("<ACT>down</ACT>", False),
                ("<TXT>second</TXT>", True),
            )
        )

    def test_serialization_marks_only_complete_txt_blocks(self) -> None:
        serialized = serialize_txt_rollout(self.example)
        text = bytes(serialized.token_ids).decode("utf-8")
        expected = set()
        cursor = 0
        for segment, is_target in self.example.segments:
            if cursor:
                cursor += 1  # newline
            if is_target:
                expected.update(range(cursor, cursor + len(segment.encode("utf-8"))))
            cursor += len(segment.encode("utf-8"))
        self.assertEqual(text, self.example.text)
        self.assertEqual(serialized.target_byte_indices, expected)

    def test_causal_shift_supervises_every_txt_byte_but_no_act_bytes(self) -> None:
        serialized = serialize_txt_rollout(self.example)
        batch = collate_txt_rollouts([self.example], context_length=512)
        labels = batch["labels"][0]
        active_targets = {index + 1 for index, label in enumerate(labels) if label != IGNORE_INDEX}
        self.assertEqual(active_targets, serialized.target_byte_indices)

        text = bytes(serialized.token_ids).decode("utf-8")
        first_txt = text.index("<TXT>first</TXT>")
        second_txt = text.index("<TXT>second</TXT>")
        self.assertEqual(labels[first_txt - 1], ord("<"))
        self.assertEqual(labels[second_txt - 1], ord("<"))
        for index in range(text.index("<ACT>right</ACT>"), text.index("<ACT>right</ACT>") + len("<ACT>right</ACT>")):
            self.assertEqual(labels[index - 1] if index else IGNORE_INDEX, IGNORE_INDEX)

    def test_context_overflow_explains_remedy(self) -> None:
        with self.assertRaisesRegex(ValueError, "increase context_length or decrease depth"):
            collate_txt_rollouts([self.example], context_length=2)


if __name__ == "__main__":
    unittest.main()
