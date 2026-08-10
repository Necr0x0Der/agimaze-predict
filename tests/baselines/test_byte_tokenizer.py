from __future__ import annotations

import unittest

from agimaze_predict.baselines.byte_transformer.tokenizer import (
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    STATE_TOKEN_ID,
    collate_byte_examples,
    serialize_example,
)
from agimaze_predict.data.per_step import PerStepExample


EXAMPLE = PerStepExample(
    input="<MAP>\n| S |\n</MAP>\n<ACT>right</ACT>",
    target="<POS>(0, 1)</POS>",
)


class ByteTokenizerTest(unittest.TestCase):
    def test_serialization_is_literal_utf8_input_newline_target(self) -> None:
        serialized = serialize_example(EXAMPLE)
        prefix = (EXAMPLE.input + "\n").encode("utf-8")
        self.assertEqual(serialized.target_start, len(prefix))
        self.assertEqual(bytes(serialized.token_ids), prefix + EXAMPLE.target.encode("utf-8"))

    def test_zero_state_tokens_preserves_literal_serialization(self) -> None:
        serialized = serialize_example(EXAMPLE, state_tokens=0)
        expected = (EXAMPLE.input + "\n" + EXAMPLE.target).encode("utf-8")
        self.assertEqual(serialized.token_ids, list(expected))
        self.assertNotIn(STATE_TOKEN_ID, serialized.token_ids)

    def test_state_slots_follow_each_complete_action_and_are_not_loss_targets(self) -> None:
        two_actions = PerStepExample(
            input="<MAP>x</MAP>\n<ACT>right</ACT>\n<ACT>up</ACT>",
            target="<POS>(0, 1)</POS>",
        )
        serialized = serialize_example(two_actions, state_tokens=2)
        first_action_end = two_actions.input.index("</ACT>") + len("</ACT>")
        first_action_end_bytes = len(two_actions.input[:first_action_end].encode("utf-8"))
        self.assertEqual(
            serialized.token_ids[first_action_end_bytes : first_action_end_bytes + 2],
            [STATE_TOKEN_ID, STATE_TOKEN_ID],
        )
        self.assertEqual(serialized.token_ids[first_action_end_bytes + 2], ord("\n"))

        second_action_end = two_actions.input.rindex("</ACT>") + len("</ACT>")
        second_action_end_bytes = len(two_actions.input[:second_action_end].encode("utf-8")) + 2
        self.assertEqual(
            serialized.token_ids[second_action_end_bytes : second_action_end_bytes + 2],
            [STATE_TOKEN_ID, STATE_TOKEN_ID],
        )
        self.assertEqual(serialized.token_ids[second_action_end_bytes + 2], ord("\n"))

        batch = collate_byte_examples([two_actions], context_length=128, state_tokens=2)
        labels = batch["labels"][0]
        self.assertTrue(all(label == IGNORE_INDEX for label in labels[: serialized.target_start - 1]))
        self.assertEqual(labels[serialized.target_start - 1], ord("<"))
        self.assertNotIn(STATE_TOKEN_ID, labels)

    def test_loss_labels_start_with_first_target_byte(self) -> None:
        batch = collate_byte_examples([EXAMPLE], context_length=128)
        ids = serialize_example(EXAMPLE).token_ids
        labels = batch["labels"][0]

        first_active = next(index for index, label in enumerate(labels) if label != IGNORE_INDEX)
        self.assertEqual(first_active, len((EXAMPLE.input + "\n").encode("utf-8")) - 1)
        self.assertEqual(labels[first_active], ord("<"))
        self.assertEqual(labels[-1], ids[-1])
        self.assertTrue(all(label == IGNORE_INDEX for label in labels[:first_active]))

    def test_padding_is_not_a_loss_target(self) -> None:
        shorter = PerStepExample(input="<MAP>x</MAP>\n<ACT>up</ACT>", target="<POS>(0, 0)</POS>")
        batch = collate_byte_examples([EXAMPLE, shorter], context_length=128)
        short_row = batch["input_ids"][1]
        short_labels = batch["labels"][1]
        self.assertIn(PAD_TOKEN_ID, short_row)
        first_pad = short_row.index(PAD_TOKEN_ID)
        self.assertTrue(all(label == IGNORE_INDEX for label in short_labels[first_pad:]))

    def test_context_overflow_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds context_length"):
            collate_byte_examples([EXAMPLE], context_length=4)

    def test_negative_state_slots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "state_tokens must be non-negative"):
            serialize_example(EXAMPLE, state_tokens=-1)


if __name__ == "__main__":
    unittest.main()
