from __future__ import annotations

import unittest

from agimaze_predict.baselines.byte_transformer.tokenizer import (
    IGNORE_INDEX,
    PAD_TOKEN_ID,
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


if __name__ == "__main__":
    unittest.main()
