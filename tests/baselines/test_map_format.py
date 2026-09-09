from __future__ import annotations

import unittest

from agimaze_predict.baselines.llama_memory.map_format import map_rows


class MapFormatTest(unittest.TestCase):
    def test_map_rows_restore_copy_paste_trimmed_trailing_spaces(self) -> None:
        rows = map_rows(
            "+---+---+---+\n|           |\n+   +   +   +\n| S       T |\n"
            "+   +   +---+\n|         K\n+   +---+---+"
        )

        self.assertEqual(rows, (
            "+---+---+---+",
            "|           |",
            "+   +   +   +",
            "| S       T |",
            "+   +   +---+",
            "|         K  ",
            "+   +---+---+",
        ))

    def test_map_rows_accept_dataset_style_map_tags_and_outer_blank_lines(self) -> None:
        rows = map_rows("\n<MAP>\n+---+\n| S |\n+---+\n</MAP>\n")

        self.assertEqual(rows, ("+---+", "| S |", "+---+"))

    def test_map_rows_reject_blank_rows_inside_map(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty row"):
            map_rows("+---+\n\n+---+")


if __name__ == "__main__":
    unittest.main()
