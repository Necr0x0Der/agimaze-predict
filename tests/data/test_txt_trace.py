from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agimaze_predict.data.txt_trace import TxtTraceDataset, TxtTraceFormatError, read_txt_trace_jsonl


def trace(*, terminal: bool = True) -> str:
    end = "\n<END>Win</END>" if terminal else ""
    return (
        "<MAP>map body</MAP>\n"
        "<START>start body</START>\n"
        "<ACT>right</ACT>\n<TXT>first observation</TXT>\n"
        "<ACT>down</ACT>\n<TXT>second observation</TXT>\n"
        "<ACT>left</ACT>\n<TXT>third observation</TXT>"
        f"{end}\n"
    )


class TxtTraceDatasetTest(unittest.TestCase):
    def write_dataset(self, root: Path, text: str) -> Path:
        path = root / "traces.jsonl"
        path.write_text(json.dumps({"trace": text}) + "\n", encoding="utf-8")
        return path

    def test_map_and_start_choose_exactly_one_initial_block_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_dataset(Path(temp_dir), trace())
            map_example = TxtTraceDataset(path, initial_context="MAP", depth=2)[0]
            start_example = TxtTraceDataset(path, initial_context="START", depth=1)[0]

        self.assertEqual(
            map_example.text,
            "<MAP>map body</MAP>\n<ACT>right</ACT>\n<TXT>first observation</TXT>\n"
            "<ACT>down</ACT>\n<TXT>second observation</TXT>",
        )
        self.assertNotIn("<START>", map_example.text)
        self.assertEqual(
            start_example.text, "<START>start body</START>\n<ACT>right</ACT>\n<TXT>first observation</TXT>"
        )
        self.assertNotIn("<MAP>", start_example.text)

    def test_short_trace_is_not_discarded_at_larger_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_dataset(
                Path(temp_dir), "<MAP>m</MAP>\n<START>s</START>\n<ACT>a</ACT>\n<TXT>t</TXT>\n"
            )
            example = TxtTraceDataset(path, initial_context="MAP", depth=8)[0]
        self.assertEqual(example.text, "<MAP>m</MAP>\n<ACT>a</ACT>\n<TXT>t</TXT>")

    def test_rejects_bad_order_and_dangling_action(self) -> None:
        cases = (
            "<START>s</START>\n<MAP>m</MAP>\n<ACT>a</ACT>\n<TXT>t</TXT>",
            "<MAP>m</MAP>\n<START>s</START>\n<TXT>t</TXT>",
            "<MAP>m</MAP>\n<START>s</START>\n<ACT>a</ACT>",
            "<MAP>m</MAP>\n<START>s</START>\n<ACT>a</ACT>\n<ACT>b</ACT>\n<TXT>t</TXT>",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for number, invalid in enumerate(cases):
                path = self.write_dataset(root, invalid)
                with self.assertRaises(TxtTraceFormatError):
                    read_txt_trace_jsonl(path)


if __name__ == "__main__":
    unittest.main()
