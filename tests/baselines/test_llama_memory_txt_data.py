from __future__ import annotations

import unittest

from agimaze_predict.baselines.llama_memory.txt_data import collate_txt_rollouts, txt_rollouts_from_trace
from agimaze_predict.data.txt_trace import TxtTrace, TxtTransition


class _Tokenizer:
    eos_token_id = 255
    pad_token_id = 254

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


TRACE = TxtTrace(
    map_text="+---+\n| S |\n+---+",
    start_text="board: 1x1\nstart: (0, 0)",
    transitions=(
        TxtTransition("right", "first"),
        TxtTransition("left", "second"),
    ),
)


class LlamaMemoryTxtDataTest(unittest.TestCase):
    def test_rollouts_retain_prior_txt_and_all_completed_actions(self) -> None:
        rollouts = txt_rollouts_from_trace(TRACE, initial_context="START", depth=2)

        self.assertEqual(len(rollouts), 2)
        self.assertEqual(rollouts[0].actions, ("right",))
        self.assertEqual(rollouts[1].actions, ("right", "left"))
        self.assertIn("<TXT>first</TXT>", rollouts[1].source_segments)
        self.assertEqual(rollouts[1].source_segments[-1], "<ACT>left</ACT>")

    def test_collator_marks_every_act_boundary_and_only_txt_target(self) -> None:
        rollout = txt_rollouts_from_trace(TRACE, initial_context="START", depth=2)[1]
        batch = collate_txt_rollouts([rollout], tokenizer=_Tokenizer(), canvas_height=3, canvas_width=5, pad_token_id=254)

        self.assertEqual(batch["event_mask"], [[1, 1]])
        self.assertEqual(len(batch["event_token_positions"][0]), 2)
        self.assertLess(batch["event_token_positions"][0][0], batch["event_token_positions"][0][1])
        prompt_length = batch["prompt_lengths"][0]
        self.assertEqual(batch["event_token_positions"][0][-1], prompt_length)
        self.assertTrue(all(label == -100 for label in batch["labels"][0][:prompt_length]))
        self.assertEqual(batch["target_token_ids"][0][-1], 255)
        self.assertTrue(all(label != -100 for label in batch["labels"][0][prompt_length:prompt_length + len(batch["target_token_ids"][0])]))


if __name__ == "__main__":
    unittest.main()
