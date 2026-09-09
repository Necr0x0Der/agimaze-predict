from __future__ import annotations

import unittest

from agimaze_predict.baselines.llama_memory.icl_api import (
    SYSTEM_PROMPT,
    build_messages,
    evaluate_examples,
    extract_pos,
    select_demonstrations,
)
from agimaze_predict.data.prepared import PreparedExample


EXAMPLE = PreparedExample(
    input="<MAP>tiny</MAP>\n<ACT>right</ACT>",
    target="<POS>(0, 1)</POS>",
)


class LlamaMemoryIclApiTest(unittest.TestCase):
    def test_prompt_contains_general_rules_demo_and_current_example(self) -> None:
        messages = build_messages(EXAMPLE, demonstrations=[EXAMPLE])

        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": EXAMPLE.input})
        self.assertEqual(messages[2], {"role": "assistant", "content": EXAMPLE.target})
        self.assertEqual(messages[3], {"role": "user", "content": EXAMPLE.input})

    def test_extract_pos_accepts_exactly_one_structured_answer(self) -> None:
        self.assertEqual(extract_pos("Answer: <POS>(12, 3)</POS>"), "<POS>(12, 3)</POS>")
        self.assertIsNone(extract_pos("<POS>(0, 1)</POS> or <POS>(0, 2)</POS>"))
        self.assertIsNone(extract_pos("(0, 1)"))

    def test_evaluation_uses_structured_exact_match_not_raw_completion(self) -> None:
        predictions = evaluate_examples(
            [EXAMPLE],
            demonstrations=(),
            complete=lambda messages: "<POS>(0, 1)</POS>\n",
        )

        self.assertEqual(len(predictions), 1)
        self.assertTrue(predictions[0].correct)
        self.assertEqual(predictions[0].extracted, EXAMPLE.target)

    def test_few_shot_sampling_is_deterministic(self) -> None:
        examples = [PreparedExample(input=f"<MAP>{index}</MAP>\n<ACT>up</ACT>", target="<POS>(0, 0)</POS>") for index in range(5)]

        self.assertEqual(
            select_demonstrations(examples, count=3, seed=7),
            select_demonstrations(examples, count=3, seed=7),
        )


if __name__ == "__main__":
    unittest.main()
