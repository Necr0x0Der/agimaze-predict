from __future__ import annotations

from dataclasses import dataclass

from agimaze_predict.baselines.llama_memory.data import collate_examples, parse_example
from agimaze_predict.data.prepared import PreparedExample


@dataclass
class _Encoded:
    input_ids: list[int]


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> _Encoded:
        return _Encoded([ord(character) for character in text])


def _example() -> PreparedExample:
    return PreparedExample(
        input="<MAP>+-+\n|@|\n+-+</MAP>\n<ACT>right</ACT>\n<ACT>down</ACT>",
        target="<POS>(1, 1)</POS>",
    )


def test_parse_separates_map_actions_and_target() -> None:
    item = parse_example(_example())
    assert item.map_rows == ("+-+", "|@|", "+-+")
    assert item.actions == ("right", "down")
    assert item.target == "<POS>(1, 1)</POS>"


def test_collator_masks_prompt_and_keeps_target_labels() -> None:
    batch = collate_examples([_example()], tokenizer=_Tokenizer(), canvas_height=4, canvas_width=5, pad_token_id=99)
    prompt_length = batch["prompt_lengths"][0]
    assert all(label == -100 for label in batch["labels"][0][:prompt_length])
    assert batch["labels"][0][prompt_length] == ord("<")
    assert batch["labels"][0][-1] == 99
    assert batch["visual_maps"][0][1][:3] == [ord("|"), ord("@"), ord("|")]
    assert batch["action_attention_mask"][0] == [[1] * 5, [1] * 4]
