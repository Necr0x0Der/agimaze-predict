from __future__ import annotations

import pytest

from agimaze_predict.baselines.llama_memory.map_format import map_rows


torch = pytest.importorskip("torch")

from agimaze_predict.baselines.llama_memory.generate import _action_tensors  # noqa: E402


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}



def test_action_tensors_allow_an_empty_action_history() -> None:
    input_ids, mask = _action_tensors([], _Tokenizer(), pad_token_id=99, device=torch.device("cpu"))

    assert input_ids.tolist() == [[[99]]]
    assert mask.tolist() == [[[0]]]


def test_action_tensors_preserve_each_completed_action() -> None:
    input_ids, mask = _action_tensors(["up", "left"], _Tokenizer(), pad_token_id=99, device=torch.device("cpu"))

    assert input_ids.tolist() == [[[ord("u"), ord("p"), 99, 99], [ord("l"), ord("e"), ord("f"), ord("t")]]]
    assert mask.tolist() == [[[1, 1, 0, 0], [1, 1, 1, 1]]]
