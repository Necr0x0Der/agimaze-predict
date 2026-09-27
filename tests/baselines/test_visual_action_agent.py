"""Live-agent serialization must agree with visual Transformer training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from agimaze_predict.baselines.visual_transformer.model import VisualTransformerConfig
from agimaze_predict.baselines.visual_transformer.tokenizer import collate_visual_examples, serialize_visual_example
from agimaze_predict.data.prepared import PreparedExample

# Captured /api/start map for a 3x3 keys stage. The open right border has a
# trailing space; stripping it would change the training-time canvas.
SERVER_MAP = "+---+---+---+\n|           |\n+   +   +   +\n| T          \n+---+   +   +\n| K       S |\n+---+---+---+"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "visual_action_agent.py"
spec = importlib.util.spec_from_file_location("visual_action_agent", SCRIPT)
agent = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(agent)


def test_server_map_and_history_match_training_serialization() -> None:
    # The open right border is a trailing space and must remain visible.
    map_ascii = SERVER_MAP
    assert map_ascii.splitlines()[3].endswith(" ")
    config = VisualTransformerConfig(context_length=192, canvas_height=9, canvas_width=17)
    visual_map = agent.prepare_map(map_ascii, config, torch.device("cpu"))
    example = PreparedExample(input=f"<MAP>\n{map_ascii}\n</MAP>\n<ACT>left</ACT>", target="<POS>(2, 1)</POS>")
    batch = collate_visual_examples([example], context_length=192, canvas_height=9, canvas_width=17)
    assert visual_map.tolist() == batch["visual_maps"]
    assert visual_map[0, 3, 12] == ord(" ")  # open right border, not a wall
    item = serialize_visual_example(example)
    assert agent.action_prefix(["left"]) == bytes(item.token_ids[:item.target_start]).split(b"\n<POS>")[0] + b"\n<ACT>"
    ids, events, counts = agent.model_inputs(agent.action_prefix(["left"]), torch.device("cpu"))
    assert events.tolist() == [list(item.event_positions)]
    assert counts[0, events[0, 0] - 1] == 0
    assert counts[0, events[0, 0]] == 1
    assert ids[0, -5:].tolist() == list(b"<ACT>")


def test_first_action_has_no_completed_visual_events() -> None:
    ids, events, counts = agent.model_inputs(agent.action_prefix([]), torch.device("cpu"))
    assert ids.tolist() == [list(b"<ACT>")]
    assert events.shape == (1, 0)
    assert counts.tolist() == [[0] * 5]


@pytest.mark.parametrize("raw,expected,noisy", [
    (b"down</ACT>", "down", False),
    (b"down</ACT>garbage", "down", True),
    (b"down?!", "down", True),
    (b"dow", None, False),
    (b"unknown</ACT>", None, False),
])
def test_action_prefix_tolerance(raw: bytes, expected: str | None, noisy: bool) -> None:
    assert agent.parse_action(raw) == (expected, noisy)


def test_episode_keeps_only_actions_accepted_by_server_and_warns_on_noise() -> None:
    map_ascii = SERVER_MAP
    model = SimpleNamespace(config=VisualTransformerConfig(context_length=192, canvas_height=9, canvas_width=17),
                            parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]))
    responses = [
        {"id": "test-id", "map": map_ascii, "task_meta": {"max_steps": 3}},
        {"text": "moved", "done": False},
        {"text": "escaped", "done": True},
    ]
    seen_histories = []
    predictions = iter([("left", b"left???", True), ("up", b"up</ACT>", False)])
    def predict_next(_model, _map, history, **_kwargs):
        seen_histories.append(list(history))
        return next(predictions)

    with patch.object(agent, "http_post_json", side_effect=responses) as post, \
         patch.object(agent, "predict_action", side_effect=predict_next), \
         pytest.warns(UserWarning, match="recognized by prefix"):
        result = agent.run_episode(model, base_url="http://127.0.0.1:8000", path="TRAINING/S0-keys/STAGE-01", verbose=False)
    assert result["success"] and result["actions"] == ["left", "up"] and result["warnings"] == 1
    assert post.call_args_list[1].args[1] == {"id": "test-id", "action": "left", "seq": 1}
    assert post.call_args_list[2].args[1] == {"id": "test-id", "action": "up", "seq": 2}
    assert seen_histories == [[], ["left"]]
