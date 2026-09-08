"""Data preparation for the Llama + spatial-memory proof of concept.

The map is intentionally excluded from Llama's textual prompt.  It is passed
only to the 2-D workspace, whereas the action text remains visible to Llama.
This makes the memory path explicit and avoids silently solving the task by
re-reading an ASCII map from the language context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from agimaze_predict.data.prepared import PreparedExample

_MAP_RE = re.compile(r"\A<MAP>(.*?)</MAP>(.*)\Z", re.DOTALL)
_ACTION_RE = re.compile(r"<ACT>(.*?)</ACT>", re.DOTALL)


class Tokenizer(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> object: ...


@dataclass(frozen=True)
class LlamaMemoryExample:
    """A map canvas plus an action-only language prompt and answer text."""

    map_rows: tuple[str, ...]
    action_text: str
    actions: tuple[str, ...]
    target: str


def parse_example(example: PreparedExample) -> LlamaMemoryExample:
    match = _MAP_RE.fullmatch(example.input)
    if match is None:
        raise ValueError("input must begin with one <MAP>...</MAP> block")
    rendered = match.group(1).strip("\n")
    rows = tuple(rendered.split("\n"))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("MAP must be a non-empty rectangular character grid")
    action_text = match.group(2).lstrip("\r\n")
    actions = tuple(_ACTION_RE.findall(action_text))
    if not actions:
        raise ValueError("input must contain at least one complete <ACT> block")
    if not example.target.startswith("<POS>"):
        raise ValueError("target must be a <POS>...</POS> block")
    return LlamaMemoryExample(rows, action_text, actions, example.target)


def _ids(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = getattr(encoded, "input_ids", None)
    if values is None and isinstance(encoded, dict):
        values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def _canvas(rows: tuple[str, ...], height: int, width: int) -> list[list[int]]:
    if len(rows) > height or len(rows[0]) > width:
        raise ValueError(f"MAP {len(rows)}x{len(rows[0])} does not fit canvas {height}x{width}")
    result = [[ord(" ")] * width for _ in range(height)]
    for row, text in enumerate(rows):
        for column, character in enumerate(text):
            code = ord(character)
            if code >= 256:
                raise ValueError(f"MAP character {character!r} is not a byte")
            result[row][column] = code
    return result


def collate_examples(
    examples: Sequence[PreparedExample],
    *,
    tokenizer: Tokenizer,
    canvas_height: int,
    canvas_width: int,
    pad_token_id: int,
) -> dict[str, list]:
    """Build right-padded text/action tensors and target-only labels.

    The model inserts its soft memory tokens at ``prompt_lengths[i]``.  An EOS
    token is appended to each answer so the last visible answer token receives
    a next-token training signal under Hugging Face's causal-LM loss shift.
    """

    if not examples:
        raise ValueError("cannot collate an empty batch")
    parsed = [parse_example(example) for example in examples]
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    action_ids: list[list[list[int]]] = []
    maps: list[list[list[int]]] = []
    targets: list[list[int]] = []
    for item in parsed:
        prompt = _ids(tokenizer, item.action_text + "\n")
        target = _ids(tokenizer, item.target)
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        target.append(tokenizer.eos_token_id)
        sequences.append([*prompt, *target])
        prompt_lengths.append(len(prompt))
        targets.append(target)
        action_ids.append([_ids(tokenizer, action) for action in item.actions])
        maps.append(_canvas(item.map_rows, canvas_height, canvas_width))

    text_width = max(map(len, sequences))
    action_count = max(map(len, action_ids))
    action_width = max(len(ids) for batch in action_ids for ids in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    action_input_ids: list[list[list[int]]] = []
    action_attention_mask: list[list[list[int]]] = []
    for sequence, prompt_len, per_action in zip(sequences, prompt_lengths, action_ids):
        padding = text_width - len(sequence)
        input_ids.append([*sequence, *([pad_token_id] * padding)])
        attention_mask.append([1] * len(sequence) + [0] * padding)
        labels.append([-100] * prompt_len + sequence[prompt_len:] + [-100] * padding)
        padded_actions = [ids + [pad_token_id] * (action_width - len(ids)) for ids in per_action]
        padded_masks = [[1] * len(ids) + [0] * (action_width - len(ids)) for ids in per_action]
        padded_actions += [[pad_token_id] * action_width] * (action_count - len(per_action))
        padded_masks += [[0] * action_width] * (action_count - len(per_action))
        action_input_ids.append(padded_actions)
        action_attention_mask.append(padded_masks)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_lengths": prompt_lengths,
        "visual_maps": maps,
        "action_input_ids": action_input_ids,
        "action_attention_mask": action_attention_mask,
        "target_token_ids": targets,
    }
