"""Byte-level serialization and batch construction for the first baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agimaze_predict.data.prepared import PreparedExample

BYTE_VOCAB_SIZE = 256
PAD_TOKEN_ID = 256
# State slots are deterministic latent positions, never byte targets.  They use
# one additional vocabulary entry only in state-token experiments, keeping the
# state_tokens=0 model vocabulary/checkpoints exactly as before.
STATE_TOKEN_ID = 257
VOCAB_SIZE = 257
VOCAB_SIZE_WITH_STATE = 258
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SerializedExample:
    """One causal sequence and the first target-byte offset."""

    token_ids: list[int]
    target_start: int


def _input_with_state_slots(model_input: str, *, state_tokens: int) -> list[int]:
    """Encode input, inserting fixed latent slots after every complete ACT block.

    Prepared examples delimit each action as ``<ACT>...</ACT>``.  Slots are
    token IDs rather than textual bytes, so no delimiter is needed between the
    action's final byte and the first slot.  The zero-slot path intentionally
    returns the old literal UTF-8 representation byte-for-byte.
    """

    if state_tokens < 0:
        raise ValueError("state_tokens must be non-negative")
    if state_tokens == 0:
        return list(model_input.encode("utf-8"))

    token_ids: list[int] = []
    cursor = 0
    while True:
        action_end = model_input.find("</ACT>", cursor)
        if action_end < 0:
            token_ids.extend(model_input[cursor:].encode("utf-8"))
            return token_ids
        action_end += len("</ACT>")
        token_ids.extend(model_input[cursor:action_end].encode("utf-8"))
        token_ids.extend([STATE_TOKEN_ID] * state_tokens)
        cursor = action_end


def serialize_example(example: PreparedExample, *, state_tokens: int = 0) -> SerializedExample:
    """Serialize as input + newline + target, with optional post-action slots.

    The target's first byte is the first byte of ``<POS>``.  No separate BOS/EOS
    tokens are used.  With ``state_tokens=0`` this is exactly the original
    literal-byte serialization.  With a positive value, each completed action
    receives that many fixed ``STATE_TOKEN_ID`` positions before the following
    newline/action or target; their hidden activations are the latent state.
    """

    prefix = _input_with_state_slots(example.input, state_tokens=state_tokens)
    prefix.append(ord("\n"))
    target = list(example.target.encode("utf-8"))
    return SerializedExample(token_ids=[*prefix, *target], target_start=len(prefix))


def collate_byte_examples(
    examples: Sequence[PreparedExample],
    *,
    context_length: int,
    state_tokens: int = 0,
) -> dict[str, list[list[int]]]:
    """Create padded causal input/label arrays with loss only on target bytes.

    ``input_ids[i]`` predicts ``labels[i]``.  Labels corresponding to context,
    padding, or positions after the target use ``IGNORE_INDEX``.  The first
    target byte is therefore predicted from the last prefix byte, as required by
    causal language modelling.
    """

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")

    serialized = [serialize_example(example, state_tokens=state_tokens) for example in examples]
    lengths = [len(item.token_ids) for item in serialized]
    longest = max(lengths)
    if longest > context_length:
        raise ValueError(
            f"serialized example length {longest} exceeds context_length {context_length}; "
            "increase --context-length"
        )

    # The model consumes every byte except the final byte and predicts every
    # following byte.  Pad to the longest sequence in this batch, not the model
    # context length, to avoid unnecessary compute.
    width = longest - 1
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []

    for item in serialized:
        ids = item.token_ids
        row_input = ids[:-1] + [PAD_TOKEN_ID] * (width - (len(ids) - 1))
        row_labels = [IGNORE_INDEX] * width
        # label index i predicts ids[i + 1], so target begins at target_start - 1.
        for index in range(item.target_start - 1, len(ids) - 1):
            row_labels[index] = ids[index + 1]
        input_ids.append(row_input)
        labels.append(row_labels)

    return {"input_ids": input_ids, "labels": labels}
