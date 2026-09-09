"""Greedy generation through a saved Llama + spatial-memory checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .map_format import map_rows
from .model import LlamaWithSpatialMemory
from .spatial_memory import SpatialMemoryConfig


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the Llama extra first: pip install -e '.[llama]'") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = getattr(encoded, "input_ids", None)
    if values is None and isinstance(encoded, dict):
        values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def _read_text(*, value: str | None, path: Path | None, option: str) -> str:
    if (value is None) == (path is None):
        raise ValueError(f"provide exactly one of --{option} or --{option}-file")
    return value if value is not None else path.read_text(encoding="utf-8").strip("\r\n")


def _visual_map(map_text: str, config: SpatialMemoryConfig, *, device: torch.device) -> Tensor:
    rows = map_rows(map_text)
    if len(rows) > config.canvas_height or len(rows[0]) > config.canvas_width:
        raise ValueError(
            f"map {len(rows)}x{len(rows[0])} does not fit canvas "
            f"{config.canvas_height}x{config.canvas_width}"
        )
    result = torch.full(
        (1, config.canvas_height, config.canvas_width), ord(" "), dtype=torch.long, device=device
    )
    for row_index, row in enumerate(rows):
        codes = [ord(character) for character in row]
        if any(code >= 256 for code in codes):
            raise ValueError("map contains a non-byte character")
        result[0, row_index, : len(codes)] = torch.tensor(codes, dtype=torch.long, device=device)
    return result


def _action_tensors(actions: Sequence[str], tokenizer, *, pad_token_id: int, device: torch.device) -> tuple[Tensor, Tensor]:
    if any(not action for action in actions):
        raise ValueError("each --action must be non-empty")
    encoded = [_token_ids(tokenizer, action) for action in actions]
    if any(not ids for ids in encoded):
        raise ValueError("each action must produce at least one tokenizer token")
    # Retain one fully masked dummy slot at episode start.  The workspace then
    # simply returns the map-derived initial state without applying an action.
    width = max(map(len, encoded), default=1)
    input_ids = torch.full((1, max(1, len(encoded)), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((1, max(1, len(encoded)), width), dtype=torch.long, device=device)
    for index, ids in enumerate(encoded):
        input_ids[0, index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[0, index, : len(ids)] = 1
    return input_ids, mask


def load_checkpoint(checkpoint: Path, *, device: torch.device) -> tuple[LlamaWithSpatialMemory, object]:
    """Load the frozen backbone plus the trained workspace/projection."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint}")
    # The checkpoint contains run metadata (including Path values), in addition
    # to tensor state dictionaries.  It is an explicitly supplied local file.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "agimaze_predict.llama_memory.v1":
        raise ValueError("unsupported spatial-memory checkpoint format")
    try:
        spatial_config = SpatialMemoryConfig(**saved["spatial_config"])
        base_model = str(saved["base_model"])
    except (KeyError, TypeError) as exc:
        raise ValueError("checkpoint lacks spatial_config or base_model") from exc

    AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    lora = saved.get("lora")
    if lora is not None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
        except ImportError as exc:
            raise RuntimeError("this checkpoint has LoRA weights; install the Llama extra") from exc
        backbone = get_peft_model(backbone, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["rank"]),
            lora_alpha=2 * int(lora["rank"]),
            lora_dropout=0.0,
            target_modules=lora["target_modules"],
        ))
        set_peft_model_state_dict(backbone, lora["state_dict"])
    model = LlamaWithSpatialMemory(backbone, spatial_config)
    model.workspace.load_state_dict(saved["workspace"])
    model.memory_projection.load_state_dict(saved["memory_projection"])
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def greedy_generate(
    model: LlamaWithSpatialMemory,
    tokenizer,
    *,
    prompt: str,
    map_text: str,
    actions: Sequence[str],
    max_new_tokens: int,
    stop: str | None = None,
) -> str:
    """Generate after ``prompt`` while injecting state built from map/actions."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    prompt_ids = _token_ids(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("prompt must produce at least one tokenizer token")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    visual_maps = _visual_map(map_text, model.workspace.config, device=device)
    action_input_ids, action_attention_mask = _action_tensors(
        actions, tokenizer, pad_token_id=pad_token_id, device=device
    )
    prompt_lengths = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_maps=visual_maps,
            action_input_ids=action_input_ids,
            action_attention_mask=action_attention_mask,
            prompt_lengths=prompt_lengths,
        )
        next_id = int(output.logits[0, -1].argmax().item())
        generated.append(next_id)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat((input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)), dim=1)
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        if stop is not None and stop in decoded:
            return decoded.split(stop, maxsplit=1)[0] + stop
    return tokenizer.decode(generated, skip_special_tokens=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="single-file training checkpoint")
    parser.add_argument("--prompt", help="arbitrary text prefix for Llama")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the text prefix")
    parser.add_argument("--map", dest="map_text", help="raw rectangular map, without <MAP> tags")
    parser.add_argument("--map-file", type=Path, help="UTF-8 file containing the raw map")
    parser.add_argument("--action", action="append", default=[], help="completed action; repeat in chronological order; omit at episode start")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--stop", help="return after this generated text is complete")
    parser.add_argument("--device", help="defaults to cuda when available, otherwise cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        prompt = _read_text(value=args.prompt, path=args.prompt_file, option="prompt")
        map_text = _read_text(value=args.map_text, path=args.map_file, option="map")
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, tokenizer = load_checkpoint(args.checkpoint, device=device)
        text = greedy_generate(
            model, tokenizer, prompt=prompt, map_text=map_text, actions=args.action,
            max_new_tokens=args.max_new_tokens, stop=args.stop,
        )
        print(json.dumps({"completion": text}, ensure_ascii=False))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        build_parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
