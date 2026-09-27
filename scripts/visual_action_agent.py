#!/usr/bin/env python3
"""Run a trained visual-only ACT predictor against the AGI Maze HTTP API.

Example:
  /opt/anaconda3/bin/python3 scripts/visual_action_agent.py \
    --checkpoint runs/visual-rational-actions-3x3-keys-visual.pt \
    --base-url http://127.0.0.1:8000 --path TRAINING/S0-keys/STAGE-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from agimaze_predict.baselines.visual_transformer.model import VisualTransformer, VisualTransformerConfig
from agimaze_predict.baselines.visual_transformer.tokenizer import _canvas, _map_rows

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "runs/visual-rational-actions-3x3-keys-visual.pt"
ACTIONS = ("down", "left", "right", "up")
ACT_CLOSE = b"</ACT>"


def http_post_json(url: str, payload: dict, *, timeout: float = 30) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return json.loads(error.read().decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise


def load_model(path: Path, device: torch.device) -> tuple[VisualTransformer, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("predict_actions"):
        raise ValueError("checkpoint was not trained to predict actions")
    config = VisualTransformerConfig(**checkpoint["model_config"])
    if config.pos_readout != "visual_only":
        raise ValueError("this agent requires a visual_only action checkpoint")
    model = VisualTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def prepare_map(map_ascii: str, config: VisualTransformerConfig, device: torch.device) -> torch.Tensor:
    """Preserve spaces (including open border exits); match training's MAP canvas."""
    if not isinstance(map_ascii, str) or not map_ascii:
        raise ValueError("server did not supply a non-empty ASCII map")
    rows = _map_rows(f"<MAP>\n{map_ascii}\n</MAP>")
    return torch.tensor([_canvas(rows, height=config.canvas_height, width=config.canvas_width,
                                  blank_char=ord(" "))], dtype=torch.long, device=device)


def action_prefix(history: list[str]) -> bytes:
    if any(action not in ACTIONS for action in history):
        raise ValueError("history contains an invalid action")
    return ("\n".join([*(f"<ACT>{action}</ACT>" for action in history), "<ACT>"])).encode("ascii")


def model_inputs(prefix: bytes, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The first action has zero completed events; never use the target action's frame."""
    events: list[int] = []
    offset = 0
    while (at := prefix.find(ACT_CLOSE, offset)) >= 0:
        events.append(at + len(ACT_CLOSE) - 1)
        offset = at + len(ACT_CLOSE)
    counts = [sum(event <= index for event in events) for index in range(len(prefix))]
    return (
        torch.tensor([list(prefix)], device=device, dtype=torch.long),
        torch.tensor([events], device=device, dtype=torch.long),
        torch.tensor([counts], device=device, dtype=torch.long),
    )


def parse_action(generated: bytes) -> tuple[str | None, bool]:
    """Accept an exact action or an unambiguous action prefix with trailing noise."""
    text = generated.decode("utf-8", errors="replace")
    for action in ACTIONS:
        if text == action + "</ACT>":
            return action, False
    for action in ACTIONS:
        if text.startswith(action):
            return action, True
    return None, False


@torch.inference_mode()
def predict_action(model: VisualTransformer, visual_map: torch.Tensor, history: list[str],
                   *, max_new_bytes: int = 20) -> tuple[str | None, bytes, bool]:
    prefix = action_prefix(history)
    remaining = model.config.context_length - len(prefix)
    if remaining <= 0:
        raise ValueError(f"action history exceeds model context ({model.config.context_length} bytes); stopping")
    device = visual_map.device
    generated = bytearray()
    for _ in range(min(max_new_bytes, remaining)):
        ids, events, counts = model_inputs(prefix + generated, device)
        # target_input_ids is needed by the visual-only POS path, but action
        # logits come exclusively from the causal text path of this same model.
        _, logits = model(ids, visual_maps=visual_map, event_positions=events,
                          event_counts=counts, target_input_ids=torch.tensor([[ord(">")]], device=device),
                          return_action_logits=True)
        next_id = int(logits[0, -1].argmax().item())
        if next_id > 255:  # PAD is not a byte of the ACT answer.
            break
        generated.append(next_id)
        if generated.endswith(ACT_CLOSE):
            break
    action, noisy = parse_action(bytes(generated))
    return action, bytes(generated), noisy


def run_episode(model: VisualTransformer, *, base_url: str, path: str,
                seed: int | None = None, max_steps: int | None = None,
                agent: str = "visual_action", verbose: bool = True,
                max_new_bytes: int = 20) -> dict:
    start_time = time.monotonic()
    base = base_url.rstrip("/")
    response = http_post_json(base + "/api/start", {
        "path": path, "seed": seed, "client": "api", "agent": agent,
    })
    sid = response.get("id")
    if not sid:
        raise RuntimeError(f"/api/start did not return a session id: {response}")
    map_ascii = response.get("map")
    visual_map = prepare_map(map_ascii, model.config, next(model.parameters()).device)
    rows = _map_rows(f"<MAP>\n{map_ascii}\n</MAP>")
    if verbose:
        print(f"Session {sid}; map {len(rows)}×{len(rows[0])}; model canvas "
              f"{model.config.canvas_height}×{model.config.canvas_width}")
        print(map_ascii)
    task_limit = response.get("task_meta", {}).get("max_steps")
    limit = max_steps if max_steps is not None else task_limit
    if limit is None:
        raise ValueError("server supplied no max_steps; pass --max-steps")
    if limit <= 0:
        raise ValueError("max_steps must be positive")
    history: list[str] = []
    warnings_count = 0
    done = False
    error = None
    for step in range(1, limit + 1):
        try:
            action, raw, noisy = predict_action(model, visual_map, history,
                                                max_new_bytes=max_new_bytes)
        except ValueError as exc:
            error = str(exc)
            break
        if action is None:
            error = f"unrecognized model output at step {step}: {raw!r}"
            break
        if noisy:
            warnings_count += 1
            warnings.warn(f"step {step}: action recognized by prefix from {raw!r}", stacklevel=2)
        result = http_post_json(base + "/api/step", {"id": sid, "action": action, "seq": step})
        if result.get("error"):
            error = f"server rejected step {step}: {result['error']}"
            break
        history.append(action)
        done = bool(result.get("done"))
        if verbose:
            print(f"[{step:03d}] {action:>5} ({raw!r}) | {result.get('text', '')}", flush=True)
        if done:
            break
    return {"success": done, "steps": len(history), "actions": history,
            "session_id": sid, "warnings": warnings_count, "error": error,
            "elapsed_s": round(time.monotonic() - start_time, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--path", default="TRAINING/S0-keys/STAGE-01")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--agent", default="visual_action")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-bytes", type=int, default=20)
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()
    if args.max_new_bytes <= 0:
        parser.error("--max-new-bytes must be positive")
    model, checkpoint = load_model(args.checkpoint, torch.device(args.device))
    if not args.no_verbose:
        print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')})")
    result = run_episode(model, base_url=args.base_url, path=args.path, seed=args.seed,
                         max_steps=args.max_steps, agent=args.agent, verbose=not args.no_verbose,
                         max_new_bytes=args.max_new_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
