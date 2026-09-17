#!/usr/bin/env python3
"""Neural network agent for AGI Maze using trained byte-transformer.

This agent uses a trained checkpoint to predict actions instead of 
algorithmic planning (BFS). It demonstrates learned maze-solving behavior.

Usage:
    python3 scripts/neural_agent.py \
      --checkpoint runs/model.pt \
      --base-url http://127.0.0.1:8000 \
      --path "TRAINING/S0-keys/STAGE-01" \
      --seed 42
"""

import sys
import json
import argparse
import urllib.request
import urllib.parse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add agimaze-predict src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.byte_transformer.model import ByteTransformer, ByteTransformerConfig
from agimaze_predict.baselines.byte_transformer.tokenizer import PAD_TOKEN_ID


def http_post_json(url: str, data: dict, timeout_s: int = 30) -> dict:
    """Simple HTTP POST helper."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


@dataclass
class Episode:
    """Episode result."""
    success: bool
    total_steps: int
    session_id: str
    map_ascii: str
    actions: list[str]
    observations: list[str]
    predicted_correctly: int  # How many actions were optimal


def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> tuple[ByteTransformer, dict]:
    """Load trained model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model from saved config
    model_config = ByteTransformerConfig(**checkpoint["model_config"])
    model = ByteTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def prepare_input_for_model(map_ascii: str, action_history: list[str]) -> str:
    """Format map and actions as model input.
    
    Format: <MAP>...</MAP>\n<ACT>action1</ACT>\n<ACT>action2</ACT>...
    """
    
    input_parts = [f"<MAP>{map_ascii}</MAP>"]
    
    for action in action_history:
        input_parts.append(f"<ACT>{action}</ACT>")
    
    return "\n".join(input_parts)


def predict_next_action(
    model: ByteTransformer,
    input_text: str,
    device: str = "cpu",
    max_new_tokens: int = 20,
) -> Optional[str]:
    """Use model to predict next action via greedy decoding.
    
    Returns action name (up/down/left/right) or None if failed.
    """
    
    # Encode input
    input_bytes = list(input_text.encode("utf-8")) + [ord("\n")]
    input_ids = torch.tensor([input_bytes], dtype=torch.long, device=device)
    
    # Greedy decode until we complete <ACT>...</ACT>
    generated_bytes = []
    
    for _ in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            logits = model(input_ids)
        
        # Greedy: take most probable next token
        next_token = logits[0, -1, :].argmax().item()
        
        # Stop at padding
        if next_token == PAD_TOKEN_ID:
            break
        
        generated_bytes.append(next_token)
        
        # Append to input for next iteration
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
        
        # Check if we completed </ACT>
        if len(generated_bytes) >= 6:
            last_6 = bytes(generated_bytes[-6:])
            if last_6 == b"</ACT>":
                break
    
    # Parse generated text
    try:
        generated_text = bytes(generated_bytes).decode("utf-8")
        
        # Extract action from <ACT>action</ACT>
        if generated_text.startswith("<ACT>") and "</ACT>" in generated_text:
            action = generated_text[5:generated_text.index("</ACT>")]
            if action in {"up", "down", "left", "right"}:
                return action
    except:
        pass
    
    return None


def run_episode(
    model: ByteTransformer,
    base_url: str,
    path: str,
    seed: Optional[int] = None,
    max_steps: int = 50,
    device: str = "cpu",
    verbose: bool = True,
) -> Episode:
    """Run one episode with neural agent."""
    
    # Start game
    start_resp = http_post_json(f"{base_url}/api/start", {
        "path": path,
        "seed": seed,
        "client": "api",
        "agent": "neural",
    })
    
    session_id = start_resp["session_id"]
    map_ascii = start_resp["map"]
    
    if verbose:
        print("=" * 60)
        print(f"START: {start_resp['message']}")
        print()
        print("[MAP]")
        print(map_ascii)
        print()
    
    action_history = []
    observations = []
    step = 0
    success = False
    
    while step < max_steps:
        step += 1
        
        # Prepare input for model
        model_input = prepare_input_for_model(map_ascii, action_history)
        
        # Predict next action
        predicted_action = predict_next_action(model, model_input, device=device)
        
        if predicted_action is None:
            if verbose:
                print(f"[{step:03d}] ERROR: Model failed to predict valid action")
            break
        
        # Execute action
        move_resp = http_post_json(f"{base_url}/api/move", {
            "session_id": session_id,
            "action": predicted_action,
        })
        
        action_history.append(predicted_action)
        observations.append(move_resp["message"])
        
        if verbose:
            print(f"[{step:03d}] {predicted_action:>5} | {move_resp['message']}")
        
        # Check if done
        if move_resp.get("done"):
            success = move_resp.get("success", False)
            break
    
    if verbose:
        print()
        if success:
            print("✓ SUCCESS!")
        else:
            print("✗ FAILED")
        print("=" * 60)
    
    return Episode(
        success=success,
        total_steps=step,
        session_id=session_id,
        map_ascii=map_ascii,
        actions=action_history,
        observations=observations,
        predicted_correctly=0,  # TODO: compare with optimal if needed
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--path", type=str, required=True, help="Maze path (e.g., TRAINING/S0-keys/STAGE-01)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model, checkpoint_info = load_checkpoint(checkpoint_path, device=args.device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trained on: {checkpoint_info.get('train_datasets', 'unknown')[:1]}")
    print(f"Epoch: {checkpoint_info.get('epoch', '?')}")
    if 'val_losses' in checkpoint_info:
        val_losses = checkpoint_info['val_losses']
        print(f"Val loss: {val_losses['total']:.4f} (ACT: {val_losses['act']:.4f}, POS: {val_losses['pos']:.4f})")
    print()
    
    # Run episodes
    results = []
    
    for i in range(args.num_episodes):
        if args.num_episodes > 1:
            print(f"\n{'='*60}")
            print(f"EPISODE {i+1}/{args.num_episodes}")
            print('='*60)
        
        episode = run_episode(
            model=model,
            base_url=args.base_url,
            path=args.path,
            seed=args.seed if args.seed is not None else None,
            max_steps=args.max_steps,
            device=args.device,
            verbose=args.verbose,
        )
        
        results.append(episode)
    
    # Summary
    if args.num_episodes > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        successes = sum(1 for ep in results if ep.success)
        total_steps = sum(ep.total_steps for ep in results)
        
        print(f"Success rate: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")
        print(f"Average steps: {total_steps/args.num_episodes:.1f}")
        print("=" * 60)
    
    # Return 0 if last episode succeeded (for scripting)
    return 0 if results[-1].success else 1


if __name__ == "__main__":
    sys.exit(main())
