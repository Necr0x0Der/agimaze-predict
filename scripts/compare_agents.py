#!/usr/bin/env python3
"""Compare neural agent vs rational agent on test set.

Usage:
    python3 scripts/compare_agents.py \
      --checkpoint runs/model.pt \
      --rational-agent /path/to/agimaze-bench/agents/pure_algorithmic_agent.py \
      --base-url http://127.0.0.1:8000 \
      --path "TRAINING/S0-keys/STAGE-01" \
      --num-episodes 30 \
      --output comparison_results.json
"""

import sys
import json
import argparse
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import neural agent functions
from neural_agent import load_checkpoint, run_episode as run_neural_episode


@dataclass
class ComparisonResult:
    """Comparison between neural and rational agent on same maze."""
    seed: int
    map_ascii: str
    
    # Neural agent
    neural_success: bool
    neural_steps: int
    neural_actions: list[str]
    
    # Rational agent
    rational_success: bool
    rational_steps: int
    rational_actions: list[str]
    
    # Comparison
    actions_match: bool
    steps_diff: int  # neural - rational (positive = neural took more steps)


def run_rational_agent(
    rational_script: Path,
    base_url: str,
    path: str,
    seed: int,
) -> dict:
    """Run rational agent via subprocess and parse JSON output."""
    
    cmd = [
        "python3",
        str(rational_script),
        "--base-url", base_url,
        "--path", path,
        "--seed", str(seed),
        "--no-verbose",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    # Parse JSON from output (after last "=" line)
    lines = result.stdout.split("\n")
    json_start = -1
    for i, line in enumerate(lines):
        if line.startswith("===="):
            json_start = i + 1
    
    if json_start >= 0:
        json_lines = lines[json_start:]
        json_text = "\n".join(json_lines)
        try:
            return json.loads(json_text)
        except:
            pass
    
    raise ValueError(f"Could not parse rational agent output: {result.stdout}")
    

def compare_on_episode(
    neural_model,
    rational_script: Path,
    base_url: str,
    path: str,
    seed: int,
    device: str = "cpu",
    verbose: bool = False,
) -> ComparisonResult:
    """Run both agents on same seed and compare."""
    
    # Run neural agent
    if verbose:
        print(f"\n[SEED {seed}] Neural agent...")
    
    neural_ep = run_neural_episode(
        model=neural_model,
        base_url=base_url,
        path=path,
        seed=seed,
        device=device,
        verbose=verbose,
    )
    
    # Run rational agent
    if verbose:
        print(f"\n[SEED {seed}] Rational agent...")
    
    rational_result = run_rational_agent(rational_script, base_url, path, seed)
    
    # Parse rational result
    rational_success = rational_result.get("success", False)
    rational_steps = rational_result.get("steps", 0)
    
    # Extract actions from rational agent (from inventory changes or other fields)
    # For now, assume we can't easily get action sequence, so just compare success/steps
    rational_actions = []  # TODO: extract if available
    
    # Compare
    actions_match = False  # Can't verify without action sequence
    steps_diff = neural_ep.total_steps - rational_steps
    
    return ComparisonResult(
        seed=seed,
        map_ascii=neural_ep.map_ascii,
        neural_success=neural_ep.success,
        neural_steps=neural_ep.total_steps,
        neural_actions=neural_ep.actions,
        rational_success=rational_success,
        rational_steps=rational_steps,
        rational_actions=rational_actions,
        actions_match=actions_match,
        steps_diff=steps_diff,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rational-agent", type=str, required=True, help="Path to pure_algorithmic_agent.py")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="comparison_results.json")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()
    
    rational_script = Path(args.rational_agent)
    if not rational_script.exists():
        print(f"Error: Rational agent not found: {rational_script}")
        return 1
    
    # Load neural model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint_info = load_checkpoint(Path(args.checkpoint), device=args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Run comparisons
    results = []
    
    print(f"Running {args.num_episodes} episodes...")
    print(f"Path: {args.path}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_episodes - 1}")
    print()
    
    start_time = time.time()
    
    for i in range(args.num_episodes):
        seed = args.start_seed + i
        
        if not args.verbose:
            print(f"[{i+1:3d}/{args.num_episodes}] Seed {seed}...", end=" ", flush=True)
        
        try:
            result = compare_on_episode(
                neural_model=model,
                rational_script=rational_script,
                base_url=args.base_url,
                path=args.path,
                seed=seed,
                device=args.device,
                verbose=args.verbose,
            )
            
            results.append(result)
            
            if not args.verbose:
                n_status = "✓" if result.neural_success else "✗"
                r_status = "✓" if result.rational_success else "✗"
                diff = f"{result.steps_diff:+d}" if result.steps_diff != 0 else "SAME"
                print(f"N:{n_status} R:{r_status} Δ={diff}")
        
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    if not results:
        print("No successful comparisons!")
        return 1
    
    # Compute statistics
    neural_successes = sum(1 for r in results if r.neural_success)
    rational_successes = sum(1 for r in results if r.rational_success)
    
    neural_avg_steps = sum(r.neural_steps for r in results if r.neural_success) / neural_successes if neural_successes > 0 else 0
    rational_avg_steps = sum(r.rational_steps for r in results if r.rational_success) / rational_successes if rational_successes > 0 else 0
    
    # Print summary
    print()
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Episodes: {len(results)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(results):.2f}s per episode)")
    print()
    print(f"Neural agent:")
    print(f"  Success: {neural_successes}/{len(results)} ({100*neural_successes/len(results):.1f}%)")
    print(f"  Avg steps (successful): {neural_avg_steps:.1f}")
    print()
    print(f"Rational agent:")
    print(f"  Success: {rational_successes}/{len(results)} ({100*rational_successes/len(results):.1f}%)")
    print(f"  Avg steps (successful): {rational_avg_steps:.1f}")
    print("=" * 60)
    
    # Save results
    output_path = Path(args.output)
    output_data = {
        "checkpoint": str(args.checkpoint),
        "rational_agent": str(rational_script),
        "path": args.path,
        "num_episodes": len(results),
        "start_seed": args.start_seed,
        "elapsed_seconds": elapsed,
        "statistics": {
            "neural_success_rate": neural_successes / len(results),
            "rational_success_rate": rational_successes / len(results),
            "neural_avg_steps": neural_avg_steps,
            "rational_avg_steps": rational_avg_steps,
        },
        "episodes": [asdict(r) for r in results],
    }
    
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
