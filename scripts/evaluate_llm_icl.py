#!/usr/bin/env python3
"""Evaluate an untrained OpenAI-compatible LLM on MAP + ACT -> POS examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct execution from a checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.llama_memory.icl_api import default_base_url, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="held-out MAP + ACT+ -> POS JSONL")
    parser.add_argument("--model", required=True, help="OpenAI/OpenRouter model identifier")
    parser.add_argument("--base-url", default=default_base_url(), help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--few-shot-dataset",
        type=Path,
        help="separate training JSONL from which demonstrations are deterministically sampled",
    )
    parser.add_argument("--few-shot-count", type=int, default=0, help="number of demonstrations; 0 is zero-shot")
    parser.add_argument("--seed", type=int, default=111, help="few-shot sampling seed")
    parser.add_argument("--max-examples", type=int, help="cap evaluation examples for a cheap smoke test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, help="optional JSONL record of every request/result")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.few_shot_count and args.few_shot_dataset is None:
            raise ValueError("--few-shot-count requires --few-shot-dataset")
        result = run(
            dataset=args.dataset,
            model=args.model,
            base_url=args.base_url,
            few_shot_dataset=args.few_shot_dataset,
            few_shot_count=args.few_shot_count,
            seed=args.seed,
            max_examples=args.max_examples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            output=args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        build_parser().error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
