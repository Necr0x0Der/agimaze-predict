"""CLI for evaluating a saved vanilla byte-Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from agimaze_predict.data.per_step import PerStepMapActToPosDataset

from .evaluate import evaluate_examples
from .model import ByteTransformer, ByteTransformerConfig
from .train import split_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="prepared per_step JSONL path")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "validation"), default="all")
    parser.add_argument(
        "--greedy-error-log",
        type=Path,
        default=None,
        help="write every incorrect greedy POS prediction to this UTF-8 text file",
    )
    parser.add_argument("--device", default=None, help="PyTorch device, default: cuda when available else cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format") != "agimaze_predict.byte_transformer.v1":
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format')!r}")

    model = ByteTransformer(ByteTransformerConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    examples = list(PerStepMapActToPosDataset(args.dataset))
    if args.split == "validation":
        split = checkpoint.get("split", {})
        if split.get("kind") != "temporary_random_example_split":
            raise ValueError("checkpoint does not contain the temporary random example split metadata")
        _, examples = split_examples(
            examples,
            validation_fraction=float(split["validation_fraction"]),
            seed=int(split["seed"]),
        )

    metrics = evaluate_examples(
        model,
        examples,
        device=device,
        greedy_error_log=args.greedy_error_log,
    )
    print(json.dumps({"checkpoint": str(args.checkpoint), "split": args.split, "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
