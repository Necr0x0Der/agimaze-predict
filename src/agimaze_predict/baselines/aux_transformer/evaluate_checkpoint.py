"""CLI for evaluating a saved auxiliary Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from agimaze_predict.data.prepared import PreparedMapActionsToPosDataset

from .evaluate import evaluate_examples
from .model import AuxTransformer, AuxTransformerConfig
from .train import split_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="prepared MAP + ACT+ -> POS JSONL path")
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
    if checkpoint.get("format") != "agimaze_predict.aux_transformer.v1":
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format')!r}")

    model = AuxTransformer(AuxTransformerConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    examples = list(PreparedMapActionsToPosDataset(args.dataset))
    if args.split == "validation":
        split = checkpoint.get("split", {})
        if split.get("kind") == "temporary_random_example_split":
            _, examples = split_examples(
                examples,
                validation_fraction=float(split["validation_fraction"]),
                seed=int(split["seed"]),
            )
        elif split.get("kind") == "explicit_train_validation_datasets":
            datasets = checkpoint["datasets"]
            validation_paths = datasets.get("validation_paths")
            if validation_paths is None:  # v1 checkpoints written before sharded datasets.
                validation_paths = [datasets["validation_path"]]
            expected_paths = {Path(path).resolve() for path in validation_paths}
            actual_path = args.dataset.resolve()
            if actual_path not in expected_paths:
                raise ValueError(
                    "--split validation for this checkpoint requires --dataset to be its "
                    f"saved validation dataset: {', '.join(map(str, sorted(expected_paths)))}"
                )
            if len(expected_paths) > 1:
                raise ValueError(
                    "--split validation cannot evaluate one shard of a checkpoint trained with "
                    "multiple validation datasets; run without --split validation to evaluate a shard"
                )
        else:
            raise ValueError("checkpoint does not contain recognised validation split metadata")

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
