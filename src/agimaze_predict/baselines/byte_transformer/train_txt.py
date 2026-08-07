"""Separate byte-Transformer trainer for MAP/START + ACT/TXT trace rollouts.

The existing ``train.py`` keeps its prepared MAP + ACT+ -> POS contract for
per-step and seq experiments.  This module consumes composed full TXT traces,
expands their first configurable number of ACT/TXT pairs, and applies loss only
to TXT blocks.  Earlier TXT observations stay in the stream as ground truth.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from agimaze_predict.data.txt_trace import TxtRolloutExample, TxtTraceDataset

from .evaluate_txt import evaluate_txt_rollouts
from .model import ByteTransformer, ByteTransformerConfig, target_cross_entropy
from .txt_config import resolve_txt_training_arguments
from .txt_tokenizer import collate_txt_rollouts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collator(context_length: int):
    def collate(examples: Sequence[TxtRolloutExample]) -> dict[str, Tensor]:
        batch = collate_txt_rollouts(examples, context_length=context_length)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }

    return collate


def make_dataloader(
    examples: Sequence[TxtRolloutExample], *, batch_size: int, context_length: int, shuffle: bool
) -> DataLoader[TxtRolloutExample]:
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=_collator(context_length))


def load_examples(paths: Sequence[Path], *, initial_context: str, depth: int) -> list[TxtRolloutExample]:
    examples: list[TxtRolloutExample] = []
    for path in paths:
        examples.extend(TxtTraceDataset(path, initial_context=initial_context, depth=depth))  # type: ignore[arg-type]
    return examples


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_examples = load_examples(
        args.train_datasets, initial_context=args.initial_context, depth=args.depth
    )
    validation_examples = load_examples(
        args.validation_datasets, initial_context=args.initial_context, depth=args.depth
    )
    config = ByteTransformerConfig(
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_multiplier=args.mlp_multiplier,
        dropout=args.dropout,
    )
    model = ByteTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_dataloader(
        train_examples, batch_size=args.batch_size, context_length=config.context_length, shuffle=True
    )
    print("Training set size:", len(train_loader.dataset))

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_bytes = 0
        loss_sum = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = target_cross_entropy(model(input_ids), labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            active_bytes = int(labels.ne(-100).sum().item())
            loss_sum += float(loss.item()) * active_bytes
            target_bytes += active_bytes

        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            metrics = evaluate_txt_rollouts(model, validation_examples, device=device)
            print(
                f"epoch={epoch} train_txt_byte_nll={loss_sum / target_bytes:.6f} "
                f"val_txt_byte_nll={metrics['txt_byte_nll']:.6f} "
                f"val_txt_span_exact_accuracy={metrics['txt_span_exact_accuracy']:.4f} "
                f"val_trace_all_txt_exact_accuracy={metrics['trace_all_txt_exact_accuracy']:.4f}",
                flush=True,
            )

    metrics = evaluate_txt_rollouts(model, validation_examples, device=device)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"checkpoint already exists: {output} (pass --overwrite to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "agimaze_predict.byte_transformer.txt_trace.v1",
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "datasets": {
            "train_paths": [str(Path(path).resolve()) for path in args.train_datasets],
            "validation_paths": [str(Path(path).resolve()) for path in args.validation_datasets],
        },
        "rollout": {"initial_context": args.initial_context, "depth": args.depth},
        "split": {
            "kind": "explicit_train_validation_datasets",
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
        "metrics": metrics,
    }
    torch.save(checkpoint, output)
    print(json.dumps({"checkpoint": str(output), "metrics": metrics}, sort_keys=True), flush=True)
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help="TOML TXT-trace training configuration file")
    parser.add_argument("--train-dataset", dest="train_datasets", type=Path, action="append")
    parser.add_argument("--validation-dataset", "--test-dataset", dest="validation_datasets", type=Path, action="append")
    parser.add_argument("--initial-context", choices=("MAP", "START"), help="select MAP or START as rollout prefix")
    parser.add_argument("--depth", type=int, help="maximum ACT/TXT pairs unrolled from each trace")
    parser.add_argument("--output", type=Path, help="checkpoint output path")
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument("--overwrite", action="store_true", help="replace an existing checkpoint")
    overwrite.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--mlp-multiplier", type=int)
    parser.add_argument("--dropout", type=float)
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = resolve_txt_training_arguments(parser)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.epochs <= 0 or args.evaluate_every <= 0 or args.batch_size <= 0 or args.depth <= 0:
        parser.error("epochs, evaluate-every, batch-size, and depth must be positive")
    try:
        train(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
