"""Training entry point for the parallel auxiliary Transformer.

Training and validation JSONL shards are supplied separately. The dataset
builder, not this script, owns the split so it can separate maze instances.
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

from agimaze_predict.data.prepared import PreparedExample, PreparedMapActionsToPosDataset

from .config import resolve_training_arguments
from .evaluate import evaluate_examples
from .model import AuxTransformer, AuxTransformerConfig, target_cross_entropy
from .tokenizer import collate_aux_examples


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_examples(
    examples: Sequence[PreparedExample], *, validation_fraction: float, seed: int
) -> tuple[list[PreparedExample], list[PreparedExample]]:
    """Reconstruct the legacy random split stored in old checkpoints.

    New training runs must receive separate datasets and never call this helper.
    It remains here only because ``evaluate_checkpoint`` supports the earlier
    ``temporary_random_example_split`` checkpoint format.
    """

    if len(examples) < 2:
        raise ValueError("at least two examples are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, min(len(examples) - 1, round(len(examples) * validation_fraction)))
    validation_indices = set(indices[:validation_size])
    train = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return train, validation


def _collator(context_length: int):
    def collate(examples: Sequence[PreparedExample]) -> dict[str, Tensor]:
        batch = collate_aux_examples(examples, context_length=context_length)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "aux_lengths": torch.tensor(batch["aux_lengths"], dtype=torch.long),
        }

    return collate


def make_dataloader(
    examples: Sequence[PreparedExample],
    *,
    batch_size: int,
    context_length: int,
    shuffle: bool,
) -> DataLoader[PreparedExample]:
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collator(context_length),
    )


def load_examples(paths: Sequence[Path]) -> list[PreparedExample]:
    """Load prepared shards in the supplied deterministic order."""

    examples: list[PreparedExample] = []
    for path in paths:
        examples.extend(PreparedMapActionsToPosDataset(path))
    return examples


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_examples = load_examples(args.train_datasets)
    validation_examples = load_examples(args.validation_datasets)

    config = AuxTransformerConfig(
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_multiplier=args.mlp_multiplier,
        dropout=args.dropout,
        aux_latents_per_token=args.aux_latents_per_token,
        aux_gate_mode=args.aux_gate_mode,
        aux_scale=args.aux_scale,
        aux_gate_init=args.aux_gate_init,
    )
    model = AuxTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_dataloader(
        train_examples,
        batch_size=args.batch_size,
        context_length=config.context_length,
        shuffle=True,
    )
    print("Training set size: ", len(train_loader.dataset))

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_bytes = 0
        loss_sum = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            aux_lengths = batch["aux_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux_diagnostics = model(
                input_ids,
                aux_lengths=aux_lengths,
                return_aux_diagnostics=True,
            )
            loss = target_cross_entropy(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            active_bytes = int(labels.ne(-100).sum().item())
            loss_sum += float(loss.item()) * active_bytes
            target_bytes += active_bytes

        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            validation = evaluate_examples(model, validation_examples, device=device)
            aux_text = ""
            if aux_diagnostics:
                gates = [
                    f"gate_l{i}={aux_diagnostics[f'gate/layer_{i}']:.3f}"
                    for i in range(config.n_layers)
                ]
                aux_text = " " + " ".join(gates)
            print(
                f"epoch={epoch} train_target_byte_nll={loss_sum / target_bytes:.6f} "
                f"val_target_byte_nll={validation['target_byte_nll']:.6f} "
                f"val_greedy_exact_target_accuracy={validation['greedy_exact_target_accuracy']:.4f}"
                f"{aux_text}",
                flush=True,
            )

    metrics = evaluate_examples(
        model,
        validation_examples,
        device=device,
        greedy_error_log=args.validation_greedy_error_log,
    )
    checkpoint = {
        "format": "agimaze_predict.aux_transformer.v1",
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "datasets": {
            "train_paths": [str(Path(path).resolve()) for path in args.train_datasets],
            "validation_paths": [str(Path(path).resolve()) for path in args.validation_datasets],
        },
        "split": {
            "kind": "explicit_train_validation_datasets",
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
        "metrics": metrics,
    }
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"checkpoint already exists: {output} (pass --overwrite to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(json.dumps({"checkpoint": str(output), "metrics": metrics}, sort_keys=True), flush=True)
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help="TOML training configuration file")
    parser.add_argument(
        "--train-dataset",
        dest="train_datasets",
        type=Path,
        action="append",
        help="prepared MAP + ACT+ -> POS JSONL used for parameter updates; may be repeated",
    )
    parser.add_argument(
        "--validation-dataset",
        "--test-dataset",
        dest="validation_datasets",
        type=Path,
        action="append",
        help="held-out prepared MAP + ACT+ -> POS JSONL evaluated after selected epochs; may be repeated",
    )
    parser.add_argument("--output", type=Path, help="checkpoint output path")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", help="replace an existing checkpoint")
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="do not replace an existing checkpoint (overrides config)",
    )
    parser.add_argument("--device", help="PyTorch device, default: cuda when available else cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument(
        "--validation-greedy-error-log",
        type=Path,
        help=(
            "after the final epoch, write every incorrect validation greedy POS "
            "prediction to this UTF-8 text file"
        ),
    )
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
    parser.add_argument(
        "--aux-latents-per-token",
        type=int,
        help="zero-content auxiliary latent slots per non-target source byte; default: 0",
    )
    parser.add_argument(
        "--aux-gate-mode",
        choices=("off", "fixed", "open", "learned"),
        help="main<-aux gate: off, fixed, open, or learned; default: learned",
    )
    parser.add_argument("--aux-scale", type=float, help="external scale for main<-aux residual; default: 1")
    parser.add_argument(
        "--aux-gate-init",
        type=float,
        help="initial learned gate value in (0, 1); default: 0.05",
    )
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = resolve_training_arguments(parser)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.epochs <= 0 or args.evaluate_every <= 0 or args.batch_size <= 0:
        parser.error("epochs, evaluate-every, and batch-size must be positive")
    if args.aux_latents_per_token <= 0:
        parser.error("aux-latents-per-token must be positive")
    if args.aux_scale < 0:
        parser.error("aux-scale must be non-negative")
    if not 0.0 < args.aux_gate_init < 1.0:
        parser.error("aux-gate-init must be strictly between zero and one")
    try:
        train(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
