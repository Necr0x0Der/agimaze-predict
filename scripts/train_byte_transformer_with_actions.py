#!/usr/bin/env python3
"""Train byte-transformer with loss on BOTH <POS> target AND <ACT> answers.

This modification enables training on rational-agent datasets where actions
are predictable, not random. The model learns to predict:
  1. Target position (<POS>...</POS>) — original behavior
  2. Action answer (<ACT>content</ACT>) — content and closing tag, not opening tag

Usage with TOML config:
    python scripts/train_byte_transformer_with_actions.py \
      --config experiments/rational-agent/seq/3x3-keys-h3.toml

Usage with CLI args:
    python scripts/train_byte_transformer_with_actions.py \
      --train-dataset datasets/seq/3x3-keys-rational-h3-train.jsonl \
      --validation-dataset datasets/seq/3x3-keys-rational-h3-valid.jsonl \
      --output runs/rational-h3.pt \
      --context-length 896
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import random
import json
import time
from dataclasses import asdict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Sequence, Optional

from agimaze_predict.data.prepared import PreparedExample, PreparedMapActionsToPosDataset
from agimaze_predict.baselines.byte_transformer.config import resolve_training_arguments, load_training_config
from agimaze_predict.baselines.byte_transformer.model import (
    ByteTransformer,
    ByteTransformerConfig,
    target_cross_entropy,
)
from agimaze_predict.baselines.byte_transformer.tokenizer import (
    VOCAB_SIZE,
    VOCAB_SIZE_WITH_STATE,
)

# Import our modified collate function
from agimaze_predict.baselines.byte_transformer.tokenizer_with_acts import (
    collate_byte_examples_with_actions,
)


class Logger:
    """Dual logger that writes to both console and file."""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear previous log
            with open(log_path, 'w') as f:
                f.write("")
    
    def log(self, message: str, console: bool = True):
        """Write message to log file and optionally to console."""
        if console:
            print(message)
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(message + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collator_with_actions(context_length: int, *, state_tokens: int = 0):
    """Collator that includes <ACT> content in loss."""
    def collate(examples: Sequence[PreparedExample]) -> dict[str, torch.Tensor]:
        batch = collate_byte_examples_with_actions(
            examples,
            context_length=context_length,
            state_tokens=state_tokens,
        )
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "act_mask": torch.tensor(batch["act_mask"], dtype=torch.bool),
            "pos_mask": torch.tensor(batch["pos_mask"], dtype=torch.bool),
        }
    return collate


def make_dataloader(
    examples: Sequence[PreparedExample],
    *,
    batch_size: int,
    context_length: int,
    state_tokens: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collator_with_actions(context_length, state_tokens=state_tokens),
        drop_last=False,
    )


def compute_losses_by_type(logits, labels, act_mask, pos_mask):
    """Compute separate losses for ACT and POS tokens."""
    # Total loss
    total_loss = target_cross_entropy(logits, labels)
    
    # ACT loss
    act_labels = labels.clone()
    act_labels[~act_mask] = -100  # Mask non-ACT tokens
    act_loss = target_cross_entropy(logits, act_labels)
    act_bytes = act_mask.sum().item()
    
    # POS loss
    pos_labels = labels.clone()
    pos_labels[~pos_mask] = -100  # Mask non-POS tokens
    pos_loss = target_cross_entropy(logits, pos_labels)
    pos_bytes = pos_mask.sum().item()
    
    return {
        "total_loss": total_loss,
        "act_loss": act_loss,
        "pos_loss": pos_loss,
        "act_bytes": act_bytes,
        "pos_bytes": pos_bytes,
        "total_bytes": act_bytes + pos_bytes,
    }


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    act_loss = 0.0
    pos_loss = 0.0
    total_bytes = 0
    act_bytes = 0
    pos_bytes = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        act_mask = batch["act_mask"].to(device)
        pos_mask = batch["pos_mask"].to(device)

        logits = model(input_ids)
        
        # Compute losses with breakdown
        losses = compute_losses_by_type(logits, labels, act_mask, pos_mask)
        loss = losses["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Accumulate weighted losses
        total_loss += float(loss.item()) * losses["total_bytes"]
        act_loss += float(losses["act_loss"].item()) * losses["act_bytes"]
        pos_loss += float(losses["pos_loss"].item()) * losses["pos_bytes"]
        total_bytes += losses["total_bytes"]
        act_bytes += losses["act_bytes"]
        pos_bytes += losses["pos_bytes"]

    return {
        "total": total_loss / total_bytes if total_bytes > 0 else 0.0,
        "act": act_loss / act_bytes if act_bytes > 0 else 0.0,
        "pos": pos_loss / pos_bytes if pos_bytes > 0 else 0.0,
    }


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    act_loss = 0.0
    pos_loss = 0.0
    total_bytes = 0
    act_bytes = 0
    pos_bytes = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            act_mask = batch["act_mask"].to(device)
            pos_mask = batch["pos_mask"].to(device)

            logits = model(input_ids)
            
            # Compute losses with breakdown
            losses = compute_losses_by_type(logits, labels, act_mask, pos_mask)

            # Accumulate weighted losses
            total_loss += float(losses["total_loss"].item()) * losses["total_bytes"]
            act_loss += float(losses["act_loss"].item()) * losses["act_bytes"]
            pos_loss += float(losses["pos_loss"].item()) * losses["pos_bytes"]
            total_bytes += losses["total_bytes"]
            act_bytes += losses["act_bytes"]
            pos_bytes += losses["pos_bytes"]

    return {
        "total": total_loss / total_bytes if total_bytes > 0 else 0.0,
        "act": act_loss / act_bytes if act_bytes > 0 else 0.0,
        "pos": pos_loss / pos_bytes if pos_bytes > 0 else 0.0,
    }


def train(args: argparse.Namespace, logger: Logger) -> dict[str, object]:
    """Main training function."""
    
    start_time = time.time()
    
    seed_everything(args.seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    logger.log("=" * 80)
    logger.log("LOADING DATASETS")
    logger.log("=" * 80)
    
    train_datasets = [PreparedMapActionsToPosDataset(path) for path in args.train_datasets]
    val_datasets = [PreparedMapActionsToPosDataset(path) for path in args.validation_datasets]
    
    # Flatten examples
    train_examples = [ex for ds in train_datasets for ex in ds]
    val_examples = [ex for ds in val_datasets for ex in ds]
    
    logger.log(f"Train: {len(train_examples)} examples from {len(train_datasets)} file(s)")
    for path in args.train_datasets:
        logger.log(f"  - {path}")
    
    logger.log(f"Valid: {len(val_examples)} examples from {len(val_datasets)} file(s)")
    for path in args.validation_datasets:
        logger.log(f"  - {path}")

    # Create dataloaders
    train_loader = make_dataloader(
        train_examples,
        batch_size=args.batch_size,
        context_length=args.context_length,
        state_tokens=args.state_tokens,
        shuffle=True,
    )

    val_loader = make_dataloader(
        val_examples,
        batch_size=args.batch_size,
        context_length=args.context_length,
        state_tokens=args.state_tokens,
        shuffle=False,
    )

    # Create model
    vocab_size = VOCAB_SIZE_WITH_STATE if args.state_tokens > 0 else VOCAB_SIZE
    config = ByteTransformerConfig(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_multiplier=args.mlp_multiplier,
        dropout=args.dropout,
        state_tokens=args.state_tokens,
    )

    model = ByteTransformer(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger.log("")
    logger.log("=" * 80)
    logger.log("MODEL ARCHITECTURE")
    logger.log("=" * 80)
    logger.log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"  d_model: {config.d_model}")
    logger.log(f"  n_heads: {config.n_heads}")
    logger.log(f"  n_layers: {config.n_layers}")
    logger.log(f"  mlp_multiplier: {config.mlp_multiplier}")
    logger.log(f"  context_length: {config.context_length}")
    logger.log(f"  dropout: {config.dropout}")
    logger.log(f"  state_tokens: {config.state_tokens}")
    logger.log(f"  vocab_size: {config.vocab_size}")
    
    logger.log("")
    logger.log("=" * 80)
    logger.log("TRAINING SETUP")
    logger.log("=" * 80)
    logger.log(f"Device: {device}")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Epochs: {args.epochs}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.learning_rate}")
    logger.log(f"Weight decay: {args.weight_decay}")
    logger.log(f"Gradient clip: {args.grad_clip}")
    logger.log(f"Evaluate every: {args.evaluate_every} epochs")
    logger.log(f"Mode: Rational-agent (loss on <ACT> content + <POS> target)")
    logger.log("")

    best_val_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {output_path}\n"
            "Use --overwrite or choose a different --output path"
        )

    logger.log("=" * 80)
    logger.log("TRAINING")
    logger.log("=" * 80)
    logger.log("Format: Epoch | Total / ACT / POS | Time")
    logger.log("")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_losses = train_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        epoch_time = time.time() - epoch_start

        # Evaluate every N epochs
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            val_losses = evaluate_model(model, val_loader, device)
            
            msg = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"Train: {train_losses['total']:.4f} / {train_losses['act']:.4f} / {train_losses['pos']:.4f} | "
                   f"Val: {val_losses['total']:.4f} / {val_losses['act']:.4f} / {val_losses['pos']:.4f} | "
                   f"{epoch_time:.1f}s")
            logger.log(msg)

            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_datasets": [str(p) for p in args.train_datasets],
                    "validation_datasets": [str(p) for p in args.validation_datasets],
                    "training_args": vars(args),
                }
                torch.save(checkpoint, output_path)
                logger.log(f"  → Checkpoint saved (val={val_losses['total']:.4f})")
        else:
            msg = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"Train: {train_losses['total']:.4f} / {train_losses['act']:.4f} / {train_losses['pos']:.4f} | "
                   f"{epoch_time:.1f}s")
            logger.log(msg)

    total_time = time.time() - start_time
    
    logger.log("")
    logger.log("=" * 80)
    logger.log("TRAINING COMPLETE")
    logger.log("=" * 80)
    logger.log(f"Best validation loss: {best_val_loss:.4f}")
    logger.log(f"Total training time: {total_time / 60:.1f} minutes")
    logger.log(f"Checkpoint saved: {output_path}")
    
    return {
        "best_val_loss": best_val_loss,
        "final_epoch": args.epochs,
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    
    parser.add_argument("--config", type=Path, help="TOML training configuration file")
    
    # Data
    parser.add_argument("--train-dataset", dest="train_datasets", action="append", type=Path)
    parser.add_argument("--validation-dataset", dest="validation_datasets", action="append", type=Path)
    
    # Model
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--mlp-multiplier", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--state-tokens", type=int)
    
    # Training
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", dest="epochs", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    
    # Run
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    
    return parser


def main():
    parser = build_parser()
    
    # Parse to get config path first
    temp_args = parser.parse_args()
    config_path = temp_args.config if hasattr(temp_args, 'config') else None
    
    args = resolve_training_arguments(parser)
    
    # Setup logging with timestamp
    output_stem = Path(args.output).stem
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.output).parent / f"{output_stem}_{timestamp}.log"
    logger = Logger(log_path)
    
    # Log header
    logger.log("=" * 80)
    logger.log("RATIONAL AGENT TRAINING - BYTE TRANSFORMER WITH ACTION LOSS")
    logger.log("=" * 80)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    # Log TOML config if used
    if config_path:
        logger.log("=" * 80)
        logger.log(f"CONFIGURATION: {config_path}")
        logger.log("=" * 80)
        try:
            with open(config_path, 'r') as f:
                logger.log(f.read())
        except Exception as e:
            logger.log(f"Could not read config file: {e}")
        logger.log("")
    
    # Log all arguments
    logger.log("=" * 80)
    logger.log("ARGUMENTS")
    logger.log("=" * 80)
    for key, value in sorted(vars(args).items()):
        if not key.startswith('_'):
            logger.log(f"{key}: {value}")
    logger.log("")
    
    # Run training
    try:
        result = train(args, logger)
        logger.log("")
        logger.log("=" * 80)
        logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 80)
        return 0
    except Exception as e:
        logger.log("")
        logger.log("=" * 80)
        logger.log("ERROR")
        logger.log("=" * 80)
        logger.log(f"{type(e).__name__}: {e}")
        import traceback
        logger.log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
