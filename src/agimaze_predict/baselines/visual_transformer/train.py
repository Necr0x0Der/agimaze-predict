"""Training entry point for the event-triggered visual-memory Transformer."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from agimaze_predict.data.prepared import PreparedExample, PreparedMapActionsToPosDataset

from .config import resolve_training_arguments
from .evaluate import evaluate_examples
from .model import VisualTransformer, VisualTransformerConfig, target_cross_entropy
from .tokenizer import collate_visual_examples


class Logger:
    """Write each training message to the console and a dated log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.path.touch(exist_ok=False)

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(paths: Sequence[Path]) -> list[PreparedExample]:
    examples: list[PreparedExample] = []
    for path in paths:
        examples.extend(PreparedMapActionsToPosDataset(path))
    return examples


def _collator(context_length: int, canvas_height: int, canvas_width: int, predict_actions: bool = False):
    def collate(examples: Sequence[PreparedExample]) -> dict[str, Tensor]:
        batch = collate_visual_examples(examples, context_length=context_length, canvas_height=canvas_height, canvas_width=canvas_width, predict_actions=predict_actions)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return collate


@torch.no_grad()
def evaluate_action_losses(model: VisualTransformer, examples: Sequence[PreparedExample], *, device: torch.device, batch_size: int = 32) -> dict[str, float]:
    """Teacher-forced byte NLL; do not confuse this with rollout/action accuracy."""
    model.eval()
    sums = {"total": 0.0, "act": 0.0, "pos": 0.0}
    counts = {key: 0 for key in sums}
    collate = _collator(model.config.context_length, model.config.canvas_height, model.config.canvas_width, True)
    for start in range(0, len(examples), batch_size):
        batch = {key: value.to(device) for key, value in collate(examples[start:start + batch_size]).items()}
        kwargs = {key: batch[key] for key in ("visual_maps", "event_positions", "event_counts")}
        if model.config.pos_readout == "visual_only":
            pos_logits, act_logits = model(batch["input_ids"], **kwargs,
                                           target_input_ids=batch["target_input_ids"], return_action_logits=True)
            losses = (("act", act_logits, batch["labels"].masked_fill(~batch["act_mask"].bool(), -100)),
                      ("pos", pos_logits, batch["target_labels"]))
        else:
            logits = model(batch["input_ids"], **kwargs)
            losses = (("act", logits, batch["labels"].masked_fill(~batch["act_mask"].bool(), -100)),
                      ("pos", logits, batch["labels"].masked_fill(~batch["pos_mask"].bool(), -100)))
        for name, logits, labels in losses:
            active = int(labels.ne(-100).sum().item())
            if active:
                sums[name] += float(target_cross_entropy(logits, labels).item()) * active
                counts[name] += active
    sums["total"] = sums["act"] + sums["pos"]
    counts["total"] = counts["act"] + counts["pos"]
    return {f"{name}_byte_nll": sums[name] / counts[name] if counts[name] else 0.0 for name in sums} | {
        f"{name}_bytes": counts[name] for name in sums
    }


def train(args: argparse.Namespace, logger: Logger | None = None) -> dict[str, object]:
    seed_everything(args.seed)
    predict_actions = getattr(args, "predict_actions", False)
    # Also support callers that invoke train() directly rather than main().
    if logger is None:
        logger = make_logger(Path(args.output))
    started = time.monotonic()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"checkpoint already exists: {output} (pass --overwrite to replace it)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = VisualTransformerConfig(
        context_length=args.context_length, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_multiplier=args.mlp_multiplier, dropout=args.dropout,
        canvas_height=args.canvas_height, canvas_width=args.canvas_width, visual_d_model=args.visual_d_model,
        visual_spatial_layers=args.visual_spatial_layers, visual_temporal_layers=args.visual_temporal_layers,
        temporal_history=args.temporal_history, pos_readout=args.pos_readout, visual_gate_init=args.visual_gate_init,
    )
    train_examples, validation_examples = load_examples(args.train_datasets), load_examples(args.validation_datasets)
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=_collator(config.context_length, config.canvas_height, config.canvas_width, predict_actions))
    model = VisualTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.log(f"Train: {len(train_examples)} examples from {len(args.train_datasets)} file(s)")
    for path in args.train_datasets:
        logger.log(f"  - {path}")
    logger.log(f"Valid: {len(validation_examples)} examples from {len(args.validation_datasets)} file(s)")
    for path in args.validation_datasets:
        logger.log(f"  - {path}")
    logger.log("\n" + "=" * 80 + "\nMODEL ARCHITECTURE\n" + "=" * 80)
    logger.log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    for key, value in asdict(config).items():
        logger.log(f"  {key}: {value}")
    logger.log("\n" + "=" * 80 + "\nTRAINING SETUP\n" + "=" * 80)
    logger.log(f"Device: {device}; seed: {args.seed}; epochs: {args.epochs}; batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.learning_rate}; weight decay: {args.weight_decay}; grad clip: {args.grad_clip}")
    logger.log(f"Evaluate every: {args.evaluate_every} epochs")
    if predict_actions:
        logger.log("Supervision: ACT content + </ACT> and POS answer; opening <ACT> remains masked; "
                   f"POS readout: {config.pos_readout}")
    logger.log("\n" + "=" * 80 + "\nTRAINING\n" + "=" * 80)
    best_val_loss = float("inf")
    best_checkpoint: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        loss_total, bytes_total = 0.0, 0
        train_act_sum = train_pos_sum = 0.0
        train_act_bytes = train_pos_bytes = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if config.pos_readout == "full_text":
                logits = model(batch["input_ids"], visual_maps=batch["visual_maps"], event_positions=batch["event_positions"], event_counts=batch["event_counts"])
                labels = batch["labels"]
            else:
                result = model(batch["input_ids"], visual_maps=batch["visual_maps"], event_positions=batch["event_positions"], event_counts=batch["event_counts"], target_input_ids=batch["target_input_ids"], return_action_logits=predict_actions)
                if predict_actions:
                    pos_logits, act_logits = result
                else:
                    logits = result
                labels = batch["target_labels"]
            if predict_actions and config.pos_readout == "visual_only":
                act_labels = batch["labels"].masked_fill(~batch["act_mask"].bool(), -100)
                act_count = int(act_labels.ne(-100).sum().item())
                pos_count = int(labels.ne(-100).sum().item())
                act_nll = target_cross_entropy(act_logits, act_labels)
                pos_nll = target_cross_entropy(pos_logits, labels)
                loss = (act_nll * act_count + pos_nll * pos_count) / (act_count + pos_count)
            else:
                loss = target_cross_entropy(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            active = (act_count + pos_count if predict_actions and config.pos_readout == "visual_only"
                      else int(labels.ne(-100).sum().item()))
            loss_total += float(loss.item()) * active
            bytes_total += active
            if predict_actions:
                with torch.no_grad():
                    if config.pos_readout == "visual_only":
                        train_act_sum += float(act_nll.item()) * act_count
                        train_pos_sum += float(pos_nll.item()) * pos_count
                        train_act_bytes += act_count
                        train_pos_bytes += pos_count
                    else:
                        per_byte = F.cross_entropy(
                            logits.detach().reshape(-1, logits.size(-1)),
                            labels.reshape(-1), ignore_index=-100, reduction="none",
                        ).reshape_as(labels)
                        act_mask = batch["act_mask"].bool()
                        pos_mask = batch["pos_mask"].bool()
                        train_act_sum += float(per_byte[act_mask].sum().item())
                        train_pos_sum += float(per_byte[pos_mask].sum().item())
                        train_act_bytes += int(act_mask.sum().item())
                        train_pos_bytes += int(pos_mask.sum().item())
        train_nll = loss_total / bytes_total
        train_breakdown = (
            f" train_act_byte_nll={train_act_sum / train_act_bytes:.6f} "
            f"train_pos_byte_nll={train_pos_sum / train_pos_bytes:.6f}"
            if predict_actions else ""
        )
        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            metrics = (evaluate_action_losses(model, validation_examples, device=device, batch_size=args.batch_size)
                       if predict_actions else evaluate_examples(model, validation_examples, device=device))
            if predict_actions:
                val_loss = metrics["total_byte_nll"]
                logger.log(f"epoch={epoch} train_supervised_byte_nll={train_nll:.6f}{train_breakdown} "
                      f"val_total_byte_nll={metrics['total_byte_nll']:.6f} "
                      f"val_act_byte_nll={metrics['act_byte_nll']:.6f} "
                      f"val_pos_byte_nll={metrics['pos_byte_nll']:.6f} "
                      f"time={time.monotonic() - epoch_start:.1f}s")
            else:
                val_loss = metrics["target_byte_nll"]
                logger.log(f"epoch={epoch} train_target_byte_nll={train_nll:.6f} "
                           f"val_target_byte_nll={val_loss:.6f} "
                           f"val_greedy_exact_target_accuracy={metrics['greedy_exact_target_accuracy']:.4f} "
                           f"time={time.monotonic() - epoch_start:.1f}s")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    "format": "agimaze_predict.visual_transformer.v0",
                    "model_config": asdict(config), "model_state_dict": model.state_dict(),
                    "metrics": metrics, "epoch": epoch, "train_byte_nll": train_nll,
                    "predict_actions": predict_actions,
                    "datasets": {"train_paths": [str(path.resolve()) for path in args.train_datasets],
                                 "validation_paths": [str(path.resolve()) for path in args.validation_datasets]},
                }
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_checkpoint, output)
                logger.log(f"  → Checkpoint saved (epoch={epoch}, val={val_loss:.6f}): {output}")
        else:
            train_metric = "train_supervised_byte_nll" if predict_actions else "train_target_byte_nll"
            logger.log(f"epoch={epoch} {train_metric}={train_nll:.6f}{train_breakdown} "
                       f"time={time.monotonic() - epoch_start:.1f}s")
    logger.log("\n" + "=" * 80 + "\nTRAINING COMPLETE\n" + "=" * 80)
    logger.log(f"Best validation loss: {best_val_loss:.6f}; epoch: {best_checkpoint['epoch']}")
    logger.log(f"Total training time: {(time.monotonic() - started) / 60:.1f} minutes")
    logger.log(f"Checkpoint saved: {output}")
    logger.log(json.dumps({"checkpoint": str(output), "metrics": best_checkpoint["metrics"]}, sort_keys=True))
    return best_checkpoint


def make_logger(output: Path) -> Logger:
    """Create a dated log next to the checkpoint without overwriting an earlier run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for suffix in range(1000):
        extra = f"_{suffix}" if suffix else ""
        try:
            return Logger(output.parent / f"{output.stem}_{timestamp}{extra}.log")
        except FileExistsError:
            continue
    raise FileExistsError("unable to choose an unused training log filename")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-dataset", dest="train_datasets", type=Path, action="append")
    parser.add_argument("--validation-dataset", "--test-dataset", dest="validation_datasets", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device"); parser.add_argument("--seed", type=int); parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluate-every", type=int); parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float); parser.add_argument("--weight-decay", type=float); parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--context-length", type=int); parser.add_argument("--d-model", type=int); parser.add_argument("--n-heads", type=int); parser.add_argument("--n-layers", type=int); parser.add_argument("--mlp-multiplier", type=int); parser.add_argument("--dropout", type=float)
    parser.add_argument("--canvas-height", type=int); parser.add_argument("--canvas-width", type=int); parser.add_argument("--visual-d-model", type=int); parser.add_argument("--visual-spatial-layers", type=int); parser.add_argument("--visual-temporal-layers", type=int); parser.add_argument("--temporal-history", type=int); parser.add_argument("--pos-readout", choices=("full_text", "visual_only")); parser.add_argument("--visual-gate-init", type=float)
    parser.add_argument("--predict-actions", action="store_true", help="supervise ACT answers with full_text or visual_only POS readout")
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = resolve_training_arguments(parser)
        if min(args.epochs, args.evaluate_every, args.batch_size) <= 0:
            parser.error("epochs, evaluate-every, and batch-size must be positive")
        logger = make_logger(Path(args.output))
        logger.log("=" * 80 + "\nVISUAL TRANSFORMER TRAINING" +
                   (" WITH ACTION LOSS" if args.predict_actions else "") + "\n" + "=" * 80)
        logger.log(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.log(f"Log file: {logger.path}")
        if hasattr(args, "config_path"):
            logger.log("=" * 80 + f"\nCONFIGURATION: {args.config_path}\n" + "=" * 80)
            logger.log(Path(args.config_path).read_text(encoding="utf-8").rstrip())
        logger.log("=" * 80 + "\nARGUMENTS\n" + "=" * 80)
        for key, value in sorted(vars(args).items()):
            logger.log(f"{key}: {value}")
        try:
            train(args, logger)
        except Exception as exc:
            logger.log(f"ERROR: {type(exc).__name__}: {exc}")
            raise
        logger.log(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
