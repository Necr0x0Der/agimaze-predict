"""Training entry point for the event-triggered visual-memory Transformer."""

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
from .model import VisualTransformer, VisualTransformerConfig, target_cross_entropy
from .tokenizer import collate_visual_examples


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


def _collator(context_length: int, canvas_height: int, canvas_width: int):
    def collate(examples: Sequence[PreparedExample]) -> dict[str, Tensor]:
        batch = collate_visual_examples(examples, context_length=context_length, canvas_height=canvas_height, canvas_width=canvas_width)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return collate


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = VisualTransformerConfig(
        context_length=args.context_length, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_multiplier=args.mlp_multiplier, dropout=args.dropout,
        canvas_height=args.canvas_height, canvas_width=args.canvas_width, visual_d_model=args.visual_d_model,
        visual_spatial_layers=args.visual_spatial_layers, visual_temporal_layers=args.visual_temporal_layers,
        temporal_history=args.temporal_history, pos_readout=args.pos_readout, visual_gate_init=args.visual_gate_init,
    )
    train_examples, validation_examples = load_examples(args.train_datasets), load_examples(args.validation_datasets)
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=_collator(config.context_length, config.canvas_height, config.canvas_width))
    model = VisualTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print("Training set size:", len(train_examples))
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total, bytes_total = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if config.pos_readout == "full_text":
                logits = model(batch["input_ids"], visual_maps=batch["visual_maps"], event_positions=batch["event_positions"], event_counts=batch["event_counts"])
                labels = batch["labels"]
            else:
                logits = model(batch["input_ids"], visual_maps=batch["visual_maps"], event_positions=batch["event_positions"], event_counts=batch["event_counts"], target_input_ids=batch["target_input_ids"])
                labels = batch["target_labels"]
            loss = target_cross_entropy(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            active = int(labels.ne(-100).sum().item())
            loss_total += float(loss.item()) * active
            bytes_total += active
        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            metrics = evaluate_examples(model, validation_examples, device=device)
            print(f"epoch={epoch} train_target_byte_nll={loss_total / bytes_total:.6f} val_target_byte_nll={metrics['target_byte_nll']:.6f} val_greedy_exact_target_accuracy={metrics['greedy_exact_target_accuracy']:.4f}", flush=True)
    metrics = evaluate_examples(model, validation_examples, device=device)
    checkpoint = {
        "format": "agimaze_predict.visual_transformer.v0",
        "model_config": asdict(config), "model_state_dict": model.state_dict(), "metrics": metrics,
        "datasets": {"train_paths": [str(path.resolve()) for path in args.train_datasets], "validation_paths": [str(path.resolve()) for path in args.validation_datasets]},
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
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = resolve_training_arguments(parser)
        if min(args.epochs, args.evaluate_every, args.batch_size) <= 0:
            parser.error("epochs, evaluate-every, and batch-size must be positive")
        train(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
