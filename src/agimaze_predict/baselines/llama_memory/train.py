"""Train frozen/LoRA Llama with a final-prefix 2-D spatial workspace."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from agimaze_predict.data.prepared import PreparedExample, PreparedMapActionsToPosDataset

from .config import resolve_training_arguments
from .data import collate_examples
from .evaluate import evaluate_examples
from .model import LlamaWithSpatialMemory
from .spatial_memory import SpatialMemoryConfig


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the Llama extra first: pip install -e '.[llama]'") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _maybe_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.lora_rank <= 0:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA needs the Llama extra: pip install -e '.[llama]'") from exc
    return get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0, target_modules=args.lora_target_modules,
    ))


def _lora_checkpoint(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, object] | None:
    """Return adapter-only weights, never a duplicate copy of the Llama backbone."""

    if args.lora_rank <= 0:
        return None
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("LoRA needs the Llama extra: pip install -e '.[llama]'") from exc
    return {
        "rank": args.lora_rank,
        "target_modules": args.lora_target_modules,
        "state_dict": get_peft_model_state_dict(model),
    }


def _load_examples(paths: Sequence[Path]) -> list[PreparedExample]:
    examples: list[PreparedExample] = []
    for path in paths:
        examples.extend(PreparedMapActionsToPosDataset(path))
    return examples


def _collator(tokenizer, config: SpatialMemoryConfig):
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Llama tokenizer has neither pad_token_id nor eos_token_id")
    def collate(examples: Sequence[PreparedExample]) -> dict[str, Tensor]:
        items = collate_examples(
            examples, tokenizer=tokenizer, canvas_height=config.canvas_height,
            canvas_width=config.canvas_width, pad_token_id=pad,
        )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in items.items() if key != "target_token_ids"}
    return collate


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("Llama-3.2-3B training requires a CUDA device; no CUDA device is available")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if args.freeze_backbone:
        backbone.requires_grad_(False)
    backbone = _maybe_lora(backbone, args)
    config = SpatialMemoryConfig(
        canvas_height=args.canvas_height, canvas_width=args.canvas_width, d_model=args.d_model,
        n_heads=args.n_heads, spatial_layers=args.spatial_layers, memory_tokens=args.memory_tokens,
    )
    model = LlamaWithSpatialMemory(backbone, config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters: enable workspace training or LoRA")
    train_examples = _load_examples(args.train_datasets)
    validation_examples = _load_examples(args.validation_datasets)
    collate = _collator(tokenizer, config)
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    print(
        f"training_examples={len(train_examples)} validation_examples={len(validation_examples)} "
        f"trainable_parameters={sum(p.numel() for p in trainable)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, tokens = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            active = int(batch["labels"].ne(-100).sum().item())
            total += float(output.loss.item()) * active
            tokens += active
        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            validation = evaluate_examples(
                model,
                validation_examples,
                tokenizer=tokenizer,
                config=config,
                collate=collate,
                device=device,
                batch_size=args.batch_size,
            )
            print(
                f"epoch={epoch} train_target_nll={total / max(tokens, 1):.6f} "
                f"val_target_token_nll={validation['target_token_nll']:.6f} "
                f"val_greedy_exact_target_accuracy={validation['greedy_exact_target_accuracy']:.4f}",
                flush=True,
            )
    output = Path(args.output)
    if output.is_dir():
        raise IsADirectoryError(f"checkpoint output must be a file, not a directory: {output}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"checkpoint already exists: {output} (use overwrite=true or --overwrite)")
    # Keep this checkpoint compatible with the other baselines: --output names
    # one file, not a directory containing several independent artifacts.
    # The base tokenizer is unchanged, so it is recovered from base_model.
    checkpoint = {
        "format": "agimaze_predict.llama_memory.v1",
        "base_model": args.base_model,
        "spatial_config": config.to_dict(),
        "workspace": model.workspace.state_dict(),
        "memory_projection": model.memory_projection.state_dict(),
        "lora": _lora_checkpoint(model.llama, args),
        "datasets": {
            "train_paths": [str(Path(path).resolve()) for path in args.train_datasets],
            "validation_paths": [str(Path(path).resolve()) for path in args.validation_datasets],
        },
        "metrics": validation,
        "arguments": vars(args),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-dataset", dest="train_datasets", type=Path, action="append")
    parser.add_argument("--validation-dataset", dest="validation_datasets", type=Path, action="append")
    parser.add_argument("--output", type=Path); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-model"); parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--lora-rank", type=int); parser.add_argument("--lora-target-modules", nargs="+")
    parser.add_argument("--canvas-height", type=int); parser.add_argument("--canvas-width", type=int)
    parser.add_argument("--d-model", type=int); parser.add_argument("--n-heads", type=int); parser.add_argument("--spatial-layers", type=int); parser.add_argument("--memory-tokens", type=int)
    parser.add_argument("--seed", type=int); parser.add_argument("--epochs", type=int); parser.add_argument("--evaluate-every", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--learning-rate", type=float); parser.add_argument("--weight-decay", type=float); parser.add_argument("--grad-clip", type=float); parser.add_argument("--device")
    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = resolve_training_arguments(parser)
        if min(args.epochs, args.evaluate_every, args.batch_size, args.memory_tokens) <= 0:
            parser.error("epochs, evaluate-every, batch-size and memory-tokens must be positive")
        print(json.dumps({"checkpoint": str(train(args))}))
    except (FileNotFoundError, FileExistsError, IsADirectoryError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
