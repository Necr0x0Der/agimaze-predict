"""TOML/CLI configuration for event-aligned Llama TXT training."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any, Sequence


_DEFAULTS: dict[str, Any] = {
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "freeze_backbone": True,
    "lora_rank": 0,
    "lora_target_modules": ["q_proj", "v_proj", "o_proj"],
    "canvas_height": 9,
    "canvas_width": 17,
    "d_model": 256,
    "n_heads": 4,
    "spatial_layers": 2,
    "memory_tokens": 16,
    "initial_context": "START",
    "depth": 8,
    "seed": 111,
    "epochs": 100,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "evaluate_every": 5,
    "greedy_evaluate_examples": 100,
    "device": None,
    "overwrite": False,
}


def _paths(value: object, *, field: str, root: Path) -> list[Path]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a non-empty list of paths")
    return [(root / item).resolve() for item in value]


def resolve_txt_training_arguments(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    supplied = parser.parse_args(argv)
    values = dict(_DEFAULTS)
    config_path = getattr(supplied, "config", None)
    if config_path is not None:
        config_path = Path(config_path).resolve()
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        allowed = {"data", "model", "memory", "training", "run"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown TOML sections: {', '.join(sorted(unknown))}")
        for section_name in ("model", "memory", "training", "run"):
            section = data.get(section_name, {})
            if not isinstance(section, dict):
                raise ValueError(f"[{section_name}] must be a TOML table")
            values.update(section)
        section = data.get("data", {})
        if not isinstance(section, dict):
            raise ValueError("[data] must be a TOML table")
        root = config_path.parent
        data_values = {key: value for key, value in section.items() if key not in {"train_files", "validation_files"}}
        unknown_data = set(data_values) - {"initial_context", "depth"}
        if unknown_data:
            raise ValueError(f"unknown [data] settings: {', '.join(sorted(unknown_data))}")
        values.update(data_values)
        values["train_datasets"] = _paths(section.get("train_files"), field="data.train_files", root=root)
        values["validation_datasets"] = _paths(section.get("validation_files"), field="data.validation_files", root=root)
        if "output" in values:
            output = Path(values["output"]).expanduser()
            values["output"] = output if output.is_absolute() else (root / output).resolve()
    for key, value in vars(supplied).items():
        if key != "config" and value is not None:
            values[key] = value
    missing = [key for key in ("train_datasets", "validation_datasets", "output") if key not in values]
    if missing:
        flags = {"train_datasets": "--train-dataset", "validation_datasets": "--validation-dataset", "output": "--output"}
        raise ValueError(f"missing required settings: {', '.join(flags[key] for key in missing)}")
    if values["initial_context"] not in {"MAP", "START"}:
        raise ValueError("initial_context must be 'MAP' or 'START'")
    values["train_datasets"] = [Path(path).resolve() for path in values["train_datasets"]]
    values["validation_datasets"] = [Path(path).resolve() for path in values["validation_datasets"]]
    values["output"] = Path(values["output"])
    return argparse.Namespace(**values)
