"""TOML configuration for visual-memory TXT rollout experiments."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Sequence

DEFAULT_TXT_TRAINING_ARGUMENTS: dict[str, object] = {
    "device": None, "seed": 0, "epochs": 50, "evaluate_every": 5, "batch_size": 32,
    "learning_rate": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0,
    "context_length": 512, "d_model": 128, "n_heads": 4, "n_layers": 4,
    "mlp_multiplier": 4, "dropout": 0.0, "canvas_height": 9, "canvas_width": 17,
    "visual_d_model": 128, "visual_spatial_layers": 2, "visual_temporal_layers": 2,
    "temporal_history": 8, "pos_readout": "full_text", "visual_gate_init": 0.99,
    "overwrite": False, "init_checkpoint": None,
}

_CONFIG_SECTIONS = {
    "data": frozenset({"train_files", "train_datasets", "validation_files", "validation_datasets", "initial_context", "depth"}),
    "model": frozenset({"context_length", "d_model", "n_heads", "n_layers", "mlp_multiplier", "dropout"}),
    "visual": frozenset({"canvas_height", "canvas_width", "visual_d_model", "visual_spatial_layers", "visual_temporal_layers", "temporal_history", "pos_readout", "visual_gate_init"}),
    "training": frozenset({"seed", "epochs", "evaluate_every", "batch_size", "learning_rate", "weight_decay", "grad_clip"}),
    "run": frozenset({"output", "overwrite", "device", "init_checkpoint"}),
}


def _error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _paths(path: Path, key: str, value: object) -> list[Path]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise _error(path, f"[data] {key} must be a non-empty array of paths")
    return [Path(item).expanduser() if Path(item).expanduser().is_absolute() else path.parent / item for item in value]


def load_visual_txt_training_config(path: str | Path) -> dict[str, object]:
    """Load TXT visual training settings; relative paths use the TOML directory."""

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw: object = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise _error(config_path, f"invalid TOML: {exc}") from exc
    if not isinstance(raw, dict):
        raise _error(config_path, "top level must be a table")
    values: dict[str, object] = {}
    for section_name, allowed in _CONFIG_SECTIONS.items():
        section = raw.pop(section_name, {})
        if not isinstance(section, dict):
            raise _error(config_path, f"[{section_name}] must be a table")
        unknown = set(section).difference(allowed)
        if unknown:
            raise _error(config_path, f"[{section_name}] has unsupported keys: {', '.join(sorted(unknown))}")
        if section_name == "data":
            for destination, aliases in (("train_datasets", ("train_files", "train_datasets")), ("validation_datasets", ("validation_files", "validation_datasets"))):
                found = [key for key in aliases if key in section]
                if len(found) > 1:
                    raise _error(config_path, f"[data] cannot set both {found[0]} and {found[1]}")
                if found:
                    values[destination] = _paths(config_path, found[0], section[found[0]])
            if "initial_context" in section:
                if section["initial_context"] not in {"MAP", "START"}:
                    raise _error(config_path, "[data] initial_context must be exactly 'MAP' or 'START'")
                values["initial_context"] = section["initial_context"]
            if "depth" in section:
                if not isinstance(section["depth"], int) or isinstance(section["depth"], bool) or section["depth"] < 1:
                    raise _error(config_path, "[data] depth must be a positive integer")
                values["depth"] = section["depth"]
        elif section_name == "run":
            for key, value in section.items():
                values[key] = (
                    Path(value).expanduser() if Path(value).expanduser().is_absolute() else config_path.parent / value
                ) if key in {"output", "init_checkpoint"} else value
        else:
            values.update(section)
    if raw:
        raise _error(config_path, f"unsupported top-level sections: {', '.join(sorted(raw))}")
    if values.get("pos_readout", "full_text") not in {"full_text", "visual_only"}:
        raise _error(config_path, "[visual] pos_readout must be 'full_text' or 'visual_only'")
    return values


def resolve_visual_txt_training_arguments(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    explicit = vars(parser.parse_args(argv))
    config_path = explicit.pop("config", None)
    values = dict(DEFAULT_TXT_TRAINING_ARGUMENTS)
    if config_path is not None:
        values.update(load_visual_txt_training_config(config_path))
    values.update(explicit)
    missing = [key for key in ("train_datasets", "validation_datasets", "output", "initial_context", "depth") if key not in values]
    if missing:
        flags = {"train_datasets": "--train-dataset", "validation_datasets": "--validation-dataset", "output": "--output", "initial_context": "--initial-context", "depth": "--depth"}
        parser.error(f"missing required setting(s): {', '.join(flags[key] for key in missing)}")
    return argparse.Namespace(**values)
