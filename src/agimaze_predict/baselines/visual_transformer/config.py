"""TOML configuration for visual-memory Transformer experiments."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Sequence

DEFAULT_TRAINING_ARGUMENTS: dict[str, object] = {
    "device": None, "seed": 0, "epochs": 50, "evaluate_every": 5, "batch_size": 32,
    "learning_rate": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0,
    "context_length": 512, "d_model": 128, "n_heads": 4, "n_layers": 4,
    "mlp_multiplier": 4, "dropout": 0.0,
    "canvas_height": 9, "canvas_width": 17, "visual_d_model": 128,
    "visual_spatial_layers": 2, "visual_temporal_layers": 2, "temporal_history": 8,
    "pos_readout": "full_text", "visual_gate_init": 0.05, "overwrite": False,
}

_CONFIG_SECTIONS = {
    "data": frozenset({"train_files", "validation_files", "test_files"}),
    "model": frozenset({"context_length", "d_model", "n_heads", "n_layers", "mlp_multiplier", "dropout"}),
    "visual": frozenset({"canvas_height", "canvas_width", "visual_d_model", "visual_spatial_layers", "visual_temporal_layers", "temporal_history", "pos_readout", "visual_gate_init", "map_supervision"}),
    "training": frozenset({"seed", "epochs", "evaluate_every", "batch_size", "learning_rate", "weight_decay", "grad_clip"}),
    "run": frozenset({"output", "overwrite", "device"}),
}


def _error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _paths(path: Path, section: str, value: object) -> list[Path]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise _error(path, f"[{section}] must be a non-empty array of paths")
    return [Path(item).expanduser() if Path(item).expanduser().is_absolute() else path.parent / item for item in value]


def load_training_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise _error(config_path, f"invalid TOML: {exc}") from exc
    values: dict[str, object] = {}
    for section_name, allowed in _CONFIG_SECTIONS.items():
        section = raw.pop(section_name, {})
        if not isinstance(section, dict):
            raise _error(config_path, f"[{section_name}] must be a table")
        unknown = set(section).difference(allowed)
        if unknown:
            raise _error(config_path, f"[{section_name}] has unsupported keys: {', '.join(sorted(unknown))}")
        if section_name == "data":
            if "train_files" in section:
                values["train_datasets"] = _paths(config_path, "data.train_files", section["train_files"])
            validation = "validation_files" if "validation_files" in section else "test_files" if "test_files" in section else None
            if validation is not None:
                values["validation_datasets"] = _paths(config_path, f"data.{validation}", section[validation])
        elif section_name == "run":
            for key, value in section.items():
                values[key] = (Path(value).expanduser() if Path(value).expanduser().is_absolute() else config_path.parent / value) if key == "output" else value
        elif section_name == "visual":
            map_supervision = section.pop("map_supervision", "none")
            if map_supervision != "none":
                raise _error(config_path, "[visual] map_supervision is reserved; v0 supports only 'none'")
            values.update(section)
        else:
            values.update(section)
    if raw:
        raise _error(config_path, f"unsupported top-level sections: {', '.join(sorted(raw))}")
    for key in ("canvas_height", "canvas_width", "visual_d_model", "visual_spatial_layers", "visual_temporal_layers", "temporal_history"):
        if key in values and (not isinstance(values[key], int) or isinstance(values[key], bool) or values[key] <= 0):
            raise _error(config_path, f"[visual] {key} must be a positive integer")
    if values.get("pos_readout", "full_text") not in {"full_text", "visual_only"}:
        raise _error(config_path, "[visual] pos_readout must be 'full_text' or 'visual_only'")
    return values


def resolve_training_arguments(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    explicit = vars(parser.parse_args(argv))
    config_path = explicit.pop("config", None)
    values = dict(DEFAULT_TRAINING_ARGUMENTS)
    if config_path is not None:
        values.update(load_training_config(config_path))
    values.update(explicit)
    missing = [key for key in ("train_datasets", "validation_datasets", "output") if key not in values]
    if missing:
        parser.error("missing required setting(s): " + ", ".join("--" + key.replace("_", "-") for key in missing))
    return argparse.Namespace(**values)
