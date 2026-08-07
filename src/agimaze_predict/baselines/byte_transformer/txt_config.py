"""TOML configuration support for the separate TXT trace trainer."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Sequence


DEFAULT_TXT_TRAINING_ARGUMENTS: dict[str, object] = {
    "device": None,
    "seed": 0,
    "epochs": 50,
    "evaluate_every": 5,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "context_length": 1024,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "mlp_multiplier": 4,
    "dropout": 0.0,
    "overwrite": False,
}

_CONFIG_SECTIONS: dict[str, frozenset[str]] = {
    "data": frozenset({"train_files", "train_datasets", "validation_files", "validation_datasets", "initial_context", "depth"}),
    "model": frozenset({"context_length", "d_model", "n_heads", "n_layers", "mlp_multiplier", "dropout"}),
    "training": frozenset({"seed", "epochs", "evaluate_every", "batch_size", "learning_rate", "weight_decay", "grad_clip"}),
    "run": frozenset({"output", "overwrite", "device"}),
}


def _error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _path_list(path: Path, key: str, value: object) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise _error(path, f"[data] {key} must be a non-empty array of strings")
    paths: list[Path] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise _error(path, f"[data] {key} must contain non-empty strings")
        candidate = Path(item).expanduser()
        paths.append(candidate if candidate.is_absolute() else path.parent / candidate)
    return paths


def _path(path: Path, key: str, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise _error(path, f"[{key}] path must be a non-empty string")
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else path.parent / candidate


def load_txt_training_config(path: str | Path) -> dict[str, object]:
    """Load a TXT trainer configuration; relative paths use the TOML directory."""

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
                    values[destination] = _path_list(config_path, found[0], section[found[0]])
            if "initial_context" in section:
                initial_context = section["initial_context"]
                if initial_context not in {"MAP", "START"}:
                    raise _error(config_path, "[data] initial_context must be exactly 'MAP' or 'START'")
                values["initial_context"] = initial_context
            if "depth" in section:
                depth = section["depth"]
                if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
                    raise _error(config_path, "[data] depth must be a positive integer")
                values["depth"] = depth
        elif section_name == "run":
            for key, value in section.items():
                values[key] = _path(config_path, f"run.{key}", value) if key == "output" else value
        else:
            values.update(section)
    if raw:
        raise _error(config_path, f"unsupported top-level sections: {', '.join(sorted(raw))}")
    return values


def resolve_txt_training_arguments(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    explicit = vars(parser.parse_args(argv))
    config_path = explicit.pop("config", None)
    values = dict(DEFAULT_TXT_TRAINING_ARGUMENTS)
    if config_path is not None:
        values.update(load_txt_training_config(config_path))
    values.update(explicit)
    missing = [key for key in ("train_datasets", "validation_datasets", "output", "initial_context", "depth") if key not in values]
    if missing:
        flags = {"train_datasets": "--train-dataset", "validation_datasets": "--validation-dataset", "output": "--output", "initial_context": "--initial-context", "depth": "--depth"}
        parser.error(f"missing required setting(s): {', '.join(flags[key] for key in missing)}")
    return argparse.Namespace(**values)
