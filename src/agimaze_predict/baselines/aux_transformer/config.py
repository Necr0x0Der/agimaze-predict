"""TOML configuration support for auxiliary Transformer training."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Sequence


DEFAULT_TRAINING_ARGUMENTS: dict[str, object] = {
    "device": None,
    "seed": 0,
    "epochs": 50,
    "evaluate_every": 5,
    "validation_greedy_error_log": None,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "context_length": 512,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "mlp_multiplier": 4,
    "dropout": 0.0,
    # One or more zero-content latent slots are created per source byte,
    # never per target byte.
    "aux_latents_per_token": 1,
    "aux_gate_mode": "learned",
    "aux_scale": 1.0,
    "aux_gate_init": 0.05,
    "aux_denoise_weight": 0.0,
    "aux_mask_rate": 0.0,
    "aux_mask_span_length": 4,
    "overwrite": False,
}

_CONFIG_SECTIONS: dict[str, frozenset[str]] = {
    "data": frozenset(
        {"train_datasets", "train_files", "validation_datasets", "validation_files", "test_datasets", "test_files"}
    ),
    "model": frozenset(
        {
            "context_length",
            "d_model",
            "n_heads",
            "n_layers",
            "mlp_multiplier",
            "dropout",
            "aux_latents_per_token",
            "aux_gate_mode",
            "aux_scale",
            "aux_gate_init",
            "aux_denoise_weight",
            "aux_mask_rate",
            "aux_mask_span_length",
        }
    ),
    "training": frozenset(
        {
            "seed",
            "epochs",
            "evaluate_every",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "grad_clip",
        }
    ),
    "run": frozenset({"output", "overwrite", "device", "validation_greedy_error_log"}),
}


def _config_error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _as_path_list(config_path: Path, section: str, value: object) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise _config_error(config_path, f"[{section}] dataset paths must be a non-empty array of strings")
    paths: list[Path] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise _config_error(config_path, f"[{section}] dataset paths must be non-empty strings")
        candidate = Path(item).expanduser()
        paths.append(candidate if candidate.is_absolute() else config_path.parent / candidate)
    return paths


def _as_path(config_path: Path, section: str, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise _config_error(config_path, f"[{section}] path must be a non-empty string")
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else config_path.parent / candidate


def load_training_config(path: str | Path) -> dict[str, object]:
    """Load and validate a auxiliary Transformer training TOML file.

    Relative dataset, output, and error-log paths are interpreted relative to the
    TOML file, rather than the shell's current working directory.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw: object = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise _config_error(config_path, f"invalid TOML: {exc}") from exc

    if not isinstance(raw, dict):  # Defensive: tomllib always returns a dict.
        raise _config_error(config_path, "top level must be a table")

    values: dict[str, object] = {}
    for section_name, allowed_keys in _CONFIG_SECTIONS.items():
        section = raw.pop(section_name, {})
        if not isinstance(section, dict):
            raise _config_error(config_path, f"[{section_name}] must be a table")
        unknown = set(section).difference(allowed_keys)
        if unknown:
            raise _config_error(
                config_path,
                f"[{section_name}] has unsupported keys: {', '.join(sorted(unknown))}",
            )

        if section_name == "data":
            has_train = "train_datasets" in section or "train_files" in section
            if "train_datasets" in section and "train_files" in section:
                raise _config_error(config_path, "[data] cannot set both train_datasets and train_files")
            has_validation = "validation_datasets" in section or "validation_files" in section
            has_test = "test_datasets" in section or "test_files" in section
            if "validation_datasets" in section and "validation_files" in section:
                raise _config_error(
                    config_path,
                    "[data] cannot set both validation_datasets and validation_files",
                )
            if "test_datasets" in section and "test_files" in section:
                raise _config_error(config_path, "[data] cannot set both test_datasets and test_files")
            if has_validation and has_test:
                raise _config_error(
                    config_path,
                    "[data] cannot set both validation_* and test_* dataset lists",
                )
            if has_train:
                train_key = "train_datasets" if "train_datasets" in section else "train_files"
                values["train_datasets"] = _as_path_list(
                    config_path, f"data.{train_key}", section[train_key]
                )
            if has_validation or has_test:
                validation_key = next(
                    key
                    for key in ("validation_datasets", "validation_files", "test_datasets", "test_files")
                    if key in section
                )
                values["validation_datasets"] = _as_path_list(
                    config_path,
                    f"data.{validation_key}",
                    section[validation_key],
                )
        elif section_name == "run":
            for key, value in section.items():
                values[key] = (
                    _as_path(config_path, f"run.{key}", value)
                    if key in {"output", "validation_greedy_error_log"} and value is not None
                    else value
                )
        else:
            values.update(section)
            if section_name == "model":
                if "aux_latents_per_token" in section:
                    value = section["aux_latents_per_token"]
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        raise _config_error(
                            config_path,
                            "[model] aux_latents_per_token must be a positive integer",
                        )
                if "aux_gate_mode" in section and section["aux_gate_mode"] not in {
                    "off",
                    "fixed",
                    "open",
                    "learned",
                }:
                    raise _config_error(
                        config_path,
                        "[model] aux_gate_mode must be one of: off, fixed, open, learned",
                    )
                for key in ("aux_scale", "aux_gate_init", "aux_denoise_weight", "aux_mask_rate"):
                    if key in section and (
                        not isinstance(section[key], (int, float)) or isinstance(section[key], bool)
                    ):
                        raise _config_error(config_path, f"[model] {key} must be numeric")
                if "aux_denoise_weight" in section and section["aux_denoise_weight"] < 0:
                    raise _config_error(config_path, "[model] aux_denoise_weight must be non-negative")
                if "aux_mask_rate" in section and not 0.0 <= section["aux_mask_rate"] < 1.0:
                    raise _config_error(config_path, "[model] aux_mask_rate must be in [0, 1)")
                if "aux_mask_span_length" in section and (
                    not isinstance(section["aux_mask_span_length"], int)
                    or isinstance(section["aux_mask_span_length"], bool)
                    or section["aux_mask_span_length"] <= 0
                ):
                    raise _config_error(config_path, "[model] aux_mask_span_length must be a positive integer")

    if raw:
        raise _config_error(config_path, f"unsupported top-level sections: {', '.join(sorted(raw))}")
    return values


def resolve_training_arguments(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    """Merge built-in defaults, TOML config values, and explicit CLI arguments.

    Later sources take precedence, so explicit command-line options always
    override values supplied by ``--config``.
    """

    explicit = vars(parser.parse_args(argv))
    config_path = explicit.pop("config", None)
    values = dict(DEFAULT_TRAINING_ARGUMENTS)
    if config_path is not None:
        values.update(load_training_config(config_path))
    values.update(explicit)

    missing = [key for key in ("train_datasets", "validation_datasets", "output") if key not in values]
    if missing:
        flag = {
            "train_datasets": "--train-dataset",
            "validation_datasets": "--validation-dataset",
            "output": "--output",
        }
        parser.error(f"missing required setting(s): {', '.join(flag[key] for key in missing)}")
    return argparse.Namespace(**values)
