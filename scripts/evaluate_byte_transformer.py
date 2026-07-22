#!/usr/bin/env python3
"""Thin CLI wrapper for a saved vanilla byte-Transformer checkpoint."""

from pathlib import Path
import sys

# Allow direct execution from a checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.byte_transformer.evaluate_checkpoint import main


if __name__ == "__main__":
    raise SystemExit(main())
