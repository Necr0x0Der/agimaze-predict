#!/usr/bin/env python3
"""Thin CLI wrapper for the vanilla byte-Transformer baseline.

Example:
    python scripts/train_byte_transformer.py \
      --train-dataset /path/to/train_per_step.jsonl \
      --validation-dataset /path/to/test_per_step.jsonl \
      --output runs/byte-transformer-v0a.pt
"""

from pathlib import Path
import sys

# Allow direct execution from a checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.byte_transformer.train import main


if __name__ == "__main__":
    raise SystemExit(main())
