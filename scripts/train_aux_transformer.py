#!/usr/bin/env python3
"""Thin CLI wrapper for the parallel auxiliary Transformer.

Example:
    python scripts/train_aux_transformer.py \
      --train-dataset /path/to/train.jsonl \
      --validation-dataset /path/to/valid.jsonl \
      --output runs/aux-transformer.pt
"""

from pathlib import Path
import sys

# Allow direct execution from a checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.aux_transformer.train import main


if __name__ == "__main__":
    raise SystemExit(main())
