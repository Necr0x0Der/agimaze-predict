#!/usr/bin/env python3
"""Train the separate byte-Transformer baseline on composed ACT/TXT traces.

Example:
    python scripts/train_txt_byte_transformer.py \
      --config experiments/byte-transformer/txt/example.toml
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.byte_transformer.train_txt import main


if __name__ == "__main__":
    raise SystemExit(main())
