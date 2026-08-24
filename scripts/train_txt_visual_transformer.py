#!/usr/bin/env python3
"""Train the visual-memory Transformer on composed ACT/TXT traces."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.visual_transformer.train_txt import main


if __name__ == "__main__":
    raise SystemExit(main())
