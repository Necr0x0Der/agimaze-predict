#!/usr/bin/env python3
"""Train the unchanged visual Transformer on ACT content and POS answers.

Use a TOML config with [visual] pos_readout = "full_text" and
[training] predict_actions = true, or pass --predict-actions with CLI arguments.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.visual_transformer.train import main

if __name__ == "__main__":
    raise SystemExit(main())
