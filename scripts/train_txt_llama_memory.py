#!/usr/bin/env python3
"""Train Llama with an ACT-event-aligned 2-D memory on TXT traces."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.llama_memory.train_txt import main


if __name__ == "__main__":
    raise SystemExit(main())
