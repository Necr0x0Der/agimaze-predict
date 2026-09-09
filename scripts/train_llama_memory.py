"""Train the Llama-3.2 spatial-memory proof of concept."""

from pathlib import Path
import sys

# Allow direct execution from a checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.baselines.llama_memory.train import main


if __name__ == "__main__":
    raise SystemExit(main())
