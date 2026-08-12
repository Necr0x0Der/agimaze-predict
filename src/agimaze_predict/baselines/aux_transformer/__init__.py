"""Vanilla byte-level causal-Transformer baseline.

Imports are intentionally light: data/tokenizer utilities remain usable without
PyTorch, while model/train/evaluation modules require the optional ``torch``
dependency.
"""

__all__ = ["model", "tokenizer", "train", "evaluate"]
