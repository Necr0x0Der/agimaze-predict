# agimaze-predict

Prediction-model baselines and shared loading utilities for prepared AGI Maze
transition datasets.

This repository complements [agimaze-bench](https://github.com/Necr0x0Der/agimaze-bench),
which provides the LLM benchmarking harness, access to the
[AGI Maze](https://agimaze.org) API, and the framework documentation. Here, the
same framework is used to study models trained from scratch rather than pretrained
LLMs: a compact, MNIST-like setting for controlled next-token-prediction experiments.

## Layout

- `src/agimaze_predict/data/` — shared dataset contracts, parsers and loaders.
- `src/agimaze_predict/baselines/` — reproducible reference models.
- `datasets/` — prepared datasets, grouped by format; see
  [`datasets/README.md`](datasets/README.md).
- `experiments/` — reproducible training configurations, grouped by model and
  dataset type.
- `scripts/` — thin command-line entry points.
- `tests/` — loader and model tests with small committed fixtures only.

## Current prepared-data contract

The first supported format is newline-delimited JSON (JSONL). Every non-empty
line must be exactly:

```json
{"input":"<MAP>...</MAP>\n<ACT>right</ACT>","target":"<POS>(0, 1)</POS>"}
```

`input` must be a `<MAP>...</MAP>` block followed by one newline and an
`<ACT>...</ACT>` block. `target` must be one `<POS>...</POS>` block. Dataset
metadata, raw traces and terminal tags are deliberately outside this contract.

The baseline accepts explicit local JSONL paths. Training requires separate train
and held-out validation/test datasets, so dataset construction owns the split by
maze/source rather than splitting correlated examples inside the trainer.
Downloading and API access are deliberately deferred.

## Vanilla byte-Transformer baseline

The first model is a decoder-only Transformer trained from random initialization:
256 literal UTF-8 byte tokens plus one batch-only PAD token, learned positional
embeddings, LayerNorm, GELU MLPs and causal attention. It serializes each example
as `input + "\\n" + target` and masks next-byte cross-entropy everywhere except
the target `<POS>...</POS>` bytes.

Install the optional PyTorch dependency, then run it against a locally prepared
dataset:

```bash
python -m pip install -e '.[torch]'
python scripts/train_byte_transformer.py \
  --train-dataset /path/to/train_per_step.jsonl \
  --validation-dataset /path/to/test_per_step.jsonl \
  --output runs/byte-transformer-v0a.pt
python scripts/evaluate_byte_transformer.py \
  --dataset /path/to/test_per_step.jsonl \
  --checkpoint runs/byte-transformer-v0a.pt \
  --split validation
```

For semantic inspection of greedy failures, add `--greedy-error-log`:

```bash
python scripts/evaluate_byte_transformer.py \
  --dataset /path/to/test_per_step.jsonl \
  --checkpoint runs/byte-transformer-v0a.pt \
  --split validation \
  --greedy-error-log runs/validation-errors.txt
```

The UTF-8 report contains one self-contained record per incorrect example:
its `MAP` and `ACT`, expected `<POS>`, generated `<POS>`, and the first byte at
which the target strings differ. Training has the analogous
`--validation-greedy-error-log` flag for its final validation pass.

`--test-dataset` is accepted as an alias for `--validation-dataset`. Both
`--train-dataset` and `--validation-dataset` can be repeated to combine prepared
JSONL shards in the supplied order. The selected paths and example counts are
stored in the checkpoint as `explicit_train_validation_datasets`. For
`evaluate_byte_transformer.py --split validation`, pass the same validation/test
JSONL path that was used for a single-file validation run; this is checked against
checkpoint metadata. For a sharded validation set, invoke evaluation without
`--split validation` once per shard.

### Reusable experiment configuration

Training can instead be driven by an optional TOML configuration file:

```bash
python scripts/train_byte_transformer.py --config experiments/byte_transformer_example.toml
```

See [`experiments/byte_transformer_example.toml`](experiments/byte_transformer_example.toml)
for the complete canonical format. It keeps dataset shard lists under `[data]`,
Transformer architecture under `[model]`, and optimizer/training settings under
`[training]`. `[run]` contains the output path and runtime settings. Relative paths
in the TOML are resolved relative to that file, not the current shell directory.

All regular CLI options remain supported. Explicit CLI options override their TOML
counterparts, for example:

```bash
python scripts/train_byte_transformer.py \
  --config experiments/byte-transformer/per_step/4x4-keys.toml \
  --learning-rate 0.0001 \
  --epochs 100 \
  --output runs/4x4-lr1e-4.pt
```

The CLI-only invocation is fully supported. When a config uses `overwrite =
true`, pass `--no-overwrite` to override it for one run.
