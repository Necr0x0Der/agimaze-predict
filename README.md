# agimaze-predict

Prediction-model baselines and shared loading utilities for prepared AGI Maze
transition datasets.

The repository intentionally has no dependency on the private `labyrinth` engine
repository. A prepared dataset is supplied as an explicit local path.

## Layout

- `src/agimaze_predict/data/` — shared dataset contracts, parsers and loaders.
- `src/agimaze_predict/baselines/` — reproducible reference models.
- `src/agimaze_predict/experiments/` — reserved for future experimental
  architectures; it is intentionally absent until there is code to place there.
- `scripts/` — thin command-line entry points.
- `tests/` — loader and model tests with small committed fixtures only.

## Current prepared-data contract

The first supported format is newline-delimited JSON (JSONL), emitted by
`labyrinth/scripts/compose_dataset.py`. Every non-empty line must be exactly:

```json
{"input":"<MAP>...</MAP>\n<ACT>right</ACT>","target":"<POS>(0, 1)</POS>"}
```

`input` must be a `<MAP>...</MAP>` block followed by one newline and an
`<ACT>...</ACT>` block. `target` must be one `<POS>...</POS>` block. Dataset
metadata, raw traces and terminal tags are deliberately outside this contract.

The first baseline accepts this JSONL file through an explicit `--dataset` path.
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
  --dataset /path/to/per_step_map_act_to_pos.jsonl \
  --output runs/byte-transformer-v0a.pt
python scripts/evaluate_byte_transformer.py \
  --dataset /path/to/per_step_map_act_to_pos.jsonl \
  --checkpoint runs/byte-transformer-v0a.pt
```

The current train script uses a fixed-seed **random example-level** validation
split only to exercise the full pipeline. It is explicitly stored in the
checkpoint as `temporary_random_example_split`; it must be replaced with a split
by source maze before reporting generalization results.
