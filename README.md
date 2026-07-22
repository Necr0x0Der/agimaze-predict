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

The first baseline will accept this JSONL file through an explicit `--dataset`
path. Downloading, API access and automatic splitting are deliberately deferred.
