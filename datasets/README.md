# Datasets

This directory contains prepared AGI Maze datasets. The top-level subdirectory
names identify the dataset format/task; individual files identify the maze family,
collection policy, and split. Keeping related train and validation shards together
makes it easy to combine selected shards in an experiment configuration.

```text
datasets/
  per_step/   # MAP + ACT -> POS transition-prediction examples
  seq/        # reserved for sequential trajectory examples
  txt/        # reserved for agent-visible text-observation examples
```

## Naming

Dataset files use hyphen-separated names:

```text
<board-size>-<mechanics>-<collection-policy>-<split>.jsonl
```

For example, `3x3-keys-rnd-train.jsonl` contains random-walk (`rnd`) data from
3×3 mazes with keys, for the training split. `valid` denotes a held-out
validation split used during model selection; it is not a final test set.

## `per_step`

`per_step` is a supervised next-position task intended as a controlled check of
whether a model can learn the geometry of maze movement. Each record describes
one game state, one requested action, and the resulting agent position. It is
not a trajectory-planning or maze-solving target by itself.

Files are UTF-8 JSONL. Every non-empty line has exactly this schema:

```json
{"input":"<MAP>...</MAP>\n<ACT>right</ACT>","target":"<POS>(0, 1)</POS>"}
```

- `input` contains the rendered map for the current state and one action.
- `target` is the position after that action as a zero-based `(row, column)`
  coordinate.
- The map includes the current agent position and maze state; map and action
  are deliberately provided together so the transition depends only on that
  observable state.

The initial committed shards were collected from random-walk traces:

| Dataset family | Train mazes | Validation mazes | Train examples | Validation examples |
| --- | ---: | ---: | ---: | ---: |
| `3x3-keys-rnd` | 100 | 25 | 2,211 | 567 |
| `4x4-keys-rnd` | 100 | 25 | 2,364 | 617 |
| `3x3-rivers-rnd` | 100 | 25 | 1,875 | 480 |
| `3x4-rivers-rnd` | 200 (two shards) | 25 | 4,058 | 542 |
| `4x5-keys-rnd` | 300 (three shards) | 25 | 6,758 | 521 |

The train and validation maze sets are separate. Example counts differ from maze
counts because each random walk supplies multiple per-state transitions.

## Using shards in an experiment

The byte-Transformer trainer accepts one or more JSONL shards per split. Paths
in TOML experiment files are relative to the TOML file itself:

```toml
[data]
train_files = ["../../../datasets/per_step/3x3-keys-rnd-train.jsonl"]
validation_files = ["../../../datasets/per_step/3x3-keys-rnd-valid.jsonl"]
```

List several files to combine dataset families in a single run. See
`../experiments/byte-transformer/per_step/` for the current canonical
configurations.
