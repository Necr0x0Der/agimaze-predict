# Training with Rational Agent Datasets

## Overview

This experiment extends the byte-transformer to predict **both**:
1. **Target position** (`<POS>...</POS>`) — original behavior
2. **Action answer** (`<ACT>content</ACT>`) — content and closing tag, not opening tag

## Motivation

**Problem with random-walk datasets:**
- Actions are random → unpredictable
- Model cannot learn meaningful action prediction
- Only position prediction is feasible

**Solution: Rational-agent datasets:**
- Actions are deterministic, optimal for the given map
- Model can learn **map → optimal action** mapping
- Enables true action prediction task

## Key Differences

### Standard Training (random-walk data)

```python
# Loss only on target position
<MAP>...<ACT>random_action</ACT>...\n<POS>target</POS>
                                     ^^^^^^^^^^^^^^^^
                                     Loss computed here
```

### Rational-Agent Training (this experiment)

```python
# Loss on BOTH action content AND target position
<MAP>...<ACT>optimal_action</ACT>...\n<POS>target</POS>
             ^^^^^^^^^^^^^^            ^^^^^^^^^^^^^^^^
             Loss here too!            Original loss
```

## Implementation

**Modified files:**
- `src/agimaze_predict/baselines/byte_transformer/tokenizer_with_acts.py`
  - New function: `collate_byte_examples_with_actions()`
  - Finds all `<ACT>content</ACT>` ranges in input
  - Includes content bytes and the closing `</ACT>` tag, but not `<ACT>`

- `scripts/train_byte_transformer_with_actions.py`
  - Training script that uses modified collator
  - Otherwise identical to standard training

**What stays the same:**
- Model architecture (vanilla byte-transformer)
- Dataset format (`seq` and `txt` JSONL)
- No code changes needed in data loading

## Visual Transformer: same supervision protocol

The visual Transformer reads the initial `<MAP>` through its visual canvas; the
text stream starts with the first `<ACT>`. Pass `--predict-actions` (or set
`[training] predict_actions = true` in TOML) to supervise each action's content
and `</ACT>`, plus the position answer after the known `<POS>` query. The opening
`<ACT>` and its start time stay masked. In `full_text`, the causal text output
predicts both ACT and POS. In `visual_only`, the **unchanged visual-only POS
decoder** still predicts POS from the final visual frame, while the already
existing causal text output predicts ACT from the map and preceding actions;
the forward interface can return both outputs during joint training. This
adds no parameters and leaves position-only inference/checkpoints compatible.

```bash
python3 scripts/train_visual_transformer_with_actions.py \
  --config experiments/rational-agent/seq/3x3-keys-visual-actions.toml
```

The visual trainer writes a timestamped `.log` next to the configured `.pt`,
mirroring the byte-action trainer's console output, configuration, arguments,
model summary, per-epoch train/validation ACT+POS losses, and run summary.
At each validation interval it replaces the configured checkpoint **only if**
the combined validation ACT+POS byte NLL strictly improves. The checkpoint's
`epoch` and `metrics` describe that best epoch, not necessarily the last one.
Existing checkpoints require `--overwrite` or `[run] overwrite = true`.

Validation reports teacher-forced total/ACT/POS byte NLL, **not** autonomous
action selection or goal-directed success. Validation demonstrations include
the correct earlier actions, so evaluate rollouts separately before claiming
policy competence. Use a Python 3.11+ environment with PyTorch installed.

### Live visual-only agent

```bash
/opt/anaconda3/bin/python3 scripts/visual_action_agent.py \
  --checkpoint runs/visual-rational-actions-3x3-keys-visual.pt \
  --base-url http://127.0.0.1:8000 \
  --path TRAINING/S0-keys/STAGE-01 --seed 42
```

The agent reads the **initial** map returned by `/api/start`, preserves open
border spaces, and builds the same padded visual canvas used in training.
Each step sends `<ACT>` plus previously accepted actions to the checkpoint's
causal text readout, greedily generates the next action, and posts it to
`/api/step`. It warns and continues if the generated text has an unambiguous
action prefix followed by noise. Invalid generations and exhausted context
stop the run instead of guessing an action or silently discarding history.
This checkpoint's text context is 192 bytes: long live trajectories can
exhaust it; short teacher-forced validation losses do not guarantee successful
closed-loop navigation. `--max-steps` can cap an exploratory run.

**Split caveat:** the rational `3x3-keys` shards currently selected above are
not disjoint as serialized examples: 618 distinct validation `(input, target)`
pairs also occur in training (across the 4- and 8-step validation files).
Treat validation loss as a pipeline diagnostic, not a generalization estimate;
regenerate maze-disjoint rational train/validation splits before comparing
held-out policy behavior. Identical rendered maps alone need not imply the
same source maze, but identical complete examples are definite overlap.

## Usage

### Training

```bash
python3 scripts/train_byte_transformer_with_actions.py \
  --config experiments/rational-agent/seq/3x3-keys-mixed.toml
```

## Comparison with Random-Walk

| Aspect | Random-Walk | Rational-Agent |
|--------|-------------|----------------|
| Action predictability | ❌ Random | ✅ Deterministic |
| Action prediction task | ❌ Impossible | ✅ Feasible |
| Position prediction | ✅ Possible | ✅ Better |
| Dataset size (100 ep) | ~1000 steps | ~960 steps (efficient) |
| Episode success rate | 60-80% | 100% |
| Training signal | Weak (POS only) | Strong (POS + ACT) |

## Notes

- **Conservative experiment:** No model architecture changes required
- **Minimal code changes:** Only collate function modified
- **Backward compatible:** Can still train on random-walk data with original script
- **Extensible:** Same approach works for rivers (S1) and pits (S2) datasets
