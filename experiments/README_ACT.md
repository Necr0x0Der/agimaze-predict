# Training with Rational Agent Datasets

## Overview

This experiment extends the byte-transformer to predict **both**:
1. **Target position** (`<POS>...</POS>`) — original behavior
2. **Action content** (`<ACT>content</ACT>`) — NEW: enables learning from rational demonstrations

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
  - Includes content bytes (not tags) in loss mask

- `scripts/train_byte_transformer_with_actions.py`
  - Training script that uses modified collator
  - Otherwise identical to standard training

**What stays the same:**
- Model architecture (vanilla byte-transformer)
- Dataset format (`seq` and `txt` JSONL)
- No code changes needed in data loading

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
