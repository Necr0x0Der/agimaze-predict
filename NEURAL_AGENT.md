# Neural Agent for AGI Maze

Neural network agent that uses trained byte-transformer checkpoints to predict actions in AGI Maze environments. The server must run in the debug mode for the agent to get the groundtruth map.

## Usage

### Single Episode

```bash
python3 scripts/neural_agent.py \
  --checkpoint runs/rational-3x3-keys-mixed.pt \
  --base-url http://127.0.0.1:8000 \
  --path "TRAINING/S0-keys/STAGE-01" \
  --seed 42
```

### Multiple Episodes

```bash
python3 scripts/neural_agent.py \
  --checkpoint runs/rational-3x3-keys-mixed.pt \
  --path "TRAINING/S0-keys/STAGE-02" \
  --num-episodes 10
```

### Comparison with Rational Agent

```bash
python3 scripts/compare_agents.py \
  --checkpoint runs/rational-3x3-keys-mixed.pt \
  --rational-agent ../agimaze-bench/agents/pure_algorithmic_agent.py \
  --path "TRAINING/S0-keys/STAGE-01" \
  --num-episodes 30 \
  --output results/comparison.json
```

## How It Works

1. **Load trained checkpoint** (byte-transformer model)
2. **Format input:** `<MAP>...<ACT>history...`
3. **Greedy decode** next action: `<ACT>action</ACT>`
4. **Execute action** via AGI Maze API
5. **Repeat** until done or max steps

## Expected Performance

Well-trained model (150 epochs):
- **Success rate:** 85-95%
- **Avg steps overhead:** +0-2 steps vs optimal

Undertrained model:
- **Success rate:** 40-70%
- **Avg steps overhead:** +2-5 steps

## Arguments

### neural_agent.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Path to .pt checkpoint |
| `--base-url` | http://127.0.0.1:8000 | AGI Maze server URL |
| `--path` | required | Maze path (e.g., TRAINING/S0-keys/STAGE-01) |
| `--seed` | None | Random seed (None = server picks) |
| `--num-episodes` | 1 | Number of episodes to run |
| `--max-steps` | 50 | Maximum steps per episode |
| `--device` | cpu | Device (cpu/cuda/mps) |
| `--verbose` | True | Show detailed output |

### compare_agents.py

Additional arguments:
- `--rational-agent`: Path to pure_algorithmic_agent.py (required)
- `--start-seed`: Starting seed for episodes
- `--output`: JSON output path


## Troubleshooting

### "Model failed to predict valid action"

**Problem:** Model outputs invalid format

**Solution:**
- Check training loss (ACT loss should be <0.5)
- Train longer (150+ epochs)
- Increase model capacity (d_model)

### Low success rate (<70%)

**Problem:** Model not trained well enough

**Solution:**
- Train on more diverse data
- Increase epochs (try 200)
- Check dataset quality

### Slow inference

**Problem:** CPU inference is slow

**Solution:**
```bash
# Use GPU if available
python3 scripts/neural_agent.py \
  --checkpoint runs/model.pt \
  --device cuda \
  --num-episodes 100
```

## Integration

Neural agent can be used anywhere the rational agent is used:

```python
from neural_agent import load_checkpoint, run_episode

# Load model
model, info = load_checkpoint("runs/model.pt")

# Run episode
episode = run_episode(
    model=model,
    base_url="http://127.0.0.1:8000",
    path="TRAINING/S0-keys/STAGE-01",
    seed=42,
)

print(f"Success: {episode.success}, Steps: {episode.total_steps}")
```

## Files

```
scripts/
├── neural_agent.py           # Neural agent implementation
├── compare_agents.py         # Comparison script
└── train_byte_transformer_with_actions.py  # Training

runs/
└── rational-3x3-keys-mixed.pt  # Trained checkpoint
```

## See Also

- Training docs: `RATIONAL_AGENT_TRAINING.md`
- Rational agent: `agimaze-bench/agents/pure_algorithmic_agent.py`
- Experiment configs: `experiments/rational-agent/seq/*.toml`
