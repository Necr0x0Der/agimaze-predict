# Llama + 2-D spatial memory (initial PoC)

This baseline attaches a persistent learned 2-D workspace to
`meta-llama/Llama-3.2-3B-Instruct` without modifying Llama decoder layers.

1. The rendered `MAP` is encoded only by the workspace as a 9×17 byte canvas.
2. Each completed `ACT` is represented by a masked mean of Llama's token
   embeddings and triggers one recurrent workspace update.
3. Learned query vectors read a fixed number of soft memory tokens from the
   final frame.
4. Those tokens are projected to Llama hidden size and inserted after the
   action prompt and immediately before the `<POS>…</POS>` target.

This is intentionally the *single final memory-prefix* version.  It isolates
whether a map-aligned 2-D workspace helps a frozen/pretrained LLM before adding
event-indexed insertion or cross-attention modifications inside Llama.

## Trainable parameters

The intended first run freezes the Llama backbone and trains only the map
encoder, spatial updater/readout, and projection into Llama hidden space.  A
later variant can add LoRA adapters to Llama attention projections.  Full
fine-tuning is deliberately excluded from the first comparison.

## Training evaluation

At epoch 1, every `evaluate_every` epochs, and the final epoch, the trainer
evaluates the held-out `validation_files` from the TOML.  It reports
`val_target_token_nll` (teacher-forced loss only on target POS and EOS tokens)
and `val_greedy_exact_target_accuracy` (the fraction of complete greedy
`<POS>...</POS>` continuations, including EOS, that match exactly).  Greedy
generation builds the map/action memory once and uses Llama's KV cache for the
target tokens, so memory is not accidentally inserted a second time.

## Required controls

At minimum, report the same tokenizer, prompt template, source-maze-disjoint
splits, and greedy exact-match metric for:

1. frozen Llama with no spatial signal (or equal-sized learned constant tokens);
2. spatial memory with action updates disabled;
3. frozen Llama with spatial memory;
4. spatial memory plus LoRA;
5. shuffled maps/actions as negative controls.

The implementation is dependency-light at import time, but actual training
requires `pip install -e '.[llama]'`, accepted Llama model access, and a CUDA
machine.  Install `bitsandbytes` separately only when using 4-bit QLoRA.

## Prompt probe

After training, `scripts/generate_llama_memory.py` can inject the saved spatial
memory into an arbitrary Llama text prefix.  A map remains mandatory; completed
actions are optional and define the workspace state after those actions.  The
map is raw rectangular text (without
`<MAP>` tags), and actions are supplied in chronological order:

```bash
python scripts/generate_llama_memory.py \
  --checkpoint runs/llama-memory-3x3-keys-4step.pt \
  --map-file /path/to/map.txt \
  --action right --action down \
  --prompt $'<ACT>right</ACT>\n<ACT>down</ACT>\n' \
  --stop '</POS>'
```

The `--prompt` may contain arbitrary text, but this checkpoint was trained only
on an action-block prompt followed immediately by a `<POS>...</POS>` target.
For its intended POS probe, it should therefore reproduce that format exactly
(as in the example).  Questions, chat templates, or new tasks are out of
distribution: the frozen Llama may generate fluent text, but there is no reason
to expect the learned memory interface to ground it reliably.
