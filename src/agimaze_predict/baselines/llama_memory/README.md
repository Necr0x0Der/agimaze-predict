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
