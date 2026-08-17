# Parallel auxiliary Transformer experiment

This package is intentionally separate from `baselines/byte_transformer`.
It implements the two-stream causal model: a byte main stream and a latent aux
stream with bidirectional causal cross-attention. Its checkpoints use the
separate `agimaze_predict.aux_transformer.v2` format and must be trained and
evaluated with `scripts/train_aux_transformer.py` and
`scripts/evaluate_aux_transformer.py` respectively.

The aux stream has no input tokens. It starts each slot from a learned latent
position embedding; content is initially zero. The collator records the source
prefix length before `<POS>`, and masks ensure aux cannot read any target byte.
