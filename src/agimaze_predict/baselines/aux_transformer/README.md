# Parallel auxiliary Transformer experiment

This package is intentionally separate from `baselines/byte_transformer`.
It implements the two-stream causal model: a byte main stream and a latent aux
stream with bidirectional causal cross-attention. Its checkpoints use the
separate `agimaze_predict.aux_transformer.v3` format and must be trained and
evaluated with `scripts/train_aux_transformer.py` and
`scripts/evaluate_aux_transformer.py` respectively.

The aux stream has no input tokens. It starts each slot from a learned latent
position embedding; content is initially zero. The collator records the source
prefix length before `<POS>`, and masks ensure aux cannot read any target byte.

With `aux_target_weight > 0`, a disposable causal target decoder is trained to
predict `<POS>...</POS>` from the final source-derived aux states plus
teacher-forced preceding target bytes. The target bytes are decoder queries
only: they never become aux keys or values. The decoder is omitted at inference.
Set `aux_gate_mode = "off"` for the required control that retains this loss but
prevents aux states from contributing to the main stream.
