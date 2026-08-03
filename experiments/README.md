# Experiments

### Per-step datasets

Per-step experiments train a model to predict the next agent position from a rendered
maze state and one requested action (`MAP + ACT → POS`). This isolates local movement
geometry from multi-step planning; the JSONL schema, dataset families, and shard sizes
are documented in the [datasets README](../datasets/README.md#per_step).

### Byte-Transformer Results

#### 3x3-keys
- Test: 3x3-keys-rnd-valid
- Train: 3x3-keys-rnd-train (2211 samples, 100 epochs)
- Result: 1.0
- Conclusion: transitions in 3x3 mazes with keys only are perfectly learned using random walks in 100 mazes

#### 4x4-keys
- Test: 4x4-keys-rnd-valid
- Train: 4x4-keys-rnd-train (2364 samples, 250 epochs)
- Result: 0.9887
- Conclusion: Increasing the maze size fom 3x3 to 4x4 results in less than 100% score on the dataset of the approximately same size possibly implying that the transformer doesn't learn general rules but learns local transitions in an ad hoc way.

#### 4x5-keys
- Test: 4x5-keys-rnd-valid
- Base training set
  - Train: 4x5-keys-rnd-train1 (2215 samples, 250 epochs)
  - Result: 0.8253
- Training set x2
  - Train: 4x5-keys-rnd-train1, 4x5-keys-rnd-train2 (4489 samples, 200 epochs)
  - Result: 0.9981
- Training set x3
  - Train: 4x5-keys-rnd-train1, 4x5-keys-rnd-train2, 4x5-keys-rnd-train3 (6758 samples, 150 epochs)
  - Result: 1.0
- Conclusion: The basic training set (random walks in 100 mazes) is insufficient to learn transactions. 200 mazes are enough to achieve a nearly perfect result, which can be turned into 100% score with further increase of the training set. The setup with no rivers and other complex elements is simple, but the sampling efficiency is indicative. 

#### 3x3-rivers
- Test: 3x3-rivers-rnd-valid
- Train: 3x3-rivers-rnd-train (1875 samples, 100 epochs)
- Result: 0.9833
- Conclusion: 3x3 mazes with rivers appear to be not too hard, but the same amount of training data as for 3x3-keys is insufficient to achienve 100% score.

#### 3x4-rivers
- Test: 3x4-rivers-rnd-valid
- Base training set
  - Train: 3x4-rivers-rnd-train1 (2051 samples, 225 epochs)
  - Result: 0.8321
- Training set x2
  - Train: 3x4-rivers-rnd-train1, 3x4-rivers-rnd-train2 (4058 samples, 150 epochs)
  - Result: 0.9354
- Extended set
  - Train: 3x4-rivers-rnd-train1, 3x4-rivers-rnd-train2, 3x3-keys-rnd-train, 4x4-keys-rnd-train, 3x3-rivers-rnd-train (10508 samples, 250 epochs)
  - Result: 0.9557
- Conclusion: 3x4 mazes with rivers appears to be relatively complex. For the base training set, the result is similar to 4x5-keys. However, 2x increase of the training set size helps much less here. Interestingly, adding more mazes of different sizes helps, but the result is not even nearly perfect.
