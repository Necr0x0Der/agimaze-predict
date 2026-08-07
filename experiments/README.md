# Experiments

## Per-step datasets

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

## Seq (n-step prediction) datasets

[DESCRIPTION]

### Byte-Transformer Results
#### 3x3-keys-2step
- Test: 3x3-keys-2step-rnd-valid
- Base training set
  - Train: 3x3-keys-2step-rnd-train1 (2000 samples, 140 epochs)
  - Result: 0.964
- Training set x2
  - Train: 3x3-keys-2step-rnd-train1, 3x3-keys-2step-rnd-train2 (4000 samples, 150 epochs)
  - Result: 0.9980

#### 3x3-keys-4step
- Test: 3x3-keys-4step-rnd-valid
- Base training set
  - Train: 3x3-keys-4step-rnd-train1 (2000 samples, 270 epochs)
  - Result: 0.845
- Base training set x2
  - Train: 3x3-keys-4step-rnd-train1, 3x3-keys-4step-rnd-train2 (4000 samples, 230 epochs)
  - Result: 0.901
- Extended set with 2-step
  - Train: 3x3-keys-2step-rnd-train1, 3x3-keys-2step-rnd-train2, 3x3-keys-4step-rnd-train1, 3x3-keys-4step-rnd-train2 (8000 samples, 210 epochs)
  - Result: 0.954
- Extended set with 2-step + 8-step
  - Train: 3x3-keys-2step-rnd-train1, 3x3-keys-2step-rnd-train2, 3x3-keys-8step-rnd-train1, 3x3-keys-8step-rnd-train2, 3x3-keys-4step-rnd-train1, 3x3-keys-4step-rnd-train2 (8000 samples, 265 epochs)
  - Result: 0.979

#### 4x4-keys-2step
- Test: 4x4-keys-2step-rnd-valid
- Base training set
  - Train: 4x4-keys-2step-rnd-train1 (2000 samples, 175 epochs)
  - Result: 0.726
- Base training set x2
  - Train: 4x4-keys-2step-rnd-train1, 4x4-keys-2step-rnd-train2 (4000 samples, 235 epochs)
  - Result: 0.894
- Multi-step training set
  - Train: 4x4-keys-1step-rnd-train1, 4x4-keys-1step-rnd-train2, 4x4-keys-4step-rnd-train1, 4x4-keys-4step-rnd-train2, 4x4-keys-2step-rnd-train1, 4x4-keys-2step-rnd-train2 (12000 samples, 220 epochs)
  - Result: 0.932
- Mixed extended training set
  - Train (20000 samples, 240 epochs):
    - 4x4-keys-1step-rnd-train1
    - 4x4-keys-1step-rnd-train2
    - 4x4-keys-4step-rnd-train1
    - 4x4-keys-4step-rnd-train2
    - 3x3-keys-1step-rnd-train1
    - 3x3-keys-1step-rnd-train2
    - 3x3-keys-2step-rnd-train1
    - 3x3-keys-2step-rnd-train2
    - 4x4-keys-2step-rnd-train1
    - 4x4-keys-2step-rnd-train2
  - Result: 0.951
- Conclusion: as it can be seen, the sampling efficiency in this case is quite low. We didn't extend the training set with more examples from the same narrow distribution (4x4-keys-2step), but augmented it with examples from other similar tasks to verify transfer and generalization capabilities.

#### 3x4-rivers-2step
- Test: 3x4-rivers-2step-rnd-valid
- Extended mixed training set (20000 samples, 240 epochs):
  - 4x4-keys-1step-rnd-train1
  - 4x4-keys-1step-rnd-train2
  - 4x4-keys-2step-rnd-train1
  - 4x4-keys-2step-rnd-train2
  - 3x4-rivers-1step-rnd-train1
  - 3x4-rivers-1step-rnd-train2
  - 3x4-rivers-4step-rnd-train1
  - 3x4-rivers-4step-rnd-train2
  - 3x4-rivers-2step-rnd-train1
  - 3x4-rivers-2step-rnd-train2
- Result: 0.881
- Conclusion: even 2-step predicion with rivers is hard (even given 1-step prediction examples for the same traces)
