# ReID threshold evaluation

The best seed-555 triplet projection was evaluated on 65 approved
validation triplets (130 same/different pairs).

| Measure | Result |
|---|---:|
| Triplet ordering accuracy | 90.8% |
| Calibrated cosine threshold | 0.2685 |
| Pair accuracy on calibration split | 79.2% |
| False positive matches | 23 |
| False negative matches | 4 |
| Mean same-vehicle similarity | 0.540 |
| Mean different-vehicle similarity | 0.232 |

The projection ranks the correct vehicle ahead of a sampled negative reliably, but a single
hard threshold still produces too many false matches. Use the model to rank proposed track
fragment links for review; do not automatically merge identities from this threshold.
