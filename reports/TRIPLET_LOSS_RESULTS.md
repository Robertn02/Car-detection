# Triplet-loss ReID pilot results

The experiment used 285 approved training triplets and 65
approved validation triplets from 420 vehicle crops. ImageNet-pretrained ResNet-18 features
were frozen; triplet loss trained a 128-dimensional projection. Early stopping selected the
checkpoint with the lowest validation loss.

| Seed | Validation accuracy | Best epoch | Epochs completed |
|---:|---:|---:|---:|
| 555 | 90.8% | 3 | 18 |
| 556 | 86.2% | 13 | 28 |
| 557 | 87.7% | 2 | 17 |

The frozen-feature baseline scored 80.0%. Triplet training averaged
88.2% (sample standard deviation 2.4 percentage
points), an average improvement of 8.2
percentage points. The best run was seed 555 at
90.8%.

This is a pilot on one reviewed clip and one fixed validation split. It supports the value
of learned appearance similarity, but deployment should wait for an independent annotated
sequence and a direct comparison of ID switches against standard BoT-SORT.
