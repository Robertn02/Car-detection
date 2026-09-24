# Appearance Embedding and Triplet-Loss Integration Experiment

## Question

Can appearance embeddings or the trained triplet-loss projection improve the selected tuned BoT-SORT tracker?

## Experiment

All variants were evaluated on the same approved 300-frame sequence with the same YOLO11n detector, 1280-pixel input, confidence threshold 0.10, and vehicle classes. Two forms of appearance matching were tested:

1. **Online BoT-SORT ReID:** appearance features participate directly in detection-to-track association. Appearance thresholds of 0.80, 0.90, and 0.95 were tested.
2. **Custom triplet-loss tracklet linking:** the trained 128-dimensional projection encoded up to eight crops per track. Track descriptors were averaged, then non-overlapping fragments were considered for merging only when class, temporal gap, predicted motion, box size, and appearance all agreed. Six cosine thresholds from the calibrated 0.2685 value through a strict 0.90 value were tested.

## Results

| Variant | IDF1 | MOTA | Precision | Recall | ID switches | Fragments |
|---|---:|---:|---:|---:|---:|---:|
| **Tuned BoT-SORT without ReID** | **72.5%** | **81.0%** | **94.1%** | 88.0% | **70** | **47** |
| Triplet linker, strict 0.90 | 71.0% | 81.0% | 94.1% | 88.0% | 70 | 47 |
| Triplet linker, 0.82 | 69.6% | 81.0% | 94.1% | 88.0% | 70 | 47 |
| Online ReID, 0.80 | 68.8% | 79.0% | 90.1% | **90.6%** | 77 | 63 |
| Online ReID, 0.95 | 68.2% | 80.3% | 91.8% | 90.1% | 79 | 57 |
| Online ReID, 0.90 | 66.6% | 79.8% | 91.4% | 90.2% | 88 | 58 |
| Triplet linker, calibrated 0.2685 | 60.5% | 81.0% | 94.1% | 88.0% | 70 | 47 |

The calibrated triplet threshold accepted 27 links and reduced the 59 input track IDs to 32. Its IDF1 fell by 12.0 percentage points, showing that many of those merges joined different vehicles. Even the strict 0.90 threshold accepted only seven links and reduced IDF1 by 1.5 percentage points.

Online ReID recovered more detections, but the added associations reduced precision and identity consistency. Its best IDF1 was 68.8%, 3.7 percentage points below the no-ReID tracker.

## Decision

**Keep appearance ReID disabled in the final tracker.** The empirical comparison shows that enabling either form of appearance matching makes identity assignments worse on the approved sequence.

The triplet model remains useful for ranking possible matches for human review. Its 90.8% triplet-ordering score measures whether a known positive ranks above one sampled negative; it does not establish that the model can safely choose among many similar cars in an open scene. The tracklet-linking experiment directly tested that harder task and exposed the false-merge problem.

## Why appearance did not help here

- Many vehicles share similar colors, shapes, and viewpoints.
- Distant boxes contain few distinguishing pixels.
- Lighting and scale change as the camera moves.
- The 420-crop training set is small and began from tracker-generated identities.
- Triplet examples do not include enough difficult, visually similar negative vehicles.
- The current reference sequence is only ten seconds long, so the motion-based tracker already handles most short occlusions.

Appearance matching should be reconsidered only after independently correcting fragmented identities, collecting hard negatives with similar vehicle color/body type, and reserving an untouched sequence for threshold selection. A readable license plate could be used as an optional high-confidence cue, but plate recognition is not required for ordinary frame-to-frame tracking.
