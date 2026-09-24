# Vehicle Detection and Tracking: Empirical Model Selection

## Decision

The selected pipeline for this video is:

**YOLO11n at 1280 px → tuned BoT-SORT with camera-motion compensation → stable-track presentation filter**

This is the best tested solution for the current moving-camera road video. It was selected from four detectors, six tracker families, four tuned tracker variants, automatic ReID, and a learned triplet-loss ReID pilot. The decision is based on identity and detection metrics measured on the same approved 300-frame sequence.

Triplet loss is retained only as a review-candidate ranking tool. It is not used to merge track identities automatically because its calibrated pair accuracy is not reliable enough.

## What was executed

- Extracted and analyzed all 53,955 frames in the 30-minute source video.
- Ran vehicle detection and produced frame-level CSV output, summary statistics, charts, and annotated examples.
- Prepared and froze 96 approved training frames and 24 approved validation frames.
- Prepared and froze a contiguous 300-frame tracking review sequence with 89 reference identities.
- Compared YOLO11n, YOLO11s, YOLO26n, and YOLO26s under the same validation protocol.
- Compared ByteTrack, BoT-SORT, TrackTrack, FastTracker, OC-SORT, and Deep OC-SORT.
- Tuned BoT-SORT, TrackTrack, FastTracker, and Deep OC-SORT for low-confidence distant vehicles and moving-camera footage.
- Tested automatic BoT-SORT ReID and rejected it after it increased identity fragmentation.
- Trained a reviewed triplet-loss projection with three random seeds and calibrated its matching threshold.
- Ran the selected tracker over the complete 60-second evaluation clip.
- Produced a synchronized four-way comparison video and a clean 60-second presentation demo.

## Detector comparison

| Detector | AP50 | Precision | Recall | F1 | Mean inference |
|---|---:|---:|---:|---:|---:|
| **YOLO11n** | **1.000** | **1.000** | **0.764** | **0.866** | 189 ms/frame |
| YOLO11s | 0.843 | 0.867 | 0.720 | 0.787 | 563 ms/frame |
| YOLO26n | 0.826 | 0.909 | 0.587 | 0.713 | **179 ms/frame** |
| YOLO26s | 0.802 | 0.852 | 0.657 | 0.742 | 467 ms/frame |

**Choice: YOLO11n at 1280 px.** It has the best agreement with the approved labels and is much faster than the larger models. YOLO26n is slightly faster, but its recall and F1 are substantially lower on these frames.

The validation labels began as YOLO11n predictions before human approval, so this detector comparison favors YOLO11n. The result supports consistency with the accepted labels; it is not an independent detector benchmark. A future independent redraw of the validation boxes is needed before claiming general superiority.

## Tracker comparison

All tracker results below use the same 300 frames, detector, image size, confidence threshold, and vehicle classes. IDF1 measures identity consistency, MOTA combines missed detections, false detections, and identity switches, and the switch/fragment counts diagnose persistence failures.

| Rank | Tracker | IDF1 | MOTA | Recall | Precision | ID switches | Fragments |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Tuned BoT-SORT** | **72.5%** | **81.0%** | 88.0% | 94.1% | 70 | **47** |
| 2 | Default BoT-SORT | 72.3% | 80.9% | 87.3% | **94.9%** | 77 | 67 |
| 3 | ByteTrack | 68.4% | 77.2% | 83.3% | **94.9%** | 76 | 66 |
| 4 | Tuned FastTracker | 68.2% | 74.3% | 85.8% | 89.3% | **53** | 76 |
| 5 | FastTracker | 67.0% | 75.0% | 87.3% | 89.1% | 73 | 97 |
| 6 | Tuned TrackTrack | 65.5% | 63.2% | **91.7%** | 77.3% | 71 | 53 |
| 7 | Tuned Deep OC-SORT | 63.2% | 76.9% | 89.0% | 90.7% | 133 | 67 |
| 8 | OC-SORT | 60.2% | 68.9% | 73.3% | 97.3% | 106 | 166 |
| 9 | Deep OC-SORT | 59.6% | 65.1% | 68.2% | 98.2% | 86 | 181 |
| 10 | TrackTrack | 43.6% | 29.0% | 29.3% | 99.5% | 4 | 23 |

**Choice: tuned BoT-SORT.** It has the highest IDF1 and MOTA, keeps strong precision and recall, and reduces switches and fragments versus default BoT-SORT. Tuned FastTracker has fewer switches, but its lower IDF1 and MOTA mean the overall identity assignments are less accurate. Tuned TrackTrack has the highest recall but too many false associations, which reduces MOTA.

The tracking reference started from default BoT-SORT output and was approved rather than independently redrawn. That can favor BoT-SORT. The synchronized comparison video is included so these numbers can be checked visually. Independent identity annotation remains the main step needed for a publication-grade benchmark.

## Selected tuning

The chosen configuration uses:

- high-confidence association threshold: 0.25
- low-confidence recovery threshold: 0.10
- new-track threshold: 0.32
- track buffer: 60 frames, approximately 2 seconds
- association match threshold: 0.85
- confidence/overlap score fusion
- sparse optical-flow global motion compensation
- automatic appearance ReID disabled

The longer buffer helps vehicles survive short occlusions. Camera-motion compensation stabilizes association while the recording vehicle moves. The stricter new-track threshold prevents weak detections from immediately becoming new identities.

## Full 60-second result

| Result | ByteTrack | Default BoT-SORT | **Tuned BoT-SORT** |
|---|---:|---:|---:|
| Raw identity hypotheses | 582 | 441 | **285** |
| Detections assigned to tracks | 20,432 | 23,050 | 22,648 |
| Median observed track duration | 0.40 s | 0.70 s | **1.07 s** |
| Tracks shorter than one second | 427 | 263 | **137** |
| Longest track span | — | 49.15 s | **49.15 s** |

Compared with default BoT-SORT, tuning reduced raw identity hypotheses by 35.4%, reduced sub-second tracks by 47.9%, and increased median observed duration by 52.4%. The validated sequence indicates that these gains are accompanied by a small improvement in identity accuracy, rather than being explained only by aggressive merging.

The clean demo displays 205 tracks that have at least 15 observations, adequate median confidence, and adequate median box area. The complete 285-track CSV remains available for analysis. Filtering affects presentation only; it does not alter the benchmark.

## Triplet loss and semi-supervised ReID

The reviewed dataset contains 420 crops and 350 triplets, split into 285 training and 65 validation triplets. A frozen ResNet-18 feature extractor plus a learned 128-dimensional triplet projection was trained with three seeds.

- frozen-feature baseline ordering accuracy: 80.0%
- trained ordering accuracy: 90.8%, 86.2%, and 87.7%
- mean trained ordering accuracy: 88.2%
- best calibrated pair accuracy: 79.2%
- false matches at the selected threshold: 23
- missed matches at the selected threshold: 4

The appearance methods were subsequently integrated with the selected tuned BoT-SORT tracker on the exact approved sequence. Online ReID was tested with appearance thresholds of 0.80, 0.90, and 0.95. Its best IDF1 was 68.8%, below the 72.5% no-ReID result, and it increased identity switches and fragments.

The custom triplet embedding was also applied as a conservative tracklet-linking stage using class, time, motion, box-size, and appearance gates. The strictest 0.90 threshold reduced IDF1 to 71.0%; the calibrated 0.2685 threshold reduced it to 60.5%. Both results argue against automatic ReID merging at this stage. The learned embedding can still rank suspicious split tracks for human review without allowing false matches to corrupt final identities.

## Status against the professor’s requested evidence

| Requirement | Status | Evidence |
|---|---|---|
| Frame extraction | Complete | full frame-level analysis and review frames |
| Vehicle detection/counting | Complete | full-video CSV, summary, plots, annotated frames |
| Labels | Complete for the pilot | approved train/validation detection labels and tracking identities |
| Persistence/tracking | Complete for the evaluated clip | ten tracker variants compared; tuned BoT-SORT selected |
| Re-identification/triplet loss | Tested and bounded | useful as a reviewer aid; rejected for automatic merging |
| Descriptive statistics | Complete | counts over time, detector statistics, track-duration statistics |
| Visual demonstration | Complete | clean 60-second demo, snapshots, synchronized comparison video |

## Practical conclusion

The project should now be presented as an empirical vehicle detection and multi-object tracking study. The evidence supports tuned BoT-SORT as the working solution for this video. Triplet loss is a secondary experiment that demonstrates why appearance matching is difficult; it should not be presented as the main method or as a solved identity merger.

For a stronger research claim, the next data task is an independently redrawn test set containing difficult occlusions, entries/exits, and distant vehicles. That set should remain untouched during tuning. The same tracker comparison can then be rerun to determine whether the selected configuration generalizes beyond the approved pilot.

## Method references

- [Ultralytics multi-object tracking documentation](https://docs.ultralytics.com/modes/track/)
- [Ultralytics tracking dataset format](https://docs.ultralytics.com/datasets/track/)
- [BoT-SORT paper](https://arxiv.org/abs/2206.14651)
- [ByteTrack paper](https://arxiv.org/abs/2110.06864)
- [Deep OC-SORT paper](https://arxiv.org/abs/2302.11813)
- [BoostTrack++ paper](https://arxiv.org/abs/2408.13003)
- [BoxMOT tracker collection and benchmark](https://github.com/mikel-brostrom/boxmot)
