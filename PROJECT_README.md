# Vehicle detection and tracking — video 1

Pretrained YOLO11 detection plus ByteTrack and BoT-SORT baselines on dashcam footage.
The detector processes the full video; tracking is evaluated on a 60-second busy clip.

**Read [`outputs/MEETING_REPORT.md`](outputs/MEETING_REPORT.md) first** for the current
milestone, then [`outputs/FINDINGS.md`](outputs/FINDINGS.md) for the original detection pass.

## Layout

```
car_detection_project/
├── videos/video1.mp4                  1920x1080, 29.97 fps, 30:00, 53,955 frames
├── clips/video1_clip60.mp4            60s test clip cut from 14:40-15:40
├── outputs/
│   ├── FINDINGS.md                    <- the writeup
│   ├── MEETING_REPORT.md              <- current detection/tracking milestone
│   ├── REID_PILOT.md                  <- bounded appearance-ReID comparison
│   ├── reid_triplet_summary/          <- repeated triplet-loss evaluation
│   ├── analysis_fullvideo/            corrected stats, per-frame counts, charts
│   ├── tracking/                      ByteTrack and BoT-SORT CSVs/videos/analysis
│   ├── detections_FULLVIDEO_yolo11n_1280.csv   387,267 detections, full 30 min
│   ├── detections_yolo11n_cars.csv    11,243 car detections (60s clip)
│   ├── video1_clip60_ALL_CLASSES_yolo11n_640.mp4
│   ├── video1_clip60_CARS_yolo11n_640.mp4
│   ├── video1_clip60_CARS_yolo11n_1280.mp4
│   ├── comparisons/                   side-by-side frames across configs
│   ├── survey_frames/                 source frames sampled across the 30 min
│   ├── annotated_frames_all_classes/  stills pulled from the annotated video
│   ├── 01_yolo11n_all_classes/        raw ultralytics output (.avi)
│   ├── 02_yolo11n_cars_only/          .avi + labels/ (per-frame .txt)
│   ├── 03_yolo11s_cars_only/          labels/
│   ├── 04_yolo11n_1280_cars_only/     labels/
│   └── 05_FULLVIDEO_yolo11n_1280/     labels/ - 52,374 files, full 30 min
├── scripts/
│   ├── summarize_detections.py        labels -> CSV + summary stats
│   ├── compare_runs.py                compare detection counts across runs
│   ├── compare_frame.py               stack one frame across runs, for eyeballing
│   ├── render_clean_overlay.py        redraw boxes readably from saved labels
│   ├── analyze_counts.py              zero-aware statistics and charts
│   ├── track_vehicles.py              persistent IDs, CSV, video and stills
│   ├── summarize_tracks.py            track-duration/fragmentation proxies
│   ├── prepare_detection_review.py    stratified human-review seed set
│   ├── prepare_tracking_review.py     contiguous identity-review sequence
│   ├── build_triplet_manifest.py      reviewed-ReID preparation
│   ├── train_triplet_reid.py          full fine-tuning option with review guard
│   ├── train_triplet_projection.py    CPU-efficient frozen-feature triplet training
│   ├── evaluate_reid_projection.py    threshold calibration and pair diagnostics
│   ├── summarize_reid_runs.py         repeated-run result aggregation
│   └── finalize_approved_reviews.py   freeze approved labels and splits
├── datasets/                          human-review and pseudo-label workspaces
├── requirements.txt
└── venv/
```

## Environment

Python 3.12.10, CPU-only (no NVIDIA GPU on this machine).

| Package | Version |
|---|---|
| ultralytics | 8.4.138 |
| torch | 2.14.0+cpu |
| opencv-python | 5.0.0 |
| pandas | 3.0.5 |

torch was installed from the CPU index deliberately — the default wheel pulls ~2 GB of
CUDA libraries that cannot be used here:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

ffmpeg was also installed (`winget install Gyan.FFmpeg`) for clipping and re-encoding.

## Reproducing

Activate the venv:

```bash
venv\Scripts\activate
```

Baseline run — all classes:

```bash
yolo predict model=yolo11n.pt source="clips/video1_clip60.mp4" project=outputs name=01_yolo11n_all_classes
```

Cars only, with machine-readable labels:

```bash
yolo predict model=yolo11n.pt source="clips/video1_clip60.mp4" classes=2 save_txt=True save_conf=True project=outputs name=02_yolo11n_cars_only
```

Higher input resolution (the configuration that performed best here):

```bash
yolo predict model=yolo11n.pt source="clips/video1_clip60.mp4" classes=2 imgsz=1280 save_txt=True save_conf=True project=outputs name=04_yolo11n_1280_cars_only
```

Turn labels into a CSV and summary:

```bash
python scripts/summarize_detections.py outputs/02_yolo11n_cars_only/labels outputs/detections_yolo11n_cars.csv
```

Compare runs:

```bash
python scripts/compare_runs.py "n@640=outputs/02_yolo11n_cars_only/labels" "s@640=outputs/03_yolo11s_cars_only/labels" "n@1280=outputs/04_yolo11n_1280_cars_only/labels"
```

## Label format

One `.txt` per frame, named `video1_clip60_<frame>.txt`, one detection per line,
coordinates normalised 0–1:

```
class  x_center  y_center  width  height  confidence
2      0.519397  0.552870  0.243320  0.309742  0.885117
```

Class `2` is `car` in COCO. `detections_yolo11n_cars.csv` is the same data flattened,
with pixel coordinates, timestamps and box areas added.

## Tracking baseline

Both trackers use YOLO11n at 1280px, confidence 0.10, and COCO vehicle classes
`2,3,5,7`. Run them with:

```bash
python scripts/track_vehicles.py clips/video1_clip60.mp4 yolo11n.pt outputs/tracking/bytetrack --tracker bytetrack.yaml
python scripts/track_vehicles.py clips/video1_clip60.mp4 yolo11n.pt outputs/tracking/botsort --tracker botsort.yaml
```

The resulting IDs are tracker hypotheses. They are not ground-truth unique vehicle counts
until the contiguous review sequence in `datasets/tracking_identity_review/` is corrected.

## Reviewed ReID pilot

The approved review set contains 350 triplets from 420 vehicle crops, split into 285
training and 65 validation triplets. The CPU-efficient experiment freezes ImageNet
ResNet-18 features and trains a 128-dimensional triplet-loss projection:

```bash
python scripts/train_triplet_projection.py datasets/reid_triplet_candidates/triplets.csv outputs/reid_triplet_model
python scripts/evaluate_reid_projection.py datasets/reid_triplet_candidates/triplets.csv outputs/reid_triplet_model/best_reid_projection.pt outputs/reid_triplet_evaluation
```

Across three seeds, validation triplet ordering improved from an 80.0% frozen-feature
baseline to 88.2% on average, with a best result of 90.8%. The calibrated binary threshold
reached only 79.2% pair accuracy and produced 23 false matches, so the projection is suitable
for ranking review candidates but is not enabled for automatic ID merging.

The trained projection was also applied to the tuned BoT-SORT output as a tracklet-linking
stage. Up to eight crops represented each track, and candidate merges were gated by time,
class, motion, box size, and cosine similarity. Thresholds from 0.2685 through 0.90 all reduced
IDF1. Online BoT-SORT ReID thresholds of 0.80, 0.90, and 0.95 also scored below the no-ReID
tracker. See `outputs/APPEARANCE_REID_EXPERIMENT.md`; the final configuration therefore keeps
`with_reid: false`.

## Tracker selection benchmark

Six available multi-object trackers and four tuned variants were run with the same
YOLO11n detector, 1280-pixel input, confidence threshold 0.10, vehicle classes, and
the same approved 300-frame sequence. Evaluate them with:

```bash
python scripts/evaluate_trackers_mot.py datasets/tracking_identity_review/approved_tracks.csv outputs/tracker_benchmark/comparison_all \
  botsort=outputs/tracker_benchmark/botsort/botsort_tracks.csv \
  botsort_tuned=outputs/tracker_benchmark/botsort_tuned/botsort_vehicle_tuned_tracks.csv \
  bytetrack=outputs/tracker_benchmark/bytetrack/bytetrack_tracks.csv \
  fasttrack_tuned=outputs/tracker_benchmark/fasttrack_tuned/fasttrack_vehicle_tuned_tracks.csv
```

Tuned BoT-SORT was selected: 81.0% MOTA, 72.5% IDF1, 88.0% recall, 70 identity
switches, and 47 fragments. It slightly improved default BoT-SORT while giving the best
overall identity score. The reference labels began as BoT-SORT output and were approved
rather than independently redrawn, so these numbers are useful for this video but are not
an unbiased public benchmark.

Run the selected pipeline and render the presentation demo with:

```bash
python scripts/track_vehicles.py clips/video1_clip60.mp4 yolo11n.pt outputs/tracking/botsort_vehicle_tuned \
  --tracker configs/botsort_vehicle_tuned.yaml --imgsz 1280 --conf 0.10 --no-video
python scripts/render_tracking_demo.py clips/video1_clip60.mp4 \
  outputs/tracking/botsort_vehicle_tuned/botsort_vehicle_tuned_tracks.csv \
  outputs/tracking/botsort_vehicle_tuned/botsort_tuned_clean_demo.mp4
```

The demo hides short, low-confidence track hypotheses to make persistence easier to inspect.
The complete CSV remains the analysis output; the stable IDs shown in the demo remain tracker
estimates until identities are corrected by a second independent reviewer.

## Notes

- Only `video1.mp4` is present. `video2.mp4` was not in the folder.
- Detection includes a completed full-video YOLO11n/1280 car-only pass. Model comparison
  and tracking runs use the 60-second clip; see FINDINGS.md and MEETING_REPORT.md.
- The raw ultralytics `.avi` outputs in `01_`/`02_` are ~470 MB each (MJPG). The MP4s in
  `outputs/` are the same content re-encoded; the `.avi` files can be deleted.
