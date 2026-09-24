# Vehicle Detection and Tracking - Submission Package

## Current milestone: bike lanes, traffic signs and signals, sharper motion cues, faster runs

Response to *"accuracy is still not acceptable, and there is no bike lane or traffic sign detection"*.

1. Read [reports/INFRASTRUCTURE_AND_ACCURACY_REPORT.md](reports/INFRASTRUCTURE_AND_ACCURACY_REPORT.md): what was
   added, how it was checked on real footage, and exactly what still has to be measured on the full corpus.
2. New stage `bikesafe.infrastructure`: painted bike lanes (lane lines and bicycle stencils in a bird's-eye view of
   the road), traffic signals with their state (red / yellow / green), stop signs and other traffic signs. Its
   per-second output joins the timeline, fixes the riding context, and feeds new events (`stop_sign`,
   `red_light_wait`).
3. New depth-free motion features from the optical flow the perception pass already stored, aimed at the most
   frequent error (moving vehicles called parked), plus each vehicle's position relative to the painted lines.
4. Faster: FP16 on the GPU, optional hardware video decoding, CPU stages in parallel across rides, and a per-stage
   timing summary.

On your GPU machine, one command reuses everything already computed, runs only what is new, and retrains:

```bash
python -m bikesafe.run videos --out-root work --render-minutes 1 --hw-decode   # new stages only; prints stage timings
python -m bikesafe.train                                                       # retrain + feature ablation
python -m bikesafe.exposure --analysis work/analysis --perception work/perception   # re-apply the new model
```

`results/typology/feature_ablation.csv` then shows what the new features are worth on the 378 labelled tracks.

## Previous milestone: vehicle typology relative to the rider

Answering the feedback *"would you be able to distinguish cars in my lane, cars sharing the road, parked cars, and cars
on side roads or across the median?"* — yes, and it now runs over all six rides (112 minutes).

1. Read [reports/VEHICLE_TYPOLOGY_REPORT.md](reports/VEHICLE_TYPOLOGY_REPORT.md).
2. Watch [demos/typology_006_bikelane_440s.mp4](demos/typology_006_bikelane_440s.mp4) — boxes coloured by relation to the rider.
3. Inspect [results/corpus/](results/corpus/) for per-ride vehicle tables, per-second timelines (Strava-joinable) and events.
4. Pipeline documentation: [docs/TYPOLOGY_PIPELINE.md](docs/TYPOLOGY_PIPELINE.md); labeling protocol:
   [data/typology/LABELING_GUIDE.md](data/typology/LABELING_GUIDE.md).

Run the whole corpus with `python -m bikesafe.run videos --out-root work --render-minutes 1`
(dependencies in [requirements-typology.txt](requirements-typology.txt)).

## Previous milestone: detection and tracking

The project detects road vehicles, maintains temporary identities across frames, compares alternative tracking approaches, evaluates appearance ReID and triplet loss, and selects the best measured pipeline for the supplied video.

## Recommended review order

1. Read [Vehicle_Tracking_Final_Report.pdf](Vehicle_Tracking_Final_Report.pdf).
2. Watch [demos/best_tracking_demo.mp4](demos/best_tracking_demo.mp4) for the selected 60-second result.
3. Watch [demos/tracker_visual_comparison.mp4](demos/tracker_visual_comparison.mp4) for a synchronized four-way comparison.
4. Inspect [results/tracker_mot_metrics.csv](results/tracker_mot_metrics.csv) and [results/appearance_reid_metrics.csv](results/appearance_reid_metrics.csv) for the numerical evidence.

## Final selection

**YOLO11n at 1280 pixels + tuned BoT-SORT + sparse optical-flow camera-motion compensation.**

| Measure | Selected result |
|---|---:|
| MOTA | 81.0% |
| IDF1 | 72.5% |
| Precision | 94.1% |
| Recall | 88.0% |
| Identity switches | 70 |
| Fragments | 47 |

Appearance embeddings and the trained triplet-loss projection were implemented and tested. Every automatic appearance-assisted variant reduced IDF1, so the final tracker keeps automatic ReID disabled. The embedding remains available as an optional human-review aid.

## Folder contents

- `Vehicle_Tracking_Final_Report.pdf` - polished final report.
- `reports/` - detailed Markdown findings and experiment reports.
- `demos/` - selected tracking demo and synchronized comparison video.
- `results/` - metrics, full selected-track CSV, charts, and review diagnostics.
- `data/` - approved 300-frame evaluation sequence and reference identities.
- `configs/` - selected and experimental tracker configurations.
- `scripts/` - reproducible analysis, training, evaluation, and rendering scripts.
- `models/` - YOLO11n detector and the experimental triplet projection.
- `requirements.txt` - pinned Python dependencies.
- `PROJECT_README.md` - full project documentation and reproduction commands.

## Reproduction

Create a Python environment and install the pinned dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run the selected tracker on a video:

```bash
python scripts/track_vehicles.py VIDEO.mp4 models/yolo11n.pt output \
  --tracker configs/botsort_vehicle_tuned.yaml --imgsz 1280 --conf 0.10 --no-video
```

Render the presentation overlay:

```bash
python scripts/render_tracking_demo.py VIDEO.mp4 \
  output/botsort_vehicle_tuned_tracks.csv output/best_tracking_demo.mp4
```

## Scope and limitations

The 7.8 GB original source video and the machine-specific virtual environment are excluded to keep the submission portable. The included approved ten-second sequence supports evaluation reproduction. The detection and tracking references began as model outputs before human approval, so an independently redrawn, untouched test sequence is still needed for a publication-grade comparison.

Track IDs represent continuity hypotheses within the video. They do not guarantee that a car returning after a long absence is the same physical vehicle, and ordinary frame-to-frame tracking does not require license-plate recognition.
