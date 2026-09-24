# Bike-route vehicle typology pipeline

Goal: for every vehicle seen from the bike, decide how it relates to the rider, which is the basis for scoring
how safe a route feels.

| Fine relation | Reporting group (professor's typology) |
|---|---|
| `ego_lane`: in the rider's lane of travel | in my lane |
| `adjacent_same`: other lane, same direction (including overtaking) | sharing the road |
| `oncoming`: opposite direction, same roadway, no median | sharing the road |
| `parked` | parked |
| `cross_side`: side roads, crossing or turning traffic, driveways, anything beyond a median or on another roadway | not sharing |

Per second the pipeline also reports the riding context (`separated_path`, `bike_lane`, `shared_road`), ego speed,
closest distances, door-zone exposure next to parked cars, and events such as overtakes.

## How it works

Think of it as a data pipeline: two heavy GPU extract steps, then cheap CPU transforms you can rerun at will.

1. **Perception (GPU, once per video)** - `bikesafe.perceive`
   - YOLO11n at 1280 px detects cars, motorcycles, buses and trucks. The tuned BoT-SORT tracker gives each
     vehicle an ID across frames, the same detector and tracker selected in the earlier milestone.
   - Frames are processed at ~12 fps. On the approved 300-frame sequence this keeps IDF1 at 69-70% (72.5% at full
     rate) while running about 2.5x faster. Camera-motion compensation reuses the dense optical flow instead of
     recomputing sparse features (`bikesafe.validate_tracking`, `results/typology/tracking_rate_validation.csv`).
   - Dense optical flow (DIS) is saved as a 16x9 background-motion grid with vehicles masked out, plus flow measurements
     inside and under every box.
   - CLIP ViT-B/32 embeds the frame and the road ahead once per second, plus crops of sizeable vehicles.
2. **Depth and lane structure (GPU, sampled at 2 fps)** - `bikesafe.scene3d`
   - Depth Anything V2 (metric, outdoor) gives a depth map; white and yellow road markings are detected in the image
     and projected into a bird's-eye grid with that depth, accumulated over a second so dashed lines are not missed.
   - Each vehicle then gets: how many lane lines lie between the rider and it, whether any of them is a yellow centre
     line, and how far it sits from the nearest line. This is the semantic the first version lacked - a human decides
     "my lane or the next one" from the paint, not from a distance in metres.
   - The learned depth is *not* used for vehicle distance: measured against car geometry it compresses range (ratio
     1.26 at 5 m falling to 0.64 at 30 m) and tracks the rider's own motion at only -0.61 m per metre ridden, against
     -0.92 for the wheel-row/box-height geometry. It is kept for the lane map, where only relative geometry matters.
3. **Geometry and track features (CPU)** - `bikesafe.geometry`, `bikesafe.tracks`
   - *Self-calibration.* Cars are all about the same height, so across thousands of boxes the wheel row grows
     linearly with box height. The intercept is the horizon and the slope is the camera height (~1.1 m here). Box
     widths of cars seen square-on give the lateral scale. No manual calibration is needed per video.
   - *Ego motion.* Road-surface flow below the horizon follows `dv = g*d^2`, which gives forward speed (`g`), yaw and
     heading (focus of expansion) for every frame.
   - *Vehicle motion.* A parked car's wheel row changes exactly as the static road predicts. The difference gives
     each vehicle's own along-road speed (sign = same way / oncoming). Its yaw-corrected lateral drift gives crossing
     motion. Lateral offset from the rider's heading gives lane position.
   - *World trajectory.* Removing the rider's own advance and turn from a track leaves the vehicle's motion in the
     world. A least-squares fit over the whole track gives its speed and heading: ~0 deg travels with the rider,
     ~180 deg is oncoming, ~90 deg crosses. On labelled tracks these land where they should (oncoming 139-161 deg,
     crossing 77-110 deg, same direction 9-56 deg, parked 1.8 m/s), which is the single most discriminative cue.
   - Each track is summarised into ~70 features (position, speed and crossing statistics, world heading, lane lines
     crossed, size, entry/exit edges, CLIP viewpoint front/rear/side, emergency-vehicle score, scene scores).
   - A readable rule baseline (`rule_relation`) turns these features into a relation.
4. **Learning and evaluation** - `bikesafe.cards`, `bikesafe.train`
   - Track cards (start / middle / end with path / zoom) are sampled across all videos and relation strata and
     labelled per `data/typology/LABELING_GUIDE.md`. Scene frames are sampled every 20 s.
   - Models are compared with **leave-one-video-out** validation, so every score comes from a ride the model never
     saw: rules, a depth-5 decision tree (learned readable rules) and gradient boosting.
5. **Outputs** - `bikesafe.exposure`, `bikesafe.render`
   - `results/corpus/<video>/tracks_typed.csv`, `timeline_1s.csv` (with local timestamps for a Strava join),
     `events.csv`, and `results/corpus/corpus_summary.csv`.
   - Colour-coded overlay clips in `demos/`.

## Running at scale on a GPU workstation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-typology.txt
python -m bikesafe.run D:\rides --out-root D:\rides_out --render-minutes 1
```

Every stage is resumable. On a modern desktop GPU, `--target-fps 24 --gmc sparse` restores full-rate tracking, and
`bikesafe.scene3d --sample-fps 4` gives denser lane geometry.

## Known limitations

- Metric distances and speeds assume a 110 degree horizontal field of view for the focal length. Relations do not
  depend on it, because they use ratios, but absolute m and m/s should be recalibrated once against Strava speed.
- Flat-road geometry. Hills and strong bike pitch bias distances until the rolling horizon re-settles.
- Night footage reduces detection recall and flow quality.
- The first labels were produced by AI-assisted visual review and should be spot-checked by a human reviewer
  (`labeler` column). The pipeline retrains from the CSVs.
