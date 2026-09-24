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

Per second the pipeline also reports the riding context (`separated_path`, `bike_lane`, `shared_road`), whether the
rider is in a painted bike lane, the state of the traffic signal ahead, the signs in view, ego speed, closest
distances, door-zone exposure next to parked cars, and events such as overtakes, stop signs passed and waits at red
lights.

## How it works

Think of it as a data pipeline: a few heavy GPU extract steps, then cheap CPU transforms you can rerun at will.

1. **Perception (GPU, once per video)** - `bikesafe.perceive`
   - YOLO11n at 1280 px detects cars, motorcycles, buses and trucks. The tuned BoT-SORT tracker gives each
     vehicle an ID across frames, the same detector and tracker selected in the earlier milestone.
   - Frames are processed at ~12 fps. On the approved 300-frame sequence this keeps IDF1 at 69-70% (72.5% at full
     rate) while running about 2.5x faster. Camera-motion compensation reuses the dense optical flow instead of
     recomputing sparse features (`bikesafe.validate_tracking`, `results/typology/tracking_rate_validation.csv`).
   - Dense optical flow (DIS) is saved as a 16x9 background-motion grid with vehicles masked out, plus flow measurements
     inside and under every box.
   - CLIP ViT-B/32 embeds the frame and the road ahead once per second, plus crops of sizeable vehicles.
   - On CUDA the detector and CLIP run in half precision (`--fp32` to switch off).
2. **Ego motion (CPU)** - `bikesafe.egomotion`: pyramidal Lucas-Kanade on road features 5-12 m ahead gives forward
   speed, yaw and heading per frame. `bikesafe.run` runs it for several rides at once.
3. **Depth and lane structure (GPU, sampled at 2 fps)** - `bikesafe.scene3d`
   - Depth Anything V2 (metric, outdoor) gives a depth map; white and yellow road markings are detected in the image
     and projected into a bird's-eye grid with that depth, accumulated over a second so dashed lines are not missed.
   - Each vehicle then gets: how many lane lines lie between the rider and it, whether any of them is a yellow centre
     line, and how far it sits from the nearest line.
   - The learned depth is *not* used for vehicle distance: measured against car geometry it compresses range (ratio
     1.26 at 5 m falling to 0.64 at 30 m) and tracks the rider's own motion at only -0.61 m per metre ridden, against
     -0.92 for the wheel-row/box-height geometry. It is kept for the lane map, where only relative geometry matters.
4. **Traffic lights, signs and bike lanes (GPU, sampled at 2 fps)** - `bikesafe.infrastructure`
   - *Lights and signs.* YOLOE-26 (open-vocabulary detection, ultralytics) gets a road-furniture vocabulary - traffic
     light, stop / yield / speed-limit / street-name / warning / no-parking signs and so on - plus distractor classes
     (billboard, shop sign, licence plate, street lamp) that absorb look-alikes. The prompts' text features are cached
     in `models/infrastructure_vocab.npz`, so the 240 MB text encoder is not needed at run time.
   - *Signal state.* The lit lamp of each signal head is found as a bright, saturated, round blob inside the housing
     (lit reds look pink-magenta on this camera, LED greens cyan-green); sky and walls behind side-on heads are
     rejected because they touch the crop's edge or form tall strips.
   - *Stop signs* are confirmed by the COCO-trained YOLO11m (class "stop sign"): the open vocabulary alone also calls
     DO NOT ENTER, NO PARKING and red shop signs stop signs. Other sign types are open-vocabulary best guesses.
   - Detections of one physical sign or signal head are linked across samples: a static object moves straight away
     from the focus of expansion and grows, so the previous box is scaled about the FOE and matched by IoU.
   - *Bike lanes* come from a bird's-eye view of the road ahead, warped with the same self-calibrated flat-road
     geometry as everything else. Lane lines run vertically there and painted stencils look like upright icons:
     - paint is found as thin (<= 0.35 m), elongated strokes brighter than the asphalt beside them; kerb concrete,
       crosswalk bars, cracks and shadow edges are rejected by width and length;
     - lines are extracted greedily (take the best-supported line, remove the paint it explains, repeat), each with
       its own slant, and classified solid / broken and white / yellow (by saturation: faded yellow ~35-70, white in
       low sun below ~30);
     - bicycle stencils are detected by YOLOE in the bird's-eye view, where the large model finds them and the
       foreshortened camera view does not;
     - per sample, transparent rules turn this into bike-lane evidence (a solid white line 0.3-2.4 m to the left, a
       narrow lane between two lines, a stencil in the rider's corridor, green paint; against: a traffic lane between
       broken lines 3+ m apart, or a yellow centre line beside the rider), which is averaged over 7 s and held through
       junctions without paint.
   - Outputs `infrastructure.parquet` (objects) and `lanes.parquet` (lane state per sample) per ride.
5. **Geometry and track features (CPU)** - `bikesafe.geometry`, `bikesafe.tracks`
   - *Self-calibration.* Cars are all about the same height, so across thousands of boxes the wheel row grows
     linearly with box height. The intercept is the horizon and the slope is the camera height (~1.1 m here). Box
     widths of cars seen square-on give the lateral scale. No manual calibration is needed per video.
   - *Vehicle motion.* A parked car's wheel row changes exactly as the static road predicts. The difference gives
     each vehicle's own along-road speed (sign = same way / oncoming). Its yaw-corrected lateral drift gives crossing
     motion. Lateral offset from the rider's heading gives lane position.
   - *World trajectory.* Removing the rider's own advance and turn from a track leaves the vehicle's motion in the
     world. A least-squares fit over the whole track gives its speed and heading.
   - *Depth-free motion from optical flow (version 3).* The perception pass stores the flow inside each box and on the
     ground just below it; they were not used before. `closing_ratio` is the body's expansion over the expansion a
     static object at that distance would show (1 parked, 0 moving with the rider, 2 oncoming at the rider's speed);
     `v_lat_flow` is the body's horizontal motion relative to the ground in the same image column (the rider's own
     motion and yaw cancel); `grid_proj` compares the body's motion with the background beside it, which also works
     for boxes cut off by the frame edge. These target the most frequent version-2 error: moving vehicles called
     parked.
   - *Lane-paint relations (version 3).* Per vehicle: painted lines between the rider's path and the vehicle at its
     distance, solid and yellow ones among them, whether it sits inside the rider's lane, and the share of the track
     during which the rider was in a bike lane.
   - Each track is summarised into ~90 features; a readable rule baseline (`rule_relation`) turns them into a relation.
   - *One id per physical vehicle (version 4)* - `bikesafe.stitch`. Two tracker artefacts make one vehicle look like
     several. **Duplicates**: the detector suppresses overlapping boxes class by class, so a pickup, van or SUV is
     often boxed twice at once, as "car" and as "truck" or "bus"; two tracks that coexist with a mean IoU >= 0.7 over at
     least half of the shorter one are one vehicle. **Breaks**: a missed detection or short occlusion makes the tracker
     start a new id; the last detection of a track is extrapolated forward and the first detection of a later track
     backward to the middle of the gap, and the pair is joined when the predictions meet and the image velocities agree
     (`e + 0.5 dv <= 0.45`, both relative to box height), the gap is <= 2 s, class and size agree, neither end is at
     the side edge of the image, no other vehicle fits about as well, and the first vehicle is gone before the second
     appears. Tracks keep their own features and labels; `vehicle_id` groups them, and `vehicle_links.parquet` lists
     every join with its evidence. On the demo clips the joins were checked by eye: all break joins and the 12
     duplicate merges with the lowest overlap were correct (`reports/UPGRADE_REPORT_2026-09.pdf`).
   - Track tables record `features_version`; `bikesafe.run` recomputes tables written by older versions.
6. **Learning and evaluation** - `bikesafe.cards`, `bikesafe.train`
   - Track cards (start / middle / end with path / zoom) are sampled across all videos and relation strata and
     labelled per `data/typology/LABELING_GUIDE.md`. Scene frames are sampled every 20 s.
   - Models are compared with **leave-one-video-out** validation, so every score comes from a ride the model never
     saw: rules, a depth-5 decision tree (learned readable rules) and gradient boosting.
   - `results/typology/feature_ablation.csv` scores the version-2 features against v2 + flow and the full version 3,
     for each model, on the same labels - the direct measure of what the new features are worth.
   - The riding context is scored for the CLIP probe alone and for the probe fused with the lane-paint detector.
7. **Licence plates** - `bikesafe.plates`
   - *Blurring.* A small plate detector (YOLOv9-tiny ONNX from open-image-models, 7 MB, downloaded on first use) runs on
     every vehicle box >= 40 px tall, cropped from the full-resolution frame; a plate found on a track stays blurred for
     6 more frames at the same place on the vehicle. `bikesafe.render` blurs by default (`--no-blur`), and
     `python -m bikesafe.plates blur <videos or images>` blurs existing files.
   - *Identity audit (opt-in, `bikesafe.run --read-plates`).* Plates of large vehicles are read on up to 6 frames per
     track by a small OCR model (fast-plate-ocr). The text never leaves the process: each read becomes an HMAC-SHA256
     hash under a random key made for that run and never stored, so hashes compare only within one ride, cannot be
     reversed and cannot follow a vehicle across rides. A plate counts only if it sits in the lower middle of its own
     vehicle's box (the neighbour's plate often shows in a crop) and two reads agree. `plates.parquet` (hashes and
     counts) stays in the analysis folder; the corpus outputs get counts only (`identity_audit.json`, `plate_*`
     columns): tracks with one plate split over several vehicle ids (missed joins, or a vehicle seen again later) and
     vehicle ids carrying two plates (wrong joins).
8. **Outputs** - `bikesafe.exposure`, `bikesafe.render`
   - `results/corpus/<video>/tracks_typed.csv`, `timeline_1s.csv` (with local timestamps for a Strava join, bike lane,
     signal state and signs per second), `events.csv` (now including `stop_sign` and `red_light_wait`), `signs.csv`
     (one row per distinct sign / signal head), and `results/corpus/corpus_summary.csv`. The summary adds
     `minutes_in_painted_bike_lane`, `minutes_lane_paint_unknown` (no usable paint view), `minutes_scene_changed_by_paint`
     (seconds whose context differs from the CLIP-only one; on a ride without CLIP that is every paint-decided second),
     `events_stop_sign`, `events_red_light_wait`, `red_light_wait_seconds` and `seconds_with_signal_ahead`.
   - Counts are per physical vehicle (version 4): each vehicle gets one relation, the detection-weighted vote of its
     tracks' class probabilities; `per_min_*`, `vehicles_*`, the per-second counts and the events use vehicles, and
     each event is reported once per vehicle. `tracks_*` keeps the per-track count of earlier runs for comparison, and
     `duplicate_links` / `break_links` say how many joins were made.
   - The riding context fuses CLIP (which separates paths from roads well) with the lane-paint detector (which decides
     bike lane vs shared road wherever it sees paint); the CLIP-only context is kept as `scene_clip`.
   - Colour-coded overlay clips in `demos/`, now with lane lines, signals (by state) and signs, one box per vehicle and
     licence plates blurred.

## Running at scale on a GPU workstation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-typology.txt
python -m bikesafe.run D:\rides --out-root D:\rides_out --render-minutes 1 --hw-decode
python -m bikesafe.train --analysis D:\rides_out\analysis --perception D:\rides_out\perception
python -m bikesafe.exposure --analysis D:\rides_out\analysis --perception D:\rides_out\perception --out results\corpus
```

Every stage is resumable, and existing perception, ego-motion and depth outputs are reused: after updating the code,
the same `bikesafe.run` command only runs the new infrastructure pass and recomputes the track features. The first run
downloads `yoloe-26l-seg.pt` (79 MB) into `models/`. At the end `bikesafe.run` prints how long each stage took.

Speed options: FP16 is on by default on CUDA; `--hw-decode` decodes with the GPU's video engine; `--jobs N` runs the
CPU stages for N rides at once; `--skip-depth` drops the depth pass (removing its features moved leave-one-video-out
scores on the existing labels by -2.3 to +1.6 points depending on the model, i.e. within noise, so it is a reasonable
trade when time matters - retrain afterwards so the model does not expect those features); `--target-fps 24 --gmc
sparse` restores full-rate tracking; `--sample-fps 4` gives denser lane and sign
sampling. For the detector itself, a TensorRT engine is a drop-in replacement:
`yolo export model=models/yolo11n.pt format=engine half=True imgsz=1280`, then pass `--model models/yolo11n.engine`
to `bikesafe.perceive`.

## Known limitations

- Metric distances and speeds assume a 110 degree horizontal field of view for the focal length. Relations do not
  depend on it, because they use ratios, but absolute m and m/s should be recalibrated once against Strava speed.
- Flat-road geometry. Hills and strong bike pitch bias distances and the bird's-eye view until the rolling horizon
  re-settles; lane lines are only judged 2.5-14 m ahead for that reason.
- Night footage reduces detection recall and flow quality, and lane paint is only visible in the headlight.
- Bike-lane evidence is lost where a car alongside hides the lane line; the 7 s smoothing bridges short gaps only.
  Sharrows (shared-lane markings) are stencils without a bike-lane line and should read as shared road, but this has
  not been checked on footage that contains them.
- Sign types other than stop signs and traffic lights are open-vocabulary guesses; counts of "traffic signs" are
  reliable, their fine types are not.
- The first labels were produced by AI-assisted visual review and should be spot-checked by a human reviewer
  (`labeler` column). The pipeline retrains from the CSVs.
- Vehicle ids join a vehicle only across gaps of up to 2 s. A vehicle that leaves the view and returns later (for
  example leapfrogging between traffic lights) is a new vehicle; the opt-in plate audit counts such returns.
- The approved 300-frame reference sequence labels the second (truck) box of a doubly detected pickup as eleven
  separate short identities, so identity scores against it penalise correct duplicate merging; see the upgrade report.
- Plate blurring finds plates on vehicles the detector boxed (>= 40 px tall); a plate on an undetected vehicle is not
  blurred. `data/approved_sequence.mp4` is benchmark input and is left unblurred.
