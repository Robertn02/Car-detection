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

Per second the pipeline also reports the riding context (`separated_path`, `bike_lane`, `shared_road`), the lane-paint
detector's view of whether the rider is in a painted bike lane, the state of the traffic signal ahead, the types of the
signs in view and the posted speed limit, ego speed, closest distances, door-zone exposure next to parked cars, and
events such as overtakes, stop signs passed and waits at red lights. Every vehicle also carries its licence plate as
read.

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
   - *Street-scene segmentation.* Mask2Former (Swin-L) trained on Mapillary Vistas
     (`facebook/mask2former-swin-large-mapillary-vistas-semantic`, `bikesafe.streetseg`) labels every pixel of the
     frame (resized to 768 rows; `--seg-height 1088` finds smaller signs at about twice the cost) with 65 street
     classes. Vistas is street-level imagery from cars, bikes and pedestrians on six continents, and it keeps apart
     what this pass needs: *traffic sign (front)* from *traffic sign (back)*, billboards and banners; *lane marking*
     from *crosswalk*; and the *bike lane* surface.
   - *Signs.* Each blob of "traffic sign (front)" is a sign. The open-vocabulary detector used before found sign boxes
     but typed almost all of them "other sign" and also boxed shop signs and billboards; on 26 ride frames the
     segmenter's signs were real traffic signs in all but one case (a shop's standee). Detections of one physical sign
     are linked across samples (below), and every sign is then read once (`bikesafe.signs`):
     - CLIP ViT-L/14 (DataComp-XL weights) compares its four largest views with descriptions of US sign types (stop,
       yield, speed limit, do not enter, one way, no turn, parking rules, bike, pedestrian, warning, street name, guide,
       lane use, bus stop, road work) and averages the probabilities; zero-shot it typed 34 of 46 hand-typed crops
       correctly, the misses being mostly small, blurred street-name signs. *Road work* also takes traffic cones and
       construction barricades: Vistas has no class for them, and the segmenter calls their red and white stripes a
       sign face. The image features of every crop are kept (`sign_clip.npz`), so typing the signs again - with new
       descriptions, rules or a reviewed-sign probe - needs only CLIP's text side;
     - Florence-2-large (Microsoft, MIT licence) reads the text of the largest view when it is at least 28 px tall
       ("SPEED LIMIT 35", "Hoover St", "NO PARKING 8AM-10AM"); text that matches a rule decides the type and gives the
       posted speed limit. The rules were checked against every text read on ride 006 and cover Florence-2's usual
       slips: words one letter off a keyword ("SPELD", "DIKE LANE", "THRU TRAFFIC MERCE LEFT"), two for PARKING and
       CROSSING when the first letter is right ("PATKINL"), "Pico BI" and "BLVO" for Bl and Blvd, "Foy" for Fwy. A
       lone "AHEAD" decides nothing (it is the plaque under a crossing sign, which CLIP types from the sign above);
     - CLIP judges only views it can see. Without deciding text, a sign keeps the generic type `other_sign` -
       counted, not typed - when it was never at least 24 px tall, was seen only edge-on (narrower than 0.3 of its
       height), its largest view is too dark or washed out (grey-level standard deviation under 20: at night), or
       CLIP gives its best type less than half of the probability (0.6 for speed limit); stop and yield signs need
       their word (or, for stop, the COCO check). On a 60 s window of ride 006 most wrong types were CLIP guesses on
       small, blurred or edge-on signs, and red pole markers taken for yield signs. On evening ride 002, CLIP rejected
       or typed dark blobs with high confidence: the contrast gate kept 583 of its 1,079 rejected signs as signs and
       untyped 90 of its 312 types, mostly wrong ones, while on day ride 006 it changes 8 of 619 types;
     - a sign is a *stop* sign when the COCO-trained YOLO11m sees a stop sign on it, or its text reads STOP; signs CLIP
       is sure are not traffic signs, on a view it can judge, are set aside (kind `rejected_sign`) and counted nowhere:
       mostly the backs of signs, billboards, shop signs and wayfinding posts.
   - *Lights.* YOLOE-26 (open-vocabulary detection, ultralytics) with a road-furniture vocabulary - traffic light,
     pedestrian signal, plus distractor classes (billboard, shop sign, licence plate, street lamp) that absorb
     look-alikes - as before. The prompts' text features are cached in `models/infrastructure_vocab.npz`.
   - *Signal state.* The lit lamp of each signal head is found as a bright, saturated, round blob inside the housing
     (lit reds look pink-magenta on this camera, LED greens cyan-green); sky and walls behind side-on heads are
     rejected because they touch the crop's edge or form tall strips.
   - Detections of one physical sign or signal head are linked across samples: a static object moves straight away
     from the focus of expansion and grows, so the previous box is scaled about the FOE and matched by IoU.
   - *Bike lanes* come from a bird's-eye view of the road ahead, warped with the same self-calibrated flat-road
     geometry as everything else. Lane lines run vertically there and painted stencils look like upright icons:
     - paint is the segmenter's lane-marking class warped into the view (before: anything brighter than the asphalt
       beside it, which also picked up cracks, kerbs, shadow edges, sunlit patches and crosswalk bars), kept only as
       thin (<= 0.35 m), elongated strokes;
     - lines are extracted greedily (take the best-supported line, remove the paint it explains, repeat), each with
       its own slant and the stretch ahead in which its paint was seen, and classified solid / broken and white /
       yellow (by saturation: faded yellow ~35-70, white in low sun below ~30);
     - bicycle stencils are detected by YOLOE in the bird's-eye view, where the large model finds them and the
       foreshortened camera view does not;
     - per sample, transparent rules turn this into bike-lane evidence (a solid white line 0.3-2.4 m to the left, a
       narrow lane between two lines, a stencil in the rider's corridor, green paint, bike-lane surface; against: a
       traffic lane between broken lines 3+ m apart, or a yellow centre line beside the rider; a crosswalk in the
       corridor makes the sample uninformative), which is averaged over 7 s and held through junctions without paint.
   - *Where each vehicle stands* (for the relation model, step 5): along the rows at its wheels, the distance from the
     vehicle's outer side to the first kerb / sidewalk / parking / verge pixel, whether another vehicle stands between
     it and the kerb, how much roadway lies on the rider's side, and how much of the ground under it is not roadway.
   - Outputs `infrastructure.parquet` (objects; signs with type, text and speed limit), `lanes.parquet` (lane state per
     sample) and `vehicle_context.parquet` per ride. `--start-s` / `--duration-s` process a window of a ride.
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
   - *Flow consistency and kerb context (version 5).* The medians of the flow cues hide how consistent they are, so
     each track also gets the share of its samples in the static / co-moving / approaching bands of `closing_ratio`
     and `grid_proj`, and the kerb context of step 4 (median gap to the kerb, share of samples within 0.8 m of it,
     share with a vehicle between it and the kerb, roadway on the rider's side, ground under it that is not roadway).
     After version 4 most remaining errors near the rider were stationary vehicles that are not parked - queued in
     the lane beside the rider or waiting on a side street - which motion alone cannot tell from parked cars.
   - *Time in the rider's path (version 6).* The share of a track with its near edge within 0.3 m of the rider's
     heading and less than 10 m ahead, and that share while the rider is stopped - queued traffic sits there, parked
     cars do not. A looser definition (box centre within 1.2 m, up to 20 m) also caught parked cars on narrow or
     curving streets. The relation model does not use it yet (with 13 near in-my-lane labels it lowered near macro-F1
     from 0.64 to 0.59); the review queue does, to find likely errors.
   - *The vehicle's own motion while the rider is stopped (version 7).* A car queued at a red light and a parked car
     look the same while the rider rolls toward them - both close at the rider's speed. While the rider waits the
     camera is still, so image motion is the vehicle's own: sideways from the body flow against the ground beside it,
     along the road from the body's expansion times its distance (median over 5 samples, so single noisy flow readings
     do not count). Features: share of the track seen while the rider is stopped, the median and 90th percentile of
     that own speed, and the share of it above 1 m/s - queued, crossing and turning traffic moves off, parked cars do
     not. Over 5 seeds (extra trees, leave-one-video-out, 321 near labels) they raised accuracy from 0.725 to 0.735
     overall and from 0.81 to 0.87 on the 57 vehicles seen mostly while the rider was stopped (crossing-traffic F1
     0.71 to 0.74); `scripts/compare_features.py` repeats the comparison.
   - *What the model uses (version 8).* Once every ride had been through the lane and kerb pass in full - those
     comparisons had lane and kerb features only where the 60 s test windows had them - the lane-paint (v3) and kerb
     (v5) features lowered the score instead of raising it, and the model now uses geometry, flow and the stopped-rider
     motion only. Over 5 seeds (321 near labels): v8 accuracy 0.730 and macro-F1 0.627 (v7 0.725 / 0.607), 85% right on
     the vehicles seen while the rider was stopped (v7 85%, v2+flow alone 82%), 39.6 moving vehicles called parked
     (v7 43.0). The lane and kerb features are still computed and kept in `tracks_typed.csv`.
   - *Not a vehicle.* The CLIP crop embeddings are also scored zero-shot against delivery robots, scooters and
     bicycles, pedestrians, bins, dumpsters and benches, which the COCO detector boxes as cars or motorcycles
     (`not_vehicle`). By day it separates them; at night, on a single dark crop, real vehicles score high too (about
     half of the labelled tracks above 0.5 were vehicles), so it flags tracks for review and is reported in
     `vehicles.csv`, and no detection is dropped on it.
   - Each track is summarised into ~130 features; a readable rule baseline (`rule_relation`) turns them into a relation.
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
     saw: rules, a depth-5 decision tree (learned readable rules), gradient boosting and extra trees.
   - `results/typology/feature_ablation.csv` scores each feature set (v2, + flow, v3, + flow consistency, v5) for each
     model on the same labels - the direct measure of what the new features are worth.
   - *Label audit (October 2026).* Labels whose physics disagreed strongly with them (closing ratio far from the
     label's range over 15+ samples) were checked again on four-frame sheets: 15 of 29 were wrong - mostly cars the
     rider followed for 10-25 s in the lane beside the bike lane, labelled parked, far-side oncoming cars labelled
     parked, and tracks whose box switches between vehicles (now `unclear`). The changes are in `track_labels.csv`
     (`labeler` = `claude-visual-review+audit`, the old label in `notes`). 48 further near-range tracks were reviewed
     the same way (33 usable, mostly oncoming, side-road and same-direction traffic, which were scarce).
   - *Where it stands (October 2026, all six rides through the full lane and sign pass).* Within 15 m (321 labels)
     the selected model (extra trees, v8 features: geometry, flow and the vehicle's own motion while the rider is
     stopped) is right on 73.0% (mean of 5 seeds; macro-F1 0.63, against 0.51 for the pipeline before the label
     audit); by closest approach 77% within 5 m, 75% at 5-10 m, 54% at 10-15 m (`relation_accuracy_by_distance.csv`).
     Per class F1: parked 0.80, side road 0.72, other lane 0.66, oncoming 0.60, in my lane 0.42 - the last from only
     13 near labels. Of the 678 counted vehicles on the six rides seen mostly while the rider waited, it calls 3.7%
     parked (70% of all counted vehicles are parked) and 91% side road or crossing; on such labelled vehicles it is
     right 85% of the time. With the lane and kerb pass over whole rides, the lane-paint and kerb features lowered the
     score (5 seeds, near macro-F1: v2+flow 0.631, v3 0.612, v5 0.600, v7 0.607; `feature_comparison_5_seeds.csv`) -
     the same noisy paint that calls shared roads bike lanes, below - so v8 leaves them out; they stay in
     `tracks_typed.csv`. The learning curve (`learning_curve.png`: macro-F1 0.54, 0.57, 0.60, 0.65 at 67, 133, 200,
     267 training labels) is still rising, so reviewed labels (step 9) are the most direct way to improve it. A veto
     of "parked" on the flow consistency shares flagged as many truly parked cars as moving ones and was not adopted.
   - The riding context is scored for the CLIP probe alone and for the probe with the lane-paint detector deciding
     bike lane vs shared road. With the full-ride lane pass, on 307 labelled frames the probe alone reached accuracy
     0.886 (macro-F1 0.754; bike lane F1 0.49) and the paint-decided context 0.50: the paint called 136 of 232
     shared-road frames bike lane, mostly from the solid line of a parking lane or the edge line of a wide kerb lane,
     which is how a bike-lane line looks too. Restricting it to the specific cues (stencil, green paint, the
     segmenter's bike-lane surface), letting it only promote, or a logistic regression on the probe's probabilities
     plus the paint score (leave-one-ride-out) all stayed below the probe alone (0.81-0.82), so the reported context
     is the probe's and the paint call is reported beside it.
7. **Licence plates** - `bikesafe.plates`
   - A small plate detector (YOLOv9-tiny ONNX from open-image-models, 7 MB, downloaded on first use) runs on vehicle
     boxes >= 60 px tall, cropped from the full-resolution frame, and a small OCR model (fast-plate-ocr, global model
     including US plates) reads each plate. Each track is read on up to 10 frames - the largest view of the vehicle in
     each tenth of the track - and a plate counts only if it sits in the lower middle of its own vehicle's box (the
     neighbour's plate often shows in a crop).
   - The consensus text per track is the text read most often, or, when every read differs, a character-by-character
     vote of the reads of the most common length; a single read stands only at >= 0.92 mean character confidence.
     `plates.parquet` keeps every read; each vehicle gets the text most of its tracks' agreeing reads support.
   - The plates also audit the vehicle ids (`identity_audit.json`): tracks with one plate split over several vehicle
     ids (missed joins, or a vehicle seen again later) and vehicle ids carrying two plates (wrong joins).
   - Nothing is blurred: the overlay boxes each plate it can read and writes the text under it.
8. **Outputs** - `bikesafe.exposure`, `bikesafe.render` - every table and column is described in `docs/OUTPUTS.md`
   - `results/corpus/<video>/vehicles.csv`: one row per physical vehicle - its relation in words, the model's
     confidence, a `check` column naming what a person should look at, when (local time) and how close it came, on
     which side, its plate, and whether it counts in the exposure figures. This is the table to read first.
   - `results/corpus/<video>/tracks_typed.csv` (every track with all features, and each vehicle's `plate`),
     `timeline_1s.csv` (with local timestamps for a Strava join, bike lane, signal state, `signs_in_view` and
     `speed_limit_posted` per second; a posted limit holds until the next speed-limit sign, at most 3 minutes),
     `events.csv` (now including `stop_sign` and `red_light_wait`), `signs.csv` (one row per distinct sign / signal
     head, with its type and text), and `results/corpus/corpus_summary.csv` (with `signs_*` counts per type and the
     median posted speed limit). The summary adds `events_stop_sign`, `events_red_light_wait`, `red_light_wait_seconds`
     and `seconds_with_signal_ahead`.
   - Counts are per physical vehicle (version 4): each vehicle gets one relation, the detection-weighted vote of its
     tracks' class probabilities; `per_min_*`, `vehicles_*`, the per-second counts and the events use vehicles, and
     each event is reported once per vehicle. `tracks_*` keeps the per-track count of earlier runs for comparison, and
     `duplicate_links` / `break_links` say how many joins were made.
   - The riding context is CLIP's (the reviewed-frame probe). The lane-paint detector's bike-lane call is reported
     beside it (`in_bike_lane`, `bike_lane_score`) and enters the relation features, but does not override the
     context: it finds most painted bike lanes and also calls most shared roads with an edge line bike lanes (step 6).
   - Colour-coded overlay clips (`bikesafe.render`; the demos are on the
     [project page](https://robertn02.github.io/Car-detection/)), with lane lines, signals (by state), signs with their
     type and text, one box per vehicle, and each licence plate boxed and read out. A clip has every frame of the
     video at its full size: vehicles are detected on every second frame, so their boxes (and plates, which move with
     their vehicle's box) are interpolated on the frames in between. Lane lines and sign boxes are measured twice a
     second; in between they are interpolated, and a line is drawn only where it was found on two of three
     consecutive samples, no further ahead than its paint was seen, not across a crosswalk, and behind the vehicles.
     Clips are H.264 (libx264, CRF 20, from the ffmpeg bundled with `imageio-ffmpeg`), so they play in a browser and
     in slides; the web demos are CRF 23 capped at 10 Mbit/s so they stream (`scripts/build_site.py`).
9. **Human review loop** - `bikesafe.review`
   - `python -m bikesafe.review` serves a review page on the local machine (http://127.0.0.1:8765). It is the
     annotation framework of the pipeline: the models' answers are the starting point, a person confirms or
     corrects them, and the corrections are what the models learn from next.
   - The project page has a copy that needs no server (`bikesafe.review --export`): the first 20 items of each queue
     with their clips and crops, as they were when it was made. Decisions made there stay in the viewer's browser
     until downloaded from its Progress tab; `bikesafe.review --import decisions.json` adds them to the labels.
   - *Vehicles.* Each vehicle is shown as a looping clip of its track at the video's frame rate (its box magenta, the
     others grey; tracks over 8 s play faster) with a zoom,
     the model's probabilities and the rule baseline's answer. Keys 1-5 pick the relation, Enter accepts the model, U
     marks it unclear, N not a vehicle, Z undoes. The "new" queue ranks unlabelled vehicles within 15 m by how likely
     the model is wrong - low probability, and conflicts with the physics (called parked while in the rider's path, or
     while the flow says it moves with or toward the rider; called oncoming while its rear is seen; rules disagree) -
     and spreads them over rides and time, so one traffic jam does not fill the queue (active learning: each review is
     spent where the model is weakest). The "verify" queue shows the existing labels, the ones the model disputes
     first; most of them came from AI-assisted review and are best checked by a person this way.
   - *Signs* (largest view and its context, type and text) and *plates* (crop and reading) have their own queues.
   - Decisions go to `data/typology/track_labels.csv` (with anchors, so they survive a perception re-run),
     `sign_labels.csv` and `plate_labels.csv`, and every decision with the model's answer to `review_log.csv`. The
     Progress tab turns the log into how often the reviewer kept each model's answer and the confusion between model
     and reviewer - a running measure of the models on exactly the cases people looked at (biased towards hard cases
     by the ranking, so it is a floor, not an overall accuracy).
   - "Retrain" runs `bikesafe.train` (relation model; and, from 20 reviewed signs on, a sign-type probe: logistic
     regression on the CLIP image features, blended with zero-shot CLIP at the weight that typed the reviewed signs
     best leave-one-ride-out - used only when that beats zero-shot CLIP on held-out rides, so it needs reviewed signs
     from at least two rides), types the signs again with the probe keeping the text already read
     (`bikesafe.infrastructure --read-only --keep-text`, from the saved CLIP features), and rebuilds the per-ride
     outputs, so the next queue comes from the corrected models. `bikesafe.run` does the same re-typing for any ride
     whose signs were typed with other settings or another probe (`sign_typing` in `infrastructure.json`).
   - Why this and not RLHF: RLHF fits a reward model to human preferences and then optimises a generative model
     against it with reinforcement learning. Nothing here generates; every model is a classifier, and a person's
     correction is a direct label, which supervised retraining uses without the reward-model and RL steps (and their
     instability). The part of RLHF that carries over - people judging the model's own outputs, in a loop - is what
     this tool does.

## Running at scale on a GPU workstation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m bikesafe.run D:\rides --out-root D:\rides_out --render-minutes 1 --hw-decode
python -m bikesafe.train --analysis D:\rides_out\analysis --perception D:\rides_out\perception
python -m bikesafe.exposure --analysis D:\rides_out\analysis --perception D:\rides_out\perception --out results\corpus
```

Every stage is resumable, and existing perception, ego-motion and depth outputs are reused: after updating the code,
the same `bikesafe.run` command re-runs the infrastructure pass where it predates the segmenter, reads the plates
where they were stored as hashes, and recomputes the track features. The first run downloads `yoloe-26l-seg.pt`
(79 MB) into `models/` and, into the Hugging Face cache (`HF_HOME`), the Vistas segmenter (0.8 GB), CLIP ViT-L/14
(1.6 GB) and Florence-2-large (1.5 GB); `python -m bikesafe.run --download-models` fetches everything in advance. The
infrastructure pass loads the segmenter and YOLOE together, saves what it measured, and only then loads the two
readers one after the other; it ran on a 2 GB laptop GPU (GeForce MX250, 8-9 s per sample, almost all of it the
segmenter: about 30 h for the 112 minutes of the six rides, against minutes on a data-centre GPU). Measured there,
FP16 is no faster than FP32 for the segmenter (9.3 vs 11.2 s per frame; Pascal GPUs have no fast half precision) but
halves its memory (1.1 vs 2.1 GB), so FP16 stays the default. The pass saves its progress every 100 samples
(`infrastructure.partial.pkl`) and, started again with the same settings, resumes there; on a test window the
resumed output was identical to an uninterrupted one. `infrastructure.json` records what part of the ride a pass
covered, so a test window (`--start-s` / `--duration-s`) is never taken for a finished ride. If the reading fails,
`python -m bikesafe.infrastructure <perception dir> --read-only` reads the signs again from the saved crops
(`sign_crops.parquet`) without the detectors. Hugging Face models are kept in `_cache/hf` in the project unless
`HF_HOME` says otherwise. At the end `bikesafe.run` prints how long each stage took. A pass whose video ends before
the ride does (a truncated or still-uploading file) stops with an error and keeps its checkpoint instead of writing a
partial ride as finished.

Without a fast GPU at hand there are two cloud routes, both through Google Colab and Google Drive:

- `scripts/colab_bundle.py` packs the inputs of the slow stage (the infrastructure pass) for chosen rides into an
  upload folder, `scripts/colab_infrastructure.ipynb` runs it on a Colab GPU, and `colab_bundle.py --fetch` copies the
  results back into `work/analysis/` (newer files only, so it never undoes later work here). On an L4 a 20-minute ride
  took 22 minutes (0.35 s per sample, against 6.5 s on the MX250); the notebook waits for each video to arrive in Drive
  at its full size before starting, and also re-types signs whose typing settings changed.
- `scripts/colab_try_your_video.ipynb` (the README's "Open in Colab" button) runs every stage on one video with the
  trained models and shows the overlay clip and tables in the page - for someone without the project installed. It
  installs the pinned versions with `uv` (under a minute) and writes everything to the user's Drive as it goes, so a
  dropped session resumes.

Colab sessions end when idle (an L4 waiting for an upload was disconnected), so long whole-corpus runs belong on a
workstation or USC CARC (`scripts/carc_run.slurm`).

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
- Sign types come from zero-shot CLIP unless the sign's text decides them; small or blurred signs (under ~28 px) are
  typed from appearance only, and street-name signs are the most frequent confusion. The posted speed limit is the
  last speed-limit sign read and is not map-matched, so it can outlive a turn onto another street (it expires after
  3 minutes).
- The labels were produced by AI-assisted visual review (including the October audit and batch) and should be
  spot-checked by a human reviewer (`labeler` column); the audit found about half of the physics-contradicted labels
  wrong, so the rest also carry some noise. The pipeline retrains from the CSVs.
- The overlay labels a vehicle only within 25 m (relations are validated within 15 m) and adds "?" where the model
  gives its relation less than 45% probability.
- Vehicle ids join a vehicle only across gaps of up to 2 s. A vehicle that leaves the view and returns later (for
  example leapfrogging between traffic lights) is a new vehicle; the opt-in plate audit counts such returns.
- The approved 300-frame reference sequence labels the second (truck) box of a doubly detected pickup as eleven
  separate short identities, so identity scores against it penalise correct duplicate merging; see the upgrade report.
- Plates are read only on vehicles the detector boxed at >= 60 px tall; front plates are missing on many California
  cars, and crossing traffic is mostly seen side-on, so most vehicles get no plate. On ride 006 (21 min) a plate was
  found on 1,377 of the 2,371 tracks checked and read consistently on 295 (278 vehicles); the identity audit then
  flagged 16 pairs of tracks with one plate less than 2 s apart that `bikesafe.stitch` had not joined. Over the
  daytime rides 12-16% of tracks get a reading; on the two evening rides (002, 004) under 1% do: lit plates are
  washed out by their lamps or smeared by motion, and the largest "plates" found at night are building fronts and
  shop signs, which yield no text and are not drawn.
- The tracker occasionally hands one id to a different vehicle, mostly after losing it for a frame or more at a
  junction; the whole track then carries one relation. `bikesafe.identity` cuts tracks where appearance (colour
  histograms) and motion both jump: on the reference sequence 5 of its 6 cuts were real and IDF1 rose from 69.6% to
  71.9%, but on the rides only about 60% of its cuts were real (small, distant and night-time boxes) and relation
  accuracy fell from 0.735 to 0.718, so it is off by default (`bikesafe.tracks --identity-split`). A vehicle
  re-identification network is the next step for this.
