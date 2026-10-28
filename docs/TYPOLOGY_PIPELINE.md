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
       CLIP gives its best type less than half of the probability it gives the sign types (not-a-sign, stop and yield
       set aside; 0.6 for speed limit); stop and yield signs need their word (or, for stop, the COCO check). On a 60 s window of ride 006 most wrong types were CLIP guesses on
       small, blurred or edge-on signs, and red pole markers taken for yield signs. On evening ride 002, CLIP rejected
       or typed dark blobs with high confidence: the contrast gate kept 583 of its 1,079 rejected signs as signs and
       untyped 90 of its 312 types, mostly wrong ones, while on day ride 006 it changes 8 of 619 types;
     - a sign is a *stop* sign when its text reads STOP (not "STOP AHEAD", "NO STOP", "STOP HERE ON RED" or "STOP FOR
       PEDESTRIANS"), or when the COCO-trained YOLO11m sees a stop sign on it and its text names no other type (a
       "DO NOT ENTER" or a parking-hours plate read on a red sign is that sign); "SPEED HUMP" is a warning; signs CLIP
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
   - *Metric scale, against GPS (version 8).* The focal length comes from an assumed 110 degree field of view, and
     distances ahead and speeds scale with it. On the five rides with a Strava track, `python -m bikesafe.gps
     --calibrate` measures how far off they are, two ways: parked cars stand still, so the distance to them shrinks
     at the rider's GPS speed (distances ahead were 1.53-1.67 times too short: the true field of view
     is nearer 80 degrees), and the rider's speed from the video against the GPS speed, which also needs the flow's
     forward rate raised by 0.95-1.06 by day and up to 1.22 at night. A ride with
     few parked cars passed (at night) is pulled towards the pooled distance factor. The factors go to
     `results/corpus/metric_calibration.json` and `bikesafe.tracks` applies them: distances ahead, speeds and the
     rider's turn angles are in metres (a ride without GPS, or a new video, takes the median, x1.58).
     Lateral distances come from car widths and were already in metres. The bird's-eye lane view keeps the assumed
     focal length it was measured with. Thresholds set on the earlier scale (the 15 m where relations were checked,
     the 8 m an overtake comes within, the stopped and static speeds) are the same ones in true metres (x 1.6:
     25 m, 13 m, ...).
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
     that own speed, and the share of it above 1.6 m/s - queued, crossing and turning traffic moves off, parked cars do
     not. Over 5 seeds (extra trees, leave-one-video-out, 338 labels within 25 m) they raise accuracy from 0.710 (v5)
     to 0.715 overall and from 0.82 to 0.85 on the 62 vehicles seen mostly while the rider was stopped
     (crossing-traffic F1 0.71 to 0.73; `feature_comparison_5_seeds.csv`, which `scripts/compare_features.py` writes).
   - *What the model uses (version 8).* Once every ride had been through the lane and kerb pass in full - those
     comparisons had lane and kerb features only where the 60 s test windows had them - the lane-paint (v3) and kerb
     (v5) features lowered the score instead of raising it, and the model now uses geometry, flow and the stopped-rider
     motion only. Over 5 seeds (338 labels within 25 m): v8 accuracy 0.727 and macro-F1 0.640 (v7 0.715 / 0.628), 85%
     right on the vehicles seen while the rider was stopped (v7 85%, v2+flow alone 82%), 46.2 moving vehicles called
     parked (v7 47.8). The lane and kerb features are still computed and kept in `tracks_typed.csv`.
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
   - *Where it stands (October 2026, all six rides through the full lane and sign pass, distances calibrated against
     GPS).* The deployed model (extra trees, v8 features: geometry, flow and the vehicle's own motion while the rider
     is stopped, trained on all 434 labelled tracks) is right on 71.4% of the 357 labelled vehicles within 25 m, in
     leave-one-video-out validation (macro-F1 0.60, against 0.51 for the pipeline before the label audit); by closest
     approach 77% within 8 m, 75% at 8-16 m, 55% at 16-25 m (`relation_lovo_predictions.csv`,
     `relation_accuracy_by_distance.csv`). Per class F1 within 25 m: parked 0.79, side road 0.73, other lane 0.58,
     oncoming 0.53, in my lane 0.38 - the last from only 15 near labels. The first human review round (October 28: 20
     vehicles from the queue of new, low-confidence ones, 19 of them relations) added harder cases: before it the
     model scored 72.5% of 338, and before the GPS calibration 71.3% on the matching range (then "15 m"). The model trained on the near labels alone scores 72.3% (macro-F1 0.62; `relation_within_near_z`
     in `evaluation_summary.json`); it is not the deployed one. The model is chosen on the same held-out predictions
     it is scored on, so these figures are slightly optimistic. Of the 708 counted vehicles on the six rides seen
     mostly while the rider waited, it calls 4.1% parked (69% of all counted vehicles are parked) and 94% side road
     or crossing; of the 57 such labelled vehicles within 25 m it gets 91% right. With the lane and kerb pass over whole rides, the lane-paint and kerb features lowered the
     score (5 seeds, near macro-F1: v2+flow 0.624, v3 0.614, v5 0.611, v7 0.628; `feature_comparison_5_seeds.csv`) -
     the same noisy paint that calls shared roads bike lanes, below - so v8 leaves them out; they stay in
     `tracks_typed.csv`. The learning curve (`learning_curve.png`, on the features before the calibration: macro-F1
     0.54, 0.57, 0.60, 0.65 at 67, 133, 200, 267 training labels) is still rising, so reviewed labels (step 9) are the most direct way to improve it. A veto
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
     each event is reported once per vehicle.
   - Overtakes are found from geometry, not from the relation: a vehicle followed for at least a second that comes
     within 13 m and pulls away beside the rider, its side gap above 0 and up to 3.5 m, at most 24 m/s faster. Without
     those limits (before October 2026) half the overtakes and close passes were sub-second fragments of parked cars
     passed close by, whose box seemed to pull away, and a few were mismeasured at up to 130 m/s: 178 overtakes and
     67 close passes on the six rides, now 45 and 25. Stop-sign objects less than 3 s apart are one sign the object
     linker split, and the summary's sign counts take signs seen on at least two samples (two in three seen on one
     sample only are such pieces). `tracks_*` keeps the per-track count of earlier runs for comparison, and
     `duplicate_links` / `break_links` say how many joins were made.
   - The riding context is CLIP's (the reviewed-frame probe). The lane-paint detector's bike-lane call is reported
     beside it (`in_bike_lane`, `bike_lane_score`) and enters the relation features, but does not override the
     context: it finds most painted bike lanes and also calls most shared roads with an edge line bike lanes (step 6).
   - Colour-coded overlay clips (`bikesafe.render`; the demos are on the
     [project page](https://robertn02.github.io/Car-detection/)), with lane lines, signals (by state), signs with their
     type and text, one box per vehicle, and each licence plate boxed and read out. A clip has every frame of the
     video at its full size: vehicles are detected on every second frame, so their boxes (and plates, which move with
     their vehicle's box) are interpolated on the frames in between. Lane lines and sign boxes are measured twice a
     second. A line found on two of three consecutive samples is predicted for every frame from the samples around
     it and then laid on that frame's paint: the handlebar camera rolls and pitches between samples, so lines drawn
     from the samples alone sat beside their paint, and seams the segmenter took for paint were drawn as lines. Along
     the line each frame is searched for a bright stripe as wide as lane paint at that distance (a kerb, a white car
     or a sunlit patch is wider), and a robust straight-line fit through the stripes of the last six frames moves the
     line part of the way there: the line stays put in the picture while the dashes of a broken line slide along it,
     so a few frames together cover it where one frame finds a dash or two. A line keeps one identity by where it is
     in the picture, not by the samples' numbering (which restarts whenever a sample misses it); it appears once paint
     is found under it in most recent frames - along at least a quarter of the rows that can be seen, so a seam or a
     faded edge with a stripe here and there does not count - and goes only when that falls well below, or a while
     after the samples stop reporting it, not while a vehicle hides its paint. Its ends ease towards the samples'
     extent instead of jumping with it. Only the rider's own lane is drawn: the nearest line on each side, by where
     the line meets the rider (its slope in the picture, which does not waver with the heading estimate), skipping
     paint inside the lane - an arrow's shaft, a bike symbol, a seam - that lies less than a lane's width (1.3 m)
     from the line opposite. A farther line does not stand in while the nearer one is faint, so three or four lines
     are never shown. No line is drawn through a junction (a crosswalk across the rider's path, or six or more short
     lines at once - crosswalk bars; a buffered bike lane's long double lines are not a junction), in a turn (the
     heading turning faster than 12 degrees a second) or where no sample reports a line beside the rider; lines come
     back only once the lanes have been steady for 1.5 s, and fade in. A line within 0.5 m of the rider's path is not
     a side either: what lies there is mostly the glare of the bike's own light at night, an arrow or a seam. On four
     demo windows lines appear or go 17-33 times a minute (40-78 before), most of them now these deliberate hides,
     and never three at once. Lines are drawn as strips of paint in perspective, down to the bottom of the picture
     where the paint continues, behind the vehicles. Box labels are short ("same way 9m"), placed nearest vehicle first so none covers another (a far vehicle with no room keeps its box but no label), and the
     banner shows the GPS speed where the ride has a track and the riding context most common over 7 s.
     Clips are H.264 (libx264, CRF 20, from the ffmpeg bundled with `imageio-ffmpeg`), so they play in a browser and
     in slides; the web demos are 2 minutes at CRF 23 capped at 5.5 Mbit/s, so each streams and stays under GitHub's
     100 MB a file (`scripts/build_site.py`).
9. **Human review loop** - `bikesafe.review`
   - `python -m bikesafe.review` serves a review page on the local machine (http://127.0.0.1:8765). It is the
     annotation framework of the pipeline: the models' answers are the starting point, a person confirms or
     corrects them, and the corrections are what the models learn from next.
   - The project page has a copy that needs no server (`bikesafe.review --export`): the first 20 items of each queue
     with their clips and crops, as they were when it was made, leaving out anything filmed in a ride's privacy zones
     (on a ride without GPS, its first and last 3 minutes), since the copy is public. Decisions made there stay in the
     viewer's browser until downloaded (header or Progress tab, as `decisions_<name>_<date>.json`); `bikesafe.review
     --import decisions.json` adds them to the labels, once: importing the same file again adds nothing.
   - The page asks for the reviewer's name before the first decision; keys act only outside text fields; in verify
     mode Enter keeps the existing label (the model's answer is shown beside it); Undo takes back only the decision
     on screen; class names are shown in words with what each means.
   - In the Colab notebook the page saves decisions in the run's Drive folder (`--labels`) and offers no Retrain:
     the models learn from the labelled rides, which are on the machine that ran the corpus. There,
     `bikesafe.review --import "G:/My Drive/bikesafe_runs/<video>"` adds the run's ride (perception, analysis,
     results) and its decisions (cards renumbered, since each run numbers its own), and Retrain learns from them.
   - *Vehicles.* Each vehicle is shown as a looping clip of its track at the video's frame rate (its box magenta, the
     others grey; tracks over 8 s play faster) with a zoom,
     the model's probabilities and the rule baseline's answer. Keys 1-5 pick the relation, Enter accepts the model, U
     marks it unclear, N not a vehicle, Z undoes. The "new" queue ranks unlabelled vehicles within 25 m by how likely
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

10. **Location** - `bikesafe.gps`
   - `python -m bikesafe.gps --strava <export folder>` matches each video to the Strava activity recorded while it
     was filmed and writes where the rider was in each second (`gps_1s.csv`). The camera's clock is not the phone's:
     the offset is the lag (within 3 minutes) at which the bike speed measured from the video best correlates with the
     GPS speed. Positions and speeds near where a track starts or ends are left out (privacy zones, as on Strava),
     since the tables and the project page are public. Each zone's radius is 400-800 m, drawn from the track itself:
     with one radius for all, the shown routes' ends would lie on circles of known size round home and work.
   - The project page uses it twice: a map beside each demo whose dot follows the video, and an exposure map of all
     rides - the per-second counts (moving traffic, vehicles in the rider's lane, parked cars in the door zone) as a
     heatmap, the riding context as coloured lines, and close passes and overtakes as points. The heat is drawn like
     Strava's, as thin unblurred lines, each stretch coloured by the counts of all the seconds within about 30 m of it;
     it is time-weighted: a minute waiting at a red light beside traffic counts sixty times.

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

- Distances ahead and speeds are calibrated against GPS (step 5) per ride; the ride without a Strava track (005)
  takes the median factor. Lateral distances, and with them the 1.5 m close-pass gap, come from the width of cars
  seen square-on (1.95 m with mirrors) and are not checked against GPS, which cannot measure them; the lateral scale
  also varies between rides (`lateral_height_m` 1.06-1.47 in the calibrations).
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
- The labels were produced by AI-assisted visual review (482 of the 517 rows of `track_labels.csv`, including the
  October audit and batch; 20 more were reviewed by a person on the review page on October 28) and should be
  spot-checked by a human reviewer (`labeler` column; the review page's verify queue shows them); the audit found
  about half of the physics-contradicted labels
  wrong, so the rest also carry some noise. The pipeline retrains from the CSVs.
- The overlay labels a vehicle only within 40 m (relations are validated within 25 m) and adds "?" where the model
  gives its relation less than 45% probability.
- Vehicle ids join a vehicle only across gaps of up to 2 s. A vehicle that leaves the view and returns later (for
  example leapfrogging between traffic lights) is a new vehicle; the opt-in plate audit counts such returns.
- The approved 300-frame reference sequence labels the second (truck) box of a doubly detected pickup as eleven
  separate short identities, so identity scores against it penalise correct duplicate merging; see the upgrade report.
- Plates are read only on vehicles the detector boxed at >= 60 px tall; front plates are missing on many California
  cars, and crossing traffic is mostly seen side-on, so most vehicles get no plate. On ride 006 (21 min) a plate was
  found on 1,377 of the 2,371 tracks checked and read consistently on 295 (278 vehicles); the identity audit then
  flagged 16 pairs of tracks with one plate less than 2 s apart that `bikesafe.stitch` had not joined. Over the
  daytime rides 12-16% of the tracks checked (8-10% of all tracks) get a reading; on the two evening rides (002, 004)
  under 1% do: lit plates are
  washed out by their lamps or smeared by motion, and the largest "plates" found at night are building fronts and
  shop signs, which yield no text and are not drawn.
- The tracker occasionally hands one id to a different vehicle, mostly after losing it for a frame or more at a
  junction; the whole track then carries one relation. `bikesafe.identity` cuts tracks where appearance (colour
  histograms) and motion both jump: on the reference sequence 5 of its 6 cuts were real and IDF1 rose from 69.6% to
  71.9%, but on the rides only about 60% of its cuts were real (small, distant and night-time boxes) and relation
  accuracy fell from 0.735 to 0.718, so it is off by default (`bikesafe.tracks --identity-split`). A vehicle
  re-identification network is the next step for this.
