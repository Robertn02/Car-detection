# Distinguishing how each vehicle relates to the rider

Response to the feedback: *"it will be important to distinguish between cars in my lane of travel, cars sharing the
road (in other lanes), parked cars, and cars on side roads or going the opposite way on the other side of the median.
Would you be able to distinguish these?"*

**Yes, and it runs end to end over all six rides.** Every vehicle track gets a relation to the rider, every second gets
a riding context (separated path / bike lane / shared road), and the two combine into exposure measures per ride. All
numbers below come from rides the model never saw during training (leave-one-video-out).

The first version of this pipeline was accurate enough to be useful only for parked cars. A diagnosis of *why* led to a
second version, and section 4 reports both, like for like.

## 1. The typology

| Relation | Your grouping | Definition |
|---|---|---|
| `ego_lane` | in my lane | In the lane or path the rider occupies, including a vehicle queued ahead and one intruding into the bike lane |
| `adjacent_same` | sharing the road | Same direction, different lane: overtaking traffic, and the travel lane beside a bike lane |
| `oncoming` | sharing the road | Opposite direction on the same roadway, no physical median |
| `parked` | parked | At a curb, in a parking lane, driveway or lot. A vehicle stopped in a traffic lane at a light is **not** parked |
| `cross_side` | not sharing | Side streets, crossing or turning traffic, driveways, and anything beyond a median or on another roadway |

A sixth class ("across a median / other roadway") was folded into `cross_side`: in 434 reviewed tracks it was almost
never separable from other not-sharing traffic, and both map to the same reporting group.

## 2. The corpus

| Ride | Start (local) | Minutes | Processed frames | Detections | Tracks |
|---|---|---:|---:|---:|---:|
| 001 | 2026-02-09 09:01 | 14.4 | 12,967 | 107,915 | 2,390 |
| 002 | 2026-02-09 18:33 | 20.4 | 14,719 | 56,109 | 1,730 |
| 003 | 2026-02-23 09:02 | 16.7 | 11,989 | 95,392 | 2,353 |
| 004 | 2026-02-23 18:35 | 22.3 | 16,077 | 62,870 | 1,969 |
| 005 | 2026-02-24 09:11 | 17.4 | 12,534 | 94,274 | 2,611 |
| 006 | 2026-02-24 16:28 | 20.9 | 15,039 | 136,441 | 3,717 |
| **All** | | **112.1** | **83,325** | **553,001** | **14,770** |

Two rides are after dark, which halves the detection count and is the weakest part of the pipeline throughout.

## 3. How a relation is decided

No extra sensors, no manual calibration — the camera calibrates itself from the footage.

1. **Horizon and camera height.** Cars are all about the same height, so across thousands of boxes the wheel row grows
   linearly with box height: the intercept is the horizon, the slope the camera height (1.14–1.21 m, consistent with a
   handlebar mount). Box widths of cars seen square-on give the lateral scale, within 4% of the height-derived value.
2. **Rider motion.** Road features 5–12 m ahead are tracked with pyramidal Lucas-Kanade; a flat-road model gives
   forward speed, yaw and heading each frame (median 3.1–5.2 m/s, cross-checked against the rate lane dashes pass).
3. **World trajectory of each vehicle — the decisive cue.** Subtracting how far the rider advanced and turned since a
   track began leaves the vehicle's own motion in the world. A least-squares fit over the whole track gives its speed
   and heading: 0° travels with the rider, 180° is oncoming, 90° crosses. On labelled tracks these land exactly where
   they should:

   | Labelled as | median world speed | heading (quartiles) |
   |---|---:|---|
   | parked | 1.8 m/s | — (speed decides) |
   | adjacent, same way | 2.5 m/s | 9–56° |
   | in my lane | 5.4 m/s | 13° |
   | crossing / side road | 6.1 m/s | 77–110° |
   | oncoming | 5.1 m/s | 139–161° |

4. **Lane structure.** White and yellow markings are detected in the image and projected into a bird's-eye grid,
   accumulated over a second so dashed lines are not missed. Each vehicle then gets *how many lane lines lie between
   the rider and it*, and whether any is a yellow centre line — the semantic the first version lacked, since a human
   decides "my lane or the next one" from the paint, not from a distance in metres.
5. **Neighbourhood.** How many stationary vehicles sit at the same lateral offset at the same time: parked cars come in
   rows, a car stopped in a traffic lane usually does not.
6. **Appearance.** CLIP scores each crop for the face visible (rear / front / side) and for emergency-vehicle likeness.

### A negative result worth recording

Learned metric depth (Depth Anything V2, metric outdoor) was added expecting it to fix distance estimation past 10 m.
Measured against car geometry it is **worse than the simple geometry** on this footage: it compresses range (its depth
is 1.26× the true distance at 5 m but 0.64× at 30 m) and, as the rider advances one metre, its depth to a parked car
changes by only 0.61 m, against 0.92 m for the wheel-row/box-height estimate. The depth model is trained on car-dashcam
intrinsics and does not transfer to a wide-angle bike camera. It is therefore **not** used for vehicle distance; it is
kept only to place lane paint in the bird's-eye grid, where only relative geometry matters.

## 4. Accuracy, version 1 vs version 2 (leave-one-video-out, 378 reviewed tracks)

Within 15 m — the range where relations are validated and where perceived safety is decided:

| Model | v1 accuracy | v2 accuracy | v1 macro F1 | v2 macro F1 | v1 4-group | v2 4-group |
|---|---:|---:|---:|---:|---:|---:|
| Readable rules | 41.0% | 39.3% | 0.343 | 0.295 | 43.1% | 43.4% |
| Decision tree (depth 5) | 48.1% | **57.3%** | 0.382 | **0.453** | 49.8% | **59.0%** |
| **Gradient boosting** | 65.8% | **68.1%** | 0.415 | **0.503** | 67.1% | **68.8%** |

Per-class F1 for gradient boosting, v1 → v2: in my lane **0.13 → 0.38**, other lane same way **0.20 → 0.42**,
parked 0.79 → 0.80, side road 0.63 → 0.64, oncoming 0.33 → 0.28. The two safety-critical classes that the first
version essentially could not find are the ones that improved most.

Over all labelled tracks including distant ones: 62.2% → 63.0% accuracy, 4-group 63.5% → 64.0%.

**Where the gain comes from.** With a matched 17-feature set, removing the new world-heading, lane and neighbourhood
features drops accuracy from 63.4% to 55.6% and macro F1 from 0.439 to 0.335. Feeding all ~80 columns to gradient
boosting scored no better than the 39 physics-defined features, which is what 378 labels should be expected to support.

**Accuracy still depends on distance:**

| Closest approach | 0–5 m | 5–10 m | 10–15 m | 15–25 m | >25 m |
|---|---:|---:|---:|---:|---:|
| Tracks | 142 | 89 | 64 | 61 | 22 |
| Accuracy | **78%** | 64% | 48% | 49% | 41% |

Every track therefore carries its closest approach, and the exposure measures below count only vehicles that came
within 15 m.

**Riding context**: a linear probe on CLIP embeddings reaches **88.6% accuracy (macro F1 0.754)** across held-out
rides — separated path 0.85, shared road 0.93, bike lane 0.49 (bike lanes are confused with shared road where paint is
faded or the lane is dropped at intersections).

## 5. What the typology shows about these rides

Vehicles per minute that came within 15 m of the rider, weighted across all six rides:

| Riding context | Minutes | In my lane | Other lane, same way | Oncoming | Parked | Side road / other roadway | Seconds within 1.5 m of a parked car |
|---|---:|---:|---:|---:|---:|---:|---:|
| Separated path | 19.8 | 0.91 | 0.15 | 0.00 | 9.5 | 2.9 | 5% |
| Bike lane | 7.3 | 0.27 | 3.30 | 4.67 | 54.9 | 6.4 | 30% |
| Shared road | 85.1 | 1.36 | 2.32 | 2.33 | 50.1 | 12.5 | 37% |

- **The separated-path stretches you added act as a negative control, and road-sharing traffic passes it**: 0.15
  adjacent-lane and 0.00 oncoming vehicles per minute, against 2.3 and 2.3 on shared roads.
- **A bike lane removes vehicles from the rider's lane but not from the rider's side**: in-lane traffic is 5× lower
  than on a shared road (0.27 vs 1.36 per minute), adjacent-lane traffic is the highest of any context, and the rider
  is still within 1.5 m of a parked car 30% of the time. The bike-lane row covers only 7.3 minutes, mostly from ride
  006, so treat it as indicative.
- **Shared roads carry the in-lane exposure** — 1.36 vehicles/min in the rider's own lane and the highest door-zone
  share.

### A label-free way to track progress

Separated paths should contain no traffic sharing the rider's lane, so the in-lane rate there is a false-positive rate
that needs no annotation at all:

| | 001 | 002 (night) | 003 | 004 (night) | 005 | 006 |
|---|---:|---:|---:|---:|---:|---:|
| False in-lane per minute on separated paths | 0.00 | 1.66 | 0.41 | 1.61 | 0.85 | 0.28 |

Daylight rides average 0.39/min, night rides 1.64/min. This single number is the cheapest progress metric for the next
iteration, and it says plainly where the remaining error lives: after dark.

Per-ride events (geometric, independent of the relation label):

| Ride | Overtakes | Close passes < 1.5 m | Median passing clearance | Near crossing traffic | Door-zone seconds |
|---|---:|---:|---:|---:|---:|
| 001 | 23 | 10 | 3.7 m | 46 | 314 |
| 002 (night) | 25 | 10 | 1.9 m | 213 | 334 |
| 003 | 35 | 14 | 3.1 m | 149 | 290 |
| 004 (night) | 23 | 9 | 2.7 m | 316 | 346 |
| 005 | 28 | 9 | 3.4 m | 203 | 298 |
| 006 | 45 | 16 | 2.3 m | 276 | 447 |

## 6. Outputs per ride

In `results/corpus/<ride>/`:

- `tracks_typed.csv` — one row per vehicle: relation, your reporting group, class probabilities, closest approach,
  lateral offset, world speed and heading, lane lines to the rider, riding context.
- `timeline_1s.csv` — per second: local timestamp, riding context, rider speed, vehicles visible by relation, nearest
  in-lane vehicle, passing clearance, door-zone count. The timestamp column is there to join Strava directly.
- `events.csv` — overtakes with clearance, close passes, near crossing traffic, close oncoming, and a ranked
  emergency-vehicle shortlist for review.
- `demos/typology_006_bikelane_440s.mp4`, `demos/typology_005_following_280s.mp4` — colour-coded overlay clips.

## 7. Limitations, stated plainly

- **Distant vehicles remain unreliable** (48% beyond 10 m). Exposure counts are restricted to vehicles within 15 m.
- **The rarest classes have the fewest labels** — 16 in-my-lane and 38 adjacent-lane reviewed tracks. A label learning
  curve shows accuracy flat from 79 to 315 labels, so *more of the same labels* is not the fix; better features were,
  and the next lever is better labels (below), not more.
- **Labels are AI-assisted, not human-verified.** All 434 track cards and 314 scene frames were reviewed visually by
  Claude (`labeler=claude-visual-review` in `data/typology/`). The cards are saved as images, so correcting them is a
  review pass; `python -m bikesafe.train` refreshes every number here.
- **Night rides are the weak point**, by every measure: fewer detections, blurrier crops, and 4× the false in-lane rate.
- **Metric distances assume a 110° field of view.** Relations use ratios and do not depend on it; absolute metres
  should be recalibrated once against Strava speed.
- **Parked-car counts are inflated by track fragmentation**; per-minute parked rates are relative, not a census.
- Flat-road geometry: hills and strong pitch bias distances until the rolling horizon estimate re-settles.

## 8. What I would do next, in order

1. **A human verification pass over the existing cards** (30–60 minutes), concentrating on in-my-lane and
   adjacent-lane. Label *intervals* rather than whole tracks where a relation changes.
2. **Fix the night rides**: a low-light detector pass, or simply weighting daytime footage for training and reporting
   night separately. The separated-path metric above measures progress without new labels.
3. **Run at full frame rate on your GPU** (`--target-fps 24 --gmc sparse`, `--sample-fps 4`): identity quality is
   IDF1 69.6% at 12 fps against 72.5% at full rate on the approved benchmark sequence.
4. **Calibrate scale against Strava**, which also turns the per-second timeline into a map-matched exposure profile.
5. **Then 360 footage.** The same geometry works per view; the missing piece is vehicles *behind* the rider, which is
   where overtakes begin — the forward camera only measures a pass once it is already alongside.

## 9. Reproducing

```bash
python -m bikesafe.run videos --out-root work --render-minutes 1
```

Resumable stages: `bikesafe.perceive` (GPU) → `bikesafe.egomotion` (CPU) → `bikesafe.scene3d` (GPU, depth + lane) →
`bikesafe.tracks` → `bikesafe.exposure` → `bikesafe.render`. Training and evaluation: `python -m bikesafe.train`.

Pipeline documentation: [`docs/TYPOLOGY_PIPELINE.md`](../docs/TYPOLOGY_PIPELINE.md).
Labeling protocol: [`data/typology/LABELING_GUIDE.md`](../data/typology/LABELING_GUIDE.md).
Version 1 metrics are kept in `results/typology/evaluation_summary_v1_flat_road_only.json` for comparison.
