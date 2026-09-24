# Bike lanes, traffic signs and signals, and the accuracy gap

Response to: *"there is progress in the latest work but the accuracy is still not acceptable, and I don't see bike
lane or traffic sign detection included. We need to improve better and faster."*

**Short version.**

| Asked for | What changed | How it was checked |
|---|---|---|
| Bike-lane detection | New: the painted bike lane itself - lane lines and bicycle stencils in a bird's-eye view of the road - decides "bike lane vs shared road" every second | 142 hand-labelled seconds of real footage: bike-lane recall **64% -> 88%**, F1 **0.78 -> 0.93**, 1 false bike-lane second in 100 labelled shared-road seconds |
| Traffic-sign detection | New: traffic signals with their state (red / yellow / green), stop signs (double-checked), other traffic signs; events for stop signs passed and waits at red lights | Signal state right on **40 of 44** real signal-head crops; stop-sign check removed **all 10** false stop signs on the demo clips and kept the real one |
| Accuracy of the typology | Diagnosed where it is lost (moving vehicles called parked) and why; new depth-free motion features from optical flow that was already computed but never used; lane-paint position of each vehicle | On 76 hand-labelled real tracks the new closing ratio separates parked cars from same-direction traffic better than any existing feature (AUC **0.97**; the best existing one, along-road speed, 0.94). The gain on the 378 labelled tracks has to be measured on your machine - one command, below |
| Faster | FP16 on the GPU by default, optional hardware video decoding, CPU stages run for several rides at once, the new stage reuses everything already computed, per-stage timing report | Implemented; GPU timings can only be measured on the GPU machine and are printed by `bikesafe.run` |

What this sandbox could and could not do: the ride videos and the intermediate `work/` files stay on your machine, and
this environment has no GPU and cannot reach HuggingFace. Everything below was therefore checked on the three demo
clips in `demos/` (re-run through the pipeline at their original 12 fps, 3 minutes of real footage) and on the
per-track tables in `results/corpus/`. The demo clips have the old overlay burned in; where that affected a result it
is said so.

## 1. Where the accuracy is lost today

Leave-one-video-out, 295 labelled tracks that came within 15 m (`results/typology`): 68.1% accuracy, macro-F1 0.503.
The confusion matrix says what kind of error dominates:

| labelled as | predicted parked | total |
|---|---:|---:|
| oncoming | **12** | 23 |
| other lane, same way | **9** | 25 |
| side road / crossing | **18** | 68 |
| in my lane | 3 | 11 |

Moving vehicles called parked are the largest single error. Three causes, each with evidence:

1. **The world trajectory drifts for parked cars.** It is fitted to the box centre, which slides outward as a passed
   car's side comes into view. Parked cars end up with a fitted world speed of **2.15 m/s at a heading of 95 degrees**
   (median), overlapping crossing traffic (4.0 m/s, 92 degrees). Splitting the velocity shows the along-road part is
   sound (parked -0.08 m/s, same direction +1.7 to +2.2, oncoming -3.6) and the lateral part carries the artefact.
2. **The depth-model trajectory is biased**: parked cars "move" at +2.1 m/s along the road in it, the range
   compression the previous report measured.
3. **Vehicles alongside lose their geometry.** A box cut off by the frame edge has no usable wheel row, and the
   model falls back to the majority class. The demo overlay shows it: the pickup in the travel lane beside the bike
   lane at 473 s and a car in the travel lane alongside at 488 s are both drawn as "parked 2 m".

Better modelling of the existing columns does not fix this. On the same leave-one-video-out bench (which reproduces the
published numbers exactly):

| model on the existing 42 features | within 15 m: accuracy / macro-F1 | all: accuracy / macro-F1 |
|---|---:|---:|
| gradient boosting (current) | 68.1% / 0.503 | 63.0% / 0.432 |
| extra trees, balanced (mean of 5 seeds) | 68.3% / 0.510 | **65.9% / 0.482** |
| random forest | 66.8% / 0.419 | 65.6% / 0.403 |
| logistic regression | 61.0% / 0.475 | 55.6% / 0.428 |
| + 16 engineered combinations (e.g. world velocity split into along / outward) | within +-2 points | within +-2 points |

With ~300 labelled tracks the standard error of an accuracy is about 2.7 points, so within 15 m none of these
differences is real, and removing whole feature groups moved scores by -2.3 to +1.6 points. The limit is the
information in the features, as the previous report concluded; the next section adds information.

One modelling change was kept: extra trees tie gradient boosting within 15 m and are better over all distances on
every one of 5 seeds (+2.9 accuracy, +5.0 macro-F1 points), so `bikesafe.train` now compares them as a third candidate.
The selection rule is unchanged (best macro-F1 within 15 m); on the version-2 features it still picks gradient
boosting, by 0.0003, so nothing changes until the new features below are in the tables. The refactored training
script reproduces every published number exactly.

## 2. New motion features: what the optical flow already knew

The perception pass has always stored, for every detection, the dense optical flow inside the box and on the ground
just below it. None of it was used. Three depth-free cues come out of it (`bikesafe.geometry.add_flow_motion`,
`bikesafe.tracks.flow_motion_summary`):

- **closing ratio** - how fast the vehicle grows in the image, divided by how fast a static object at the same
  distance would grow: about 1 for a parked car, 0 for a vehicle moving with the rider, 2 for an oncoming one at the
  rider's speed. No absolute scale is involved.
- **lateral flow speed** - the body's horizontal image motion minus the ground's in the same image column, converted to
  m/s with box height (cars are 1.55 m tall). The rider's own motion and yaw cancel exactly, so there is no drift for
  parked cars. (It also contains the vehicle's along-road motion times its bearing; removing that term with the
  measured expansion is exact in theory, but per detection the expansion is too noisy - on real footage the corrected
  version was worse, so the classifier gets the raw quantity alongside the closing ratio.)
- **grid projection** - the body's motion compared with the background beside it; unlike the other two it also works
  for boxes cut off by the frame edge.

Checked on 76 hand-labelled tracks from the demo clips (I labelled 88 real tracks from their start / middle / end
frames; 12 were too ambiguous and are excluded; `results/infrastructure/demo_track_labels.csv`). Separability of parked cars from each kind of moving vehicle (ROC
AUC, 0.5 = useless, 1 = perfect):

| parked vs ... | existing `world_speed` | existing `v_along` | **closing ratio** | **lateral flow** |
|---|---:|---:|---:|---:|
| same direction, rider moving (59 tracks) | 0.924 | 0.944 | **0.969** | 0.782 |
| crossing (62) | **0.947** | 0.532 | 0.520 | 0.872 |
| oncoming (54) | **0.865** | 0.780 | 0.645 | 0.860 |

Read honestly: the closing ratio is the best single separator for the "same-direction traffic called parked" error,
and the lateral flow nearly matches the best existing feature for oncoming traffic; for crossing traffic the existing
trajectory remains better. They are complements, not replacements, and 76 tracks from two rides are too few to put a
number on the gain. The definitive measure is the new ablation table (`results/typology/feature_ablation.csv`), which
scores the old feature set against old + flow and the full new set on the 378 labelled tracks, per model.

Each vehicle also gets its position relative to the painted lines found by the new infrastructure stage: lines
between the rider's path and the vehicle at its distance (solid and yellow ones counted separately), whether it sits
inside the rider's lane, and the share of the track during which the rider was in a bike lane - "in front of me in my
bike lane" is an intrusion, "just left of my bike-lane line" is traffic in the next lane.

## 3. Bike lanes

**Method** (`bikesafe.infrastructure`). Every sampled frame (2 per second) is warped into a bird's-eye view of the road
2.5-18 m ahead with the self-calibrated flat-road geometry the rest of the pipeline already uses. In that view lane
lines are vertical and painted stencils look like upright icons.

- Paint is found as thin (at most 0.35 m), elongated strokes brighter than the asphalt beside them, which rejects kerb
  concrete, crosswalk bars, cracks and shadow edges. Lines are extracted greedily (the best-supported line first, then
  the paint it explains is removed) each with its own slant, and classified solid / broken and white / yellow.
- Bicycle stencils are found by YOLOE-26 in the bird's-eye view. In the camera view the stencil is too foreshortened;
  in the bird's-eye view the large model found the stencils at 454, 461, 481 and 493 s, while the medium model found
  none of the three it was tried on.
- Per sample, transparent rules give bike-lane evidence: for - a solid white line 0.3-2.4 m to the left (US lane lines
  between same-direction traffic lanes are broken), a narrow lane between two lines, a stencil in the rider's corridor,
  green paint; against - a traffic lane between well-painted broken lines 3 m or more apart, a yellow centre line
  beside the rider. The evidence is averaged over 7 s and held through junctions, where paint disappears.
- The riding context keeps CLIP's call for separated paths (which it gets right, F1 0.85) and lets the paint decide
  bike lane vs shared road wherever it has evidence. The CLIP-only context is kept as `scene_clip`.

**Check**: the three demo clips, labelled per second by hand from the road view (junction crossings excluded): the
bike-lane clip has a boulevard section (shared road), a bike pocket before an intersection and a long kerbside bike
lane with a solid line and parked cars; the other two clips are shared residential streets.

| riding context, 142 labelled seconds | seconds decided | accuracy | bike-lane precision | bike-lane recall | F1 |
|---|---:|---:|---:|---:|---:|
| CLIP probe (current) | 100% | 89.4% | 100% | 64.3% | 0.78 |
| lane paint alone | 83% | 94.9% | 97.4% | 88.1% | 0.93 |
| **fused (new default)** | 100% | **95.8%** | 97.4% | **88.1%** | **0.93** |

The residual misses are where a car travelling alongside hides the bike-lane line for several seconds. The rules and
thresholds were tuned while looking at these same clips, so this table is optimistic; the unbiased number comes from
your 314 labelled scene frames, which `bikesafe.train` now scores for the CLIP probe alone and fused with the lane
paint (`scene_confusion_probe_plus_lane_paint.csv`). Night rides, where only the headlight shows paint, are untested.

## 4. Traffic signals and signs

**Method.** YOLOE-26 runs on each sampled camera frame with a road-furniture vocabulary (traffic light, pedestrian
signal, stop / yield / speed-limit / street-name / one-way / no-parking / no-turn / do-not-enter / bike-lane /
pedestrian-crossing / warning / lane-use / guide signs, generic traffic sign) and distractor classes (billboard, shop
sign, licence plate, street lamp, vehicles, people) that absorb look-alikes. The prompts' text features are cached in
`models/infrastructure_vocab.npz` (33 KB), so the 240 MB text encoder is never needed at run time.

- **Signal state.** The lit lamp is a bright, saturated, round blob inside the housing. On this camera lit reds look
  pink-magenta and LED greens cyan-green; sky and walls behind side-on heads touch the crop's edge or form tall strips
  and are rejected. On 44 real signal-head crops from the demo clips (16 with a lit lamp facing the camera), **40 are
  right**; of the lit ones 14 of 16 (the misses are a tiny far head and a faint side-on one), and two unlit heads were
  called yellow because of a yellow building behind them - so a yellow reading must be confirmed by two heads.
- **Stop signs.** The open vocabulary alone called 12 things stop signs in the demo clips; 2 were stop signs. The
  others were DO NOT ENTER and NO PARKING signs, a red shop sign and a basketball hoop. Every stop-sign candidate is
  now confirmed by the COCO-trained YOLO11m already in the repository (class "stop sign"): it kept the real ALL-WAY stop
  facing the rider and rejected all 10 false ones; it also drops a small side-on stop sign that faces cross traffic,
  which is not a stop sign the rider passed.
- **Other signs** are detected well as signs, but their fine type is an open-vocabulary guess (a BIKE LANE sign was
  called a speed-limit sign). Reports use the instance counts and treat the types as indicative.
- **One sign, one row.** Detections of the same physical sign or signal head are linked across samples (a static
  object moves straight away from the focus of expansion and grows), giving `signs.csv`.

**Spot check on the demo clips** (60 distinct signal heads and signs sampled at random, best detection of each
viewed): 23 of 24 "signal heads" are signal heads (one teal awning is not). Of 36 "signs", 20 are this demo's own
burned-in overlay labels ("parked 2m" boxes), which raw footage does not have; of the other 16, about 12 are real
traffic signs (street names, DO NOT ENTER, parking restrictions) and 4 are shop signs - roughly 75% precision, with
shop signs the main confusion. Crops and verdicts for the signal states are in `results/infrastructure/`.

**New outputs.** Per second in `timeline_1s.csv`: `in_bike_lane`, `bike_lane_score`, `traffic_light_state`,
`n_signal_heads`, `n_signs`, `stop_sign`, `bike_stencil`, plus the fused `scene` and the old `scene_clip`. New events:
`stop_sign` (one per sign passed; on the demo clips exactly the one real stop sign) and `red_light_wait` (a stop of 3 s
or more with a red signal read for at least 2 s of it; on the demo clips exactly the two real waits, one of them 23 s
long with a bus crossing in front of the signal - defining the wait by the stop rather than by the signal readings
keeps such a wait in one piece). New summary columns: minutes in a painted bike lane, stop signs, red light waits and
their total duration, seconds with a signal ahead. The overlay clip draws the lane lines (green when
the rider is in a bike lane), signals in their state colour and signs.

## 5. Faster

- **FP16 on the GPU by default** for the vehicle detector, CLIP and the depth model (`--fp32` to switch off). Half
  precision typically gives 1.5-2x detector throughput on GPUs with tensor cores; boxes differ only in rounding.
- **Hardware video decoding** (`bikesafe.run --hw-decode`): every stage decodes the whole ride, and for 360 exports
  decoding rather than the models can dominate. It falls back to software decoding silently.
- **CPU stages in parallel across rides** (`--jobs`, default half the cores): ego motion and track features.
- **Nothing recomputed that does not need to be**: after this update the same `bikesafe.run` command runs only the
  new infrastructure pass and the track features (their tables carry a version and are recomputed only when older).
- **Optional**: `--skip-depth` drops the depth pass (its features moved leave-one-video-out scores by -2.3 to +1.6
  points, within noise); a TensorRT engine is a drop-in replacement for the detector
  (`yolo export model=models/yolo11n.pt format=engine half=True imgsz=1280`).
- `bikesafe.run` ends with the time each stage took, so the next bottleneck is visible rather than guessed.

On this sandbox's 4-core CPU the new infrastructure pass took about 2 s per sample with the large YOLOE model (a
45 s clip at 2 samples per second in under 3 minutes); on a GPU its two detector passes per sample are a few tens of
milliseconds, so like the depth pass it should be bounded by video decoding. No GPU timing is claimed here.

## 6. What to run

```bash
python -m bikesafe.run videos --out-root work --render-minutes 1 --hw-decode
python -m bikesafe.train
python -m bikesafe.exposure --analysis work/analysis --perception work/perception
```

The first command reuses perception, ego motion and depth, runs the new infrastructure pass (downloads
`yoloe-26l-seg.pt`, 79 MB, once) and recomputes track features. The second retrains and writes
`feature_ablation.csv` and the scene-context comparison. The third re-applies the retrained model. Then
`python -m bikesafe.figures` refreshes the figures.

## 7. Limitations and next steps

- **The accuracy gain of the new features is not measured yet** - it needs the per-detection data on your machine.
  If `feature_ablation.csv` shows no gain, the next lever is the one the previous report named: a human pass over the
  track cards, concentrating on moving vehicles labelled parked and vice versa.
- **Labels are still AI-assisted.** My own labelling of the demo tracks left 13 of 88 unclear from start / middle / end
  frames, which suggests the existing 378 labels contain errors too.
- Bike-lane detection fails where a car hides the line for longer than the smoothing window, and is untested at night
  and on sharrow-marked streets. A segmentation model trained on Mapillary Vistas (which has a bike-lane class) would
  be the natural upgrade; it could not be downloaded here.
- Sign types other than stop signs and signals are guesses; a sign classifier trained on US signs (e.g. the Mapillary
  Traffic Sign Dataset) would fix that if sign types matter for the analysis.
- The per-vehicle lane-paint features are noisier than the bike-lane decision: bright edges along the bodies of parked
  cars sometimes read as lines on the rider's right. The bike-lane rule needs a solid line on the left and is not
  affected, but the "lines between rider and vehicle" counts for parked cars are; the ablation will show whether they
  help.
- The demo clips carry the old overlay; the burned-in legend and labels produced some detections (ignored here, but
  worth remembering if these clips are reused).
