# Output tables

Everything `python -m bikesafe.run` writes for people to use is in `results/corpus/`: one folder per ride and two
tables across rides. Times are seconds from the start of the video (`t_*`, `second`) and, where given, local camera
time (`time_local`, `timestamp_local`), which joins to Strava GPS. Distances are metres along the road ahead (`z`,
`closest_m`) and to the side (`lateral_*`, negative = left of the rider's heading).

Start with `vehicles.csv` and `timeline_1s.csv`; the other tables are for checking and modelling.

## The relations

| Code | In words | Means |
|---|---|---|
| `ego_lane` | in my lane | In the lane or path the rider rides in: ahead in the same lane (moving, or **stopped at a light or in a queue**), or intruding into the rider's bike lane |
| `adjacent_same` | other lane, same way | Same direction, another lane of the rider's road, including vehicles overtaking the rider |
| `oncoming` | oncoming | Opposite direction on the rider's road, no physical median |
| `parked` | parked | Left at the kerb, in a parking lane, a driveway or a lot - **not** a vehicle stopped in traffic |
| `cross_side` | side road / other roadway | Side streets and crossing or turning traffic (including vehicles **waiting** at a junction), beyond a median, another road |

A vehicle stopped at a red light, at a stop sign or in a queue is labelled and modelled by where it is, never as
parked. The model still gets some of them wrong - see "Known limitations" in `TYPOLOGY_PIPELINE.md`.
`group` merges them into the professor's four groups: `in_my_lane`, `sharing_road` (adjacent + oncoming), `parked`,
`not_sharing` (cross_side).

## Per ride: `results/corpus/<video>/`

### `vehicles.csv` - one row per physical vehicle (read this first)

| Column | Meaning |
|---|---|
| `vehicle_id` | One id per physical vehicle (tracks of one vehicle joined, double boxes merged) |
| `time_local`, `t_start`, `t_end`, `duration_s` | When it was in view |
| `relation_text`, `relation`, `group` | What it was to the rider (table above) |
| `confidence` | Share of the model's vote that went to that relation (0.2 = pure guess among 5, 1 = certain) |
| `check` | Why a person should look at it: `model unsure` (confidence < 0.45), `may not be a vehicle` |
| `counted` | Counts in the per-minute exposure: came within 15 m (where relations are validated), box >= 35 px, seen >= 1 s |
| `vehicle_class` | car / motorcycle / bus / truck (detector's majority class) |
| `closest_m`, `lateral_at_closest_m`, `side` | Closest approach ahead, the gap to its near side at that moment, and on which side |
| `seen_while_rider_stopped` | Share of its time in view during which the rider was stopped (at a light, a stop sign) |
| `not_vehicle_score` | Zero-shot probability that the "vehicle" is a delivery robot, scooter, bin, bench or cart (reliable by day only) |
| `plate` | Licence plate as read (most vehicles have none: no plate in view, or not readable) |
| `box_h_max_px`, `n_tracks`, `track_ids` | Largest box height, and the tracks (rows of `tracks_typed.csv`) it is made of |

### `timeline_1s.csv` - one row per second

| Column | Meaning |
|---|---|
| `second`, `timestamp_local` | The second, and its local time |
| `ego_speed_mps`, `ego_moving` | Rider speed (m/s) and whether the rider moves (> 1 m/s) |
| `scene` | Riding context: `separated_path`, `bike_lane`, `shared_road` (CLIP probe; accuracy 0.886 on 307 labelled frames, bike lane F1 0.49), `p_*` its probabilities |
| `bike_lane_score`, `in_bike_lane` | The lane-paint detector's evidence that the rider is in a painted bike lane (-1..1, smoothed over 7 s) and its call; empty where no paint was in view. Not the riding context: it finds most painted bike lanes but also calls most shared roads with an edge line bike lanes |
| `lane_lines_visible` | Most painted lines found in one sample of that second |
| `traffic_light_state` | `red` / `yellow` / `green` of the signal heads facing the rider, empty if none was read |
| `n_signal_heads`, `n_signs`, `signs_in_view` | Signal heads and traffic signs in view, and the types of the signs |
| `stop_sign`, `bike_stencil` | A stop sign / a painted bicycle stencil in view |
| `speed_limit_posted` | Last speed-limit sign read (mph), held for up to 3 minutes |
| `n_ego_lane` ... `n_cross_side` | Vehicles of each relation visible that second (box >= 20 px) |
| `min_z_ego_lane_m` | Distance to the nearest vehicle in the rider's lane |
| `min_lateral_adjacent_m`, `min_lateral_oncoming_m` | Smallest side gap to same-direction / oncoming traffic within 20 m |
| `door_zone_parked` | Parked vehicles less than 1.5 m to the right and within 10 m ahead (dooring exposure) |

### `events.csv` - one row per event and vehicle

| `event` | When |
|---|---|
| `overtaken_by_vehicle` | A vehicle came within 8 m and passed the rider faster than it rode (from geometry, not the relation); `clearance_m` is the side gap, `side` the side, `speed_difference_mps` |
| `close_pass_under_1p5m` | An overtake with less than 1.5 m clearance |
| `vehicle_close_in_my_lane` | A vehicle in the rider's lane came within 6 m |
| `crossing_traffic_near` | Side-road / crossing vehicle within 12 m |
| `oncoming_close_pass` | Oncoming vehicle within 15 m ahead and less than 1.5 m to the side |
| `emergency_vehicle_review` | Up to 10 large vehicles CLIP thinks may be police, ambulance or fire - to check by eye (`score`) |
| `stop_sign` | A stop sign passed (`object_id` in `signs.csv`) |
| `red_light_wait` | The rider stopped >= 3 s with a red signal read ahead for >= 2 s (`duration_s`, `red_seconds`) |

`z_min_m` and `x_inner_absmin_m` are the vehicle's closest approach and smallest side gap.

### `signs.csv` - one row per physical sign or signal head

`object_id`, `kind` (`sign` / `light`), `category` (sign type: stop, yield, speed_limit, do_not_enter, one_way,
no_turn, parking, bike, pedestrian, warning, street_name, guide, lane_use, bus_stop, roadwork - cones, barricades and
road-work, closure and detour signs, which the segmenter finds as signs; `other_sign` when it was too small or unclear
to type; `traffic_light`, `pedestrian_signal`), `t_first` / `t_last` (seen), `n_samples`, `conf_max`,
`box_h_max` (px), `states` (lamp states read on a signal head), `sign_text` (what Florence-2 read), `speed_limit`.

### `tracks_typed.csv` - one row per track, with every model feature

The input of the relation model and its output per track (`relation`, `proba_*`, `relation_confidence`) next to
the vehicle-level answer (`vehicle_relation`, `vehicle_id`). About 130 columns; the features are described in
`TYPOLOGY_PIPELINE.md` (step 5). `rule_relation` is the transparent rule baseline. Use `vehicles.csv` for counts.

### `identity_audit.json`

How the vehicle ids agree with the plates read: tracks with one plate split over several vehicle ids (missed joins,
or a vehicle seen again later), vehicle ids carrying two plates (wrong joins).

## Across rides

### `corpus_summary.csv` - one row per ride

`minutes`, `moving_minutes`, `median_moving_speed_mps`; minutes per riding context (`minutes_*`); per relation the
counted vehicles (`vehicles_*`), vehicles per minute (`per_min_*`) and tracks (`tracks_*`, for comparison with runs
before vehicle ids); `vehicles_counted`; vehicle-id joins (`duplicate_links`, `break_links`); plate audit
(`plate_*`); event counts (`events_*`); `door_zone_seconds`; `median_overtake_clearance_m`; signs passed by
type (`signs_*`, box >= 18 px); `posted_speed_limit_median`, `minutes_posted_limit_known`;
`red_light_wait_seconds`, `seconds_with_signal_ahead`.

### `exposure_by_scene.csv`

Per ride and riding context: minutes, median speed, vehicles per minute of each relation, share of time in the door
zone.

## Where the pipeline's own records are

`work/analysis/<video>/` holds the stage outputs these tables are built from (detections with features, lane
samples, sign detections and crops, plates with every read, vehicle links) and `infrastructure.json`, which says
what part of the ride the sign and lane pass covered. `data/typology/` holds the labels, the review log of
`bikesafe.review` and the labelling guide; `results/typology/` the model evaluation.
