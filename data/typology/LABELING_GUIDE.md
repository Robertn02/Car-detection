# Labeling guide: vehicle relation to the rider

Each track card shows one vehicle track (magenta box) at its start, middle and end, the path of its wheel point
(yellow), and a zoomed crop. Label the vehicle's relation to the rider **over the visible part of the track**.
If the relation changes (for example a car pulls out of a parking space), use the relation for the longest part.

| Label | Professor's group | Use when |
|---|---|---|
| `ego_lane` | in my lane | In the lane or path the rider is travelling in: ahead of the rider in the same lane (moving or queued at a light), or intruding into the rider's bike lane. |
| `adjacent_same` | sharing the road | Same direction on the rider's roadway but a different lane, including vehicles overtaking the rider and vehicles in the traffic lane beside the rider's bike lane. |
| `oncoming` | sharing the road | Opposite direction on the rider's roadway with no physical median (painted centre lines only). |
| `parked` | parked | Parked at a curb, in a parking lane, a driveway or a lot; not part of traffic. Vehicles stopped in a traffic lane at a light are **not** parked. |
| `cross_side` | not sharing | Not on the rider's own roadway: side roads and intersecting streets, traffic crossing or turning across the rider's path, driveways, and vehicles beyond a raised median or on another roadway (freeway, frontage road, overpass). |
| `unclear` | excluded | Too small, dark, occluded, or the box switches between different vehicles. |

Tips

- Parked cars also slide outward in the image as the rider approaches; compare their motion with the curb and other
  parked cars rather than with the frame edge.
- On a separated bike path or campus walkway nothing is `ego_lane`; nearby street traffic is `adjacent_same`,
  `oncoming` or `cross_side` depending on where it is.
- A vehicle waiting on a side street to enter is `cross_side`.

Files

- `track_labels.csv`: `card_id, video, track_id, label, labeler, notes`
- `scene_labels.csv`: `card_id, video, frame, label, labeler, notes` with `label` one of
  `separated_path` (bike path, campus walkway, sidewalk or plaza away from car lanes),
  `bike_lane` (painted or buffered bike lane on a street), `shared_road` (riding in a general traffic lane,
  including sharrows and intersections without a bike lane), or `unclear`.

The first labeling pass was done by Claude (AI-assisted visual review, `labeler=claude-visual-review`). A human
reviewer should spot-check and correct these files; re-running `python -m bikesafe.train` refreshes every metric.
