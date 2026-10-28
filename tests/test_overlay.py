"""The overlay's lane lines laid on each frame's paint - only the rider's lane, only on steady lanes - its label
placement and banner, and street-name completion."""

import numpy as np
import pandas as pd

from bikesafe.render import RESUME_S, LaneOverlay, find_paint, place_label, smoothed_scene
from bikesafe.signs import StreetNames

CALIB = {"width": 1920, "height": 1080, "horizon_v": 600.0, "focal_px": 672.0, "cam_height_m": 1.2,
         "lateral_height_m": 1.32}
FPS = 24.0


def road(x_m: float | list[float] | None, u_foe: float = 960.0) -> np.ndarray:
    """A grey road with white lines x_m metres to the side of the heading (none for x_m None)."""
    image = np.full((CALIB["height"], CALIB["width"], 3), 90, np.uint8)
    for x in [] if x_m is None else x_m if isinstance(x_m, list) else [x_m]:
        for v in range(int(CALIB["horizon_v"]) + 30, CALIB["height"]):
            d = v - CALIB["horizon_v"]
            u = u_foe + x * d / CALIB["lateral_height_m"]
            half = 0.06 * d / CALIB["lateral_height_m"]
            image[v, int(u - half): int(u + half) + 1] = 200
    return image


def lanes_table(x_m: float | list[float], seconds: float = 4.0, crosswalk: tuple[float, float] | None = None) -> pd.DataFrame:
    """Samples twice a second reporting the lines x_m; a crosswalk across the rider's path over `crosswalk` (s)."""
    xs = x_m if isinstance(x_m, list) else [x_m]
    times = np.arange(0, seconds, 0.5)
    at_crossing = (times >= crosswalk[0]) & (times <= crosswalk[1]) if crosswalk else np.zeros(len(times), bool)
    return pd.DataFrame({"time_s": times, "frame": (times * 12).astype(int), "u_foe": 960.0, "n_lines": len(xs),
                         "crosswalk_share": np.where(at_crossing, 0.4, 0.0),
                         "lines_json": ";".join(f"{x:.2f}:1:0.00:0.000:2.8:13.8" for x in xs)})


def shown(overlay: LaneOverlay, image: np.ndarray, frames: range) -> list[list[float]]:
    """Per frame, where the drawn lines are (m across the road at the rider), left to right."""
    out = []
    for frame in frames:
        lines = overlay.update(image, frame, np.zeros((0, 4)))
        out.append(sorted(round(float(st["ab"][1]) * CALIB["lateral_height_m"], 1) for st in lines if st["alpha"] == 1.0))
    return out


def u_at(line: dict, d: float) -> float:
    return float(line["ab"][0] + line["ab"][1] * d)


def test_a_line_off_its_paint_is_moved_onto_it():
    image = road(-1.2)  # the paint is 0.2 m further left than predicted
    lateral = CALIB["lateral_height_m"]
    a, b = 960.0, -1.0 / lateral
    found = find_paint(image[..., 2].astype(np.float32), a, b, (60.0, 470.0), CALIB["horizon_v"], lateral,
                       np.zeros((0, 4)), 15.0)
    assert found is not None
    fa, fb, lowest = found
    for d in (100.0, 300.0):
        assert abs((fa + fb * d) - (960.0 - 1.2 * d / lateral)) < 3
    assert lowest > 400  # found down to the bottom of the picture
    assert find_paint(road(None)[..., 2].astype(np.float32), a, b, (60.0, 470.0), CALIB["horizon_v"], lateral,
                      np.zeros((0, 4)), 15.0) is None


def test_lines_fade_in_on_paint_follow_it_and_fade_out_without_it():
    overlay = LaneOverlay(lanes_table(-1.0), CALIB, FPS)
    shifted = road(-1.15)  # the camera rolled: the paint sits 0.15 m left of the samples' line
    for frame in range(24):
        lines = overlay.update(shifted, frame, np.zeros((0, 4)))
    assert len(lines) == 1 and lines[0]["alpha"] == 1.0
    assert abs(u_at(lines[0], 300.0) - (960.0 - 1.15 * 300.0 / 1.32)) < 8  # on the paint, not the prediction
    blank = road(None)  # paint gone (a seam the segmenter took for paint)
    for frame in range(24, 70):
        lines = overlay.update(blank, frame, np.zeros((0, 4)))
    assert lines == []


def test_only_the_riders_own_lane_is_drawn():
    xs = [-2.6, -1.0, 0.9, 2.8]  # the rider's lane between -1.0 and 0.9, a lane either side
    overlay = LaneOverlay(lanes_table(xs), CALIB, FPS)
    assert shown(overlay, road(xs), range(30))[-1] == [-1.0, 0.9]
    arrow = [-1.0, -0.35, 0.9]  # paint inside the lane (an arrow's shaft) is not one of its sides
    overlay = LaneOverlay(lanes_table(arrow), CALIB, FPS)
    assert shown(overlay, road(arrow), range(30))[-1] == [-1.0, 0.9]


def test_no_lines_through_a_junction_or_a_turn_and_back_only_once_steady():
    xs = [-1.0, 0.9]
    overlay = LaneOverlay(lanes_table(xs, seconds=10.0, crosswalk=(3.0, 4.0)), CALIB, FPS)
    frames = shown(overlay, road(xs), range(int(10 * FPS)))
    at = lambda t: frames[int(t * FPS)]  # noqa: E731
    assert at(2.0) == [-1.0, 0.9]
    assert at(2.6) == [] and at(4.5) == []  # gone before the crosswalk, not back straight after it
    assert at(4.6 + RESUME_S + 0.5) == [-1.0, 0.9]
    turn = pd.DataFrame({"time_s": np.arange(0, 10, 0.1)})  # a 90 degree turn between 4 and 9 s
    turn["turn_rad"] = np.radians(np.clip((turn.time_s - 4.0) * 18.0, 0.0, 90.0))
    overlay = LaneOverlay(lanes_table(xs, seconds=10.0), CALIB, FPS, ego=turn)
    frames = shown(overlay, road(xs), range(int(10 * FPS)))
    assert at(3.0) == [-1.0, 0.9] and at(6.0) == [] and at(9.2) == []


def test_labels_do_not_cover_one_another_or_leave_the_picture():
    taken: list[tuple[int, int, int, int]] = []
    first = place_label(taken, 100, 400, 500, 200, 30, 1920, 56)
    second = place_label(taken, 150, 405, 520, 200, 30, 1920, 56)  # a box just beside and below the first
    a, b = taken
    assert first == (100, 370) and second is not None
    assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    edge = place_label(taken, 1850, 300, 400, 200, 30, 1920, 56)
    assert edge is not None and edge[0] + 200 <= 1920
    spot = place_label(taken, 600, 40, 200, 200, 30, 1920, 56)
    assert spot is not None and spot[1] >= 56  # never under the banner
    crowd = [(0, 0, 1920, 1080)]  # no room anywhere: no label rather than one over another
    assert place_label(crowd, 100, 400, 500, 200, 30, 1920, 56, steps=1) is None


def test_the_banner_context_does_not_flip_every_second():
    scene = pd.Series(["bike_lane", "shared_road", "bike_lane", "bike_lane", "shared_road", "bike_lane", "bike_lane"])
    assert (smoothed_scene(scene) == "bike_lane").all()


def test_partial_street_names_are_completed_from_names_read_in_full():
    reads = (["Westmoreland"] * 5 + ["estmoreland"] * 4 + ["Hoover St"] * 6 + ["Hoover"] * 3 + ["Holly Av"] * 3
             + ["West"] * 3 + ["tmor", "HOOVE", "Hoovel St", "Ho", "Pico BI", "Westmoreland 800 S"])
    names = StreetNames(reads)
    assert "estmoreland" not in names.names  # a frequent fragment is not a name
    assert names("tmor") == "Westmoreland" and names("estmoreland") == "Westmoreland"
    assert names("HOOVE") == "Hoover St" and names("Hoovel St") == "Hoover St"
    assert names("Westmoreland 800 S") == "Westmoreland 800 S"  # already holds a name: kept with its block number
    assert names("Pico BI") == "Pico Bl"  # unknown street: only the suffix put right
    assert names("Ho") == "Ho"  # too short to complete
    assert names("West") == "West"  # a direction, not part of Westmoreland
    assert names("Hol") == "Hol" and StreetNames(reads + ["Hollywood"] * 3)("Holl") == "Holl"  # two streets fit
