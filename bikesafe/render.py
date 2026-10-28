"""Render a typology overlay clip: boxes coloured by relation to the rider, scene and ego-speed banner, legend.

When bikesafe.infrastructure has run, the clip also shows the painted lane lines it found (green when the rider is in
a bike lane), traffic lights with their state, and traffic signs with their type and text ("SPEED LIMIT 35",
"Hoover St"). Licence plates are boxed and read out (bikesafe.plates: the track's consensus reading when
bikesafe.plates has run, otherwise a confident reading of the frame; --no-plates to switch off). A vehicle boxed twice
by the detector (car + truck) is drawn once, in the colour of its vehicle-level relation.

Lane lines and road furniture are measured on sampled frames (2 per second). In between, a sign's box is
interpolated from the samples before and after. A lane line found on two of three consecutive samples is predicted
for every frame from the samples around it, then laid on the paint of that frame (LaneOverlay): the handlebar camera
rolls and pitches between samples, so a line drawn from the samples alone drifts off its paint. Lines with no paint
under them in the frame (a crack or seam the segmenter took for paint) fade out, as do lines at a junction's
crosswalk; lines are drawn as strips of road paint in perspective, no further ahead than their paint was seen, and
behind the vehicles.

Labels are short ("same way", "side road": the legend gives the colours in full) and placed so they do not cover
one another. The banner's riding context is the most common one over SCENE_SMOOTH_S around the moment, so it does
not flip with every second's call (timeline_1s.csv keeps the per-second call). Its speed is the GPS speed where the
ride has a GPS track (bikesafe.gps; the speed measured from the video reads about 0.6 of it), else the video's.

The clip has every frame of the video at its full size (--width to shrink it). Vehicles are detected on every second
frame (bikesafe.perceive --target-fps), so their boxes are interpolated on the frames in between. It is written as
H.264 when imageio-ffmpeg is installed (bikesafe.common.VideoOut), else as MPEG-4.

    python -m bikesafe.render VID_20260224_162848_00_006 --start-s 300 --duration-s 60
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import RELATION_COLOURS_BGR, RELATION_TEXT, RELATIONS, ROOT, VIDEOS, VideoOut, open_video, read_json
from bikesafe.infrastructure import JUNCTION_CROSSWALK, JUNCTION_LINES, LINE_Z_MAX, REF_Z, line_extents, parse_lines

LABELS = RELATION_TEXT  # the legend
BOX_LABELS = {"ego_lane": "in my lane", "adjacent_same": "same way", "oncoming": "oncoming", "parked": "parked",
              "cross_side": "side road"}  # on the boxes
SCENE_SMOOTH_S = 7  # seconds of riding context the banner takes the most common of
LIGHT_COLOURS_BGR = {"red": (60, 60, 255), "yellow": (0, 215, 255), "green": (90, 220, 60), "off": (200, 200, 200)}
SIGN_COLOURS_BGR = {"stop": (40, 40, 230), "do_not_enter": (40, 40, 230), "yield": (40, 40, 230),
                    "warning": (0, 210, 255), "pedestrian": (0, 230, 200), "street_name": (90, 200, 60),
                    "guide": (90, 200, 60), "bike": (80, 230, 80), "roadwork": (0, 140, 255), "bus_stop": (230, 160, 60)}
SIGN_COLOUR_BGR = (255, 255, 0)  # other regulatory signs
SIGN_NAMES = {"parking": "parking", "no_turn": "no turn", "one_way": "one way", "do_not_enter": "do not enter",
              "lane_use": "lane use", "bike": "bike sign", "pedestrian": "pedestrian", "warning": "warning",
              "guide": "guide", "street_name": "street", "other_sign": "sign", "yield": "YIELD", "stop": "STOP",
              "roadwork": "road work", "bus_stop": "bus stop"}
LANE_COLOURS_BGR = {"bike": (80, 230, 80), "solid": (255, 255, 255), "broken": (180, 180, 180), "yellow": (0, 220, 255)}
LABEL_MAX_Z = 25.0  # m; farther vehicles get a thin box and no label (relations are validated within 15 m)
UNSURE_CONFIDENCE = 0.45  # a relation the model gives less than this probability is labelled with a "?"
MAX_BOX_GAP = 2  # processed frames (1/6 s at 12 fps) a track's box is carried across when the detector misses it
SAMPLE_HOLD_S = 0.3  # draw an object or line on frames up to this far from the last sample that saw it
MAX_INTERP_GAP_S = 1.2  # interpolate between two samples of one object or line at most this far apart
LINE_MATCH_M = 0.45  # a line in the next sample within this lateral distance (at REF_Z) is the same line
MAX_LINE_X = 3.5  # metres either side of the rider's path that lines are drawn
LINES_PER_SIDE = 2
EXTEND_M = 4.0
DUPLICATE_M = 0.6
# Laying predicted lines on the paint of each frame (LaneOverlay)
PAINT_W_M = 0.12  # lane paint width, for judging a bright stripe in the image
DRAW_W_M = 0.17  # width of the strip drawn over it
SNAP_SEARCH_M = 0.3  # paint is looked for this far either side of the predicted line
SNAP_ROWS = 28  # image rows looked at along a line
SNAP_MIN_POINTS = 5  # rows that must find paint for the frame to place the line
SNAP_MIN_CONTRAST, SNAP_CONTRAST_SHARE = 8.0, 0.15  # paint stands this far above the asphalt (grey levels, share)
SNAP_GAIN = 0.5  # weight of a frame's measurement against the line's running estimate
SUPPORT_DECAY = 0.85  # per frame; a line is shown while its share of frames with paint found stays above SHOW_SUPPORT
SHOW_SUPPORT = 0.4
KEEP_S = 0.8  # a line followed on its paint stays this long after the samples stop reporting it
FADE_S = 0.25  # lines fade in and out over this long


def draw_legend(frame: np.ndarray, lanes: bool = True) -> None:
    """Box colours and, when lane lines are drawn, their colours, on a translucent panel in the bottom-left corner."""
    rows = [(LABELS[rel], RELATION_COLOURS_BGR[rel], "box") for rel in RELATIONS]
    if lanes:
        rows += [("lane line", LANE_COLOURS_BGR["solid"], "line"), ("bike-lane line", LANE_COLOURS_BGR["bike"], "line")]
    step, x = 27, 16
    y = frame.shape[0] - 14 - step * len(rows)
    top, bottom, right = y - 22, frame.shape[0] - 8, x + 262
    frame[top:bottom, x - 8:right] = (frame[top:bottom, x - 8:right] * 0.35 + 20 * 0.65).astype(np.uint8)
    for i, (text, colour, kind) in enumerate(rows):
        cy = y + step * i
        if kind == "box":
            cv2.rectangle(frame, (x, cy - 13), (x + 22, cy + 5), colour, -1)
        else:
            cv2.line(frame, (x, cy - 4), (x + 22, cy - 4), colour, 5, cv2.LINE_AA)
        cv2.putText(frame, text, (x + 32, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def place_label(taken: list[tuple[int, int, int, int]], x: int, box_top: int, box_bottom: int, w: int, h: int,
                frame_w: int, min_y: int, steps: int = 6) -> tuple[int, int] | None:
    """Top-left corner for a w x h label of the box whose left edge is x and top and bottom box_top and box_bottom:
    just above the box if that is free, else inside its top, else stacked up to `steps` labels higher, never over a
    label already placed (`taken`, which the label is added to) and never off the picture. None when no place is
    free (a crowd of distant boxes): the box is then drawn without a label."""
    x = int(min(max(0, x), max(0, frame_w - w)))
    above = max(min_y, box_top - h)
    tries = [above, box_top + 2 if box_bottom - box_top > 2.5 * h else above]
    tries += [max(min_y, above - k * (h + 2)) for k in range(1, steps + 1)]
    for y in tries:
        rect = (x, y, x + w, y + h)
        if not any(rect[0] < b[2] and b[0] < rect[2] and rect[1] < b[3] and b[1] < rect[3] for b in taken):
            taken.append(rect)
            return x, y
    return None


def project_line(x0: float, shear: float, u_foe: float, calib: dict, z_range: tuple[float, float] = (2.5, 14.0)) -> np.ndarray:
    """Image polyline of a lane line X(Z) = x0 + shear*(Z - REF_Z) on the flat road (inverse of the bird's-eye warp)."""
    zs = np.linspace(z_range[0], z_range[1], 12)
    xs = x0 + shear * (zs - REF_Z)
    rows_below = calib["focal_px"] * calib["cam_height_m"] / zs
    u = u_foe + xs * rows_below / calib["lateral_height_m"]
    v = calib["horizon_v"] + rows_below
    return np.column_stack([u, v]).astype(np.int32)


# --------------------------------------------------------------------------------------------------------------------
# lane lines: matched across samples, smoothed, interpolated


def lane_line_tracks(lanes: pd.DataFrame, calib: dict | None = None) -> pd.DataFrame:
    """One row per line per sample with a line id that follows the same painted line from sample to sample, its
    position and slant smoothed over the neighbouring samples, and whether it is confirmed (found on at least two of
    three consecutive samples).

    Line positions are measured from the rider's heading, whose estimate (the focus of expansion) can move by tens of
    pixels between samples - 0.4 m at REF_Z for 50 px. Lines are therefore matched and smoothed in `x_img`, the
    position relative to the image centre column, which does not move with that estimate (needs `calib`)."""
    rows, next_id, previous = [], 0, []
    for i, sample in enumerate(lanes.itertuples()):
        junction = sample.n_lines >= JUNCTION_LINES or getattr(sample, "crosswalk_share", 0.0) >= JUNCTION_CROSSWALK
        offset = 0.0
        if calib is not None:
            rows_at_ref = calib["focal_px"] * calib["cam_height_m"] / REF_Z
            offset = calib["lateral_height_m"] * (float(sample.u_foe) - calib["width"] / 2) / rows_at_ref
        current = []
        for (x, solid, yellow, shear), (z_near, z_far) in zip(parse_lines(sample.lines_json),
                                                              line_extents(sample.lines_json)):
            x_img = x + offset
            near = [p for p in previous if abs(p["x_img"] - x_img) < LINE_MATCH_M and p["line_id"] not in
                    {c["line_id"] for c in current}]
            line_id = min(near, key=lambda p: abs(p["x_img"] - x_img))["line_id"] if near else next_id
            next_id += line_id == next_id
            current.append({"sample": i, "line_id": line_id, "x": x, "x_img": x_img, "offset": offset, "solid": solid,
                            "yellow": yellow >= 0.5, "shear": shear, "z_near": z_near, "z_far": z_far,
                            "junction": junction})
        rows += current
        previous = current
    table = pd.DataFrame(rows, columns=["sample", "line_id", "x", "x_img", "offset", "solid", "yellow", "shear",
                                        "z_near", "z_far", "junction"])
    if table.empty:
        return table.assign(confirmed=pd.Series(dtype=bool))
    seen = table.groupby("line_id")["sample"].apply(set)
    table["confirmed"] = [sum(s + k in seen[lid] for k in (-1, 0, 1)) >= 2 for s, lid in zip(table["sample"], table.line_id)]
    group = table.groupby("line_id")
    for col in ("x_img", "shear"):
        table[col] = group[col].transform(lambda c: c.rolling(3, center=True, min_periods=1).median())
    table["x"] = table.x_img - table.offset
    table["solid"] = group.solid.transform(lambda c: c.rolling(3, center=True, min_periods=1).mean() >= 0.5)
    table["yellow"] = group.yellow.transform(lambda c: c.rolling(3, center=True, min_periods=1).mean() >= 0.5)
    return table


def lines_at(t: float, lanes: pd.DataFrame, tracks: pd.DataFrame, calib: dict | None = None) -> tuple[list[dict], float | None]:
    """Lines to draw at time t (interpolated between the samples around it) and the heading column to draw them with."""
    lines, u_foe, _ = lines_near(t, lanes, tracks, calib)
    return lines, u_foe


def lines_near(t: float, lanes: pd.DataFrame, tracks: pd.DataFrame,
               calib: dict | None = None) -> tuple[list[dict], float | None, bool]:
    """lines_at, and whether t is at a junction (no lane lines there)."""
    times = lanes.time_s.to_numpy()
    after = int(np.searchsorted(times, t))
    before = after - 1
    candidates = []
    if before >= 0 and t - times[before] <= MAX_INTERP_GAP_S:
        candidates.append(before)
    if after < len(times) and times[after] - t <= MAX_INTERP_GAP_S:
        candidates.append(after)
    if not candidates:
        return [], None, False
    if len(candidates) == 2:
        w = (t - times[before]) / max(times[after] - times[before], 1e-6)
    else:
        w = 0.0 if candidates[0] == before else 1.0
        if abs(times[candidates[0]] - t) > SAMPLE_HOLD_S:
            return [], None, False
    u_foe = float(np.interp(t, times[candidates], lanes.u_foe.to_numpy()[candidates])) if len(candidates) == 2 \
        else float(lanes.u_foe.iloc[candidates[0]])
    a = tracks[(tracks["sample"] == before) & tracks.confirmed] if before in candidates else tracks.iloc[:0]
    b = tracks[(tracks["sample"] == after) & tracks.confirmed] if after in candidates else tracks.iloc[:0]
    if (len(a) and a.junction.iloc[0]) or (len(b) and b.junction.iloc[0]):
        return [], u_foe, True  # crosswalk bars and turn paint at a junction are not lane structure
    out = []
    for line_id in set(a.line_id) | set(b.line_id):
        ra = a[a.line_id == line_id]
        rb = b[b.line_id == line_id]
        if len(ra) and len(rb):
            ra, rb = ra.iloc[0], rb.iloc[0]
            line = {k: (1 - w) * ra[k] + w * rb[k] for k in ("x", "shear", "z_near", "z_far")}
            line["line_id"] = int(line_id)
            line.update(solid=bool(ra.solid if w < 0.5 else rb.solid), yellow=bool(ra.yellow if w < 0.5 else rb.yellow),
                        both=True)
        else:  # seen on one side only: hold it briefly, do not invent motion
            only, when = (ra.iloc[0], times[before]) if len(ra) else (rb.iloc[0], times[after])
            if abs(t - when) > SAMPLE_HOLD_S:
                continue
            line = {k: only[k] for k in ("x", "shear", "z_near", "z_far", "solid", "yellow")}
            line["line_id"] = int(line_id)
            if calib is not None:  # where it is in the image, from the heading it is drawn with
                line["x"] = only["x"] + only["offset"] - calib["lateral_height_m"] * (u_foe - calib["width"] / 2) / (
                    calib["focal_px"] * calib["cam_height_m"] / REF_Z)
            line["both"] = False
        if abs(line["x"]) <= MAX_LINE_X:
            out.append(line)
    # one painted line can briefly carry two ids (found again under a new id after a missed sample); draw it once,
    # preferring lines seen on both sides of t
    kept: list[dict] = []
    for line in sorted(out, key=lambda ln: not ln["both"]):
        if all(abs(line["x"] - k["x"]) >= DUPLICATE_M for k in kept):
            kept.append(line)
    out = kept
    left = sorted([ln for ln in out if ln["x"] < 0], key=lambda ln: -ln["x"])[:LINES_PER_SIDE]
    right = sorted([ln for ln in out if ln["x"] >= 0], key=lambda ln: ln["x"])[:LINES_PER_SIDE]
    return left + right, u_foe, False


def draw_lane_lines(image: np.ndarray, lines: list[dict], u_foe: float, calib: dict, in_bike_lane: bool,
                    vehicle_boxes: np.ndarray) -> None:
    """Lines no further ahead than their paint was seen, hidden behind vehicles."""
    layer = np.zeros_like(image)
    drawn = np.zeros(image.shape[:2], np.uint8)
    for line in lines:
        colour = LANE_COLOURS_BGR["yellow"] if line["yellow"] else LANE_COLOURS_BGR["solid" if line["solid"] else "broken"]
        if in_bike_lane and -2.4 <= line["x"] <= -0.3 and not line["yellow"]:
            colour = LANE_COLOURS_BGR["bike"]
        # A line seen over EXTEND_M or more is drawn from just ahead of the wheel (the paint nearest the bike is
        # motion-blurred and often not segmented) to where its paint was last seen; a shorter one only where seen.
        z1 = min(LINE_Z_MAX, line["z_far"])
        z0 = 2.5 if z1 - line["z_near"] >= EXTEND_M else max(2.5, line["z_near"])
        if z1 - z0 < 1.0:
            continue
        pts = project_line(line["x"], line["shear"], u_foe, calib, (z0, z1))
        cv2.polylines(layer, [pts], False, colour, 6 if line["solid"] else 3, cv2.LINE_AA)
        cv2.polylines(drawn, [pts], False, 255, 6 if line["solid"] else 3, cv2.LINE_AA)
    for x1, y1, x2, y2 in vehicle_boxes:
        drawn[max(0, int(y1)): int(y2), max(0, int(x1)): int(x2)] = 0
    np.copyto(image, layer, where=(drawn > 0)[..., None])


def find_paint(bright: np.ndarray, a: float, b: float, d_range: tuple[float, float], horizon: float,
               lateral_h: float, boxes: np.ndarray, contrast: float) -> tuple[float, float, float] | None:
    """The image line u = a + b*d (d: rows below the horizon) moved onto the paint near it in this frame.

    Along the predicted line, each sampled row is searched SNAP_SEARCH_M either side for a stripe brighter than the
    asphalt beside it by `contrast`, about as wide as lane paint at that distance (a kerb, a white car or a sunlit
    patch is wider). A straight line through the stripes found, robust to strays, is the line in this frame; it is
    returned as (a, b, lowest row found below the horizon), or None when too few rows found paint."""
    height, width = bright.shape
    found = []
    for d in np.linspace(d_range[0], d_range[1], SNAP_ROWS):
        v = int(round(horizon + d))
        if v < 1 or v >= height - 1:
            continue
        px_per_m = d / lateral_h
        u = a + b * d
        paint_px = max(2.0, PAINT_W_M * px_per_m)
        reach = max(6.0, SNAP_SEARCH_M * px_per_m)
        lo, hi = int(max(0, u - reach - 2 * paint_px)), int(min(width, u + reach + 2 * paint_px))
        if hi - lo < 12 or any(y1 <= v <= y2 and x1 < hi and lo < x2 for x1, y1, x2, y2 in boxes):
            continue  # off the picture, or hidden by a vehicle
        profile = bright[v - 1: v + 2, lo:hi].mean(axis=0)
        k = max(1, int(paint_px / 2))
        profile = np.convolve(profile, np.ones(k) / k, mode="same")
        background = float(np.median(profile))
        inside = np.flatnonzero(np.abs(np.arange(lo, hi) - u) <= reach)
        if not len(inside):
            continue
        i = int(inside[np.argmax(profile[inside])])
        rise = profile[i] - background
        if rise < contrast:
            continue
        half = background + rise / 2
        left, right = i, i
        while left > 0 and profile[left - 1] > half:
            left -= 1
        while right < len(profile) - 1 and profile[right + 1] > half:
            right += 1
        if not 0.3 * paint_px <= right - left + 1 <= 3 * paint_px + 3:
            continue
        found.append((d, lo + (left + right) / 2, rise))
    if len(found) < SNAP_MIN_POINTS:
        return None
    ds, us, weights = (np.array(c, float) for c in zip(*found))
    keep = np.ones(len(ds), bool)
    fit_a = fit_b = 0.0
    for _ in range(2):  # fit, drop the rows far off the fit, fit again
        fit_b, fit_a = np.polyfit(ds[keep], us[keep], 1, w=weights[keep])
        keep = np.abs(us - (fit_a + fit_b * ds)) <= 0.08 * ds / lateral_h + 3
        if keep.sum() < SNAP_MIN_POINTS:
            return None
    if np.ptp(ds[keep]) < 0.25 * (d_range[1] - d_range[0]):
        return None  # all in one short stretch: not enough to set the line's slant
    return float(fit_a), float(fit_b), float(ds[keep].max())


class LaneOverlay:
    """The lane lines of each frame of a clip, predicted from the samples around it (lines_near) and laid on the
    paint of the frame (find_paint).

    Each line is followed as an image line u = a + b*d (d = rows below the horizon), which is what a straight line on
    a flat road looks like. Its estimate starts from the samples' prediction, is moved part of the way to where this
    frame shows its paint, and is pulled back gently towards the prediction, so it neither jitters with the frame nor
    wanders off with it. A line is shown while paint is found under it in most recent frames, kept for KEEP_S after
    the samples stop reporting it as long as its paint is still there, and faded in and out."""

    def __init__(self, lanes: pd.DataFrame, calib: dict, fps: float) -> None:
        self.lanes, self.calib, self.fps = lanes, calib, fps
        self.tracks = lane_line_tracks(lanes, calib)
        self.fh = calib["focal_px"] * calib["cam_height_m"]
        self.lateral_h = calib["lateral_height_m"]
        self.horizon = calib["horizon_v"]
        self.state: dict[int, dict] = {}

    def image_line(self, x: float, shear: float, u_foe: float) -> tuple[float, float]:
        """(a, b) of the metric line X(Z) = x + shear*(Z - REF_Z) seen from the heading u_foe (as project_line)."""
        return u_foe + shear * self.fh / self.lateral_h, (x - shear * REF_Z) / self.lateral_h

    def update(self, image: np.ndarray, frame: int, boxes: np.ndarray) -> list[dict]:
        """Lines to draw on this frame, each with its image line, rows and opacity."""
        lines, u_foe, junction = lines_near(frame / self.fps, self.lanes, self.tracks, self.calib)
        bright = (image[..., 1].astype(np.float32) + image[..., 2]) / 2  # green + red: white and yellow paint both
        road = bright[int(self.horizon) + 20:]
        contrast = max(SNAP_MIN_CONTRAST, SNAP_CONTRAST_SHARE * float(np.median(road))) if road.size else SNAP_MIN_CONTRAST
        bottom = image.shape[0] - self.horizon - 2
        seen = set()
        for line in lines:
            assert u_foe is not None
            lid = line["line_id"]
            seen.add(lid)
            prediction = np.array(self.image_line(line["x"], line["shear"], u_foe))
            st = self.state.get(lid)
            if st is None or frame - st["frame"] > 2:
                st = self.state[lid] = {"ab": prediction, "support": 0.5, "alpha": 0.0, "frame": frame}
            else:  # its own estimate, held near the samples' prediction
                st["ab"] = 0.85 * st["ab"] + 0.15 * prediction
            z1 = min(LINE_Z_MAX, line["z_far"])
            z0 = 2.5 if z1 - line["z_near"] >= EXTEND_M else max(2.5, line["z_near"])
            st.update(x=line["x"], solid=line["solid"], yellow=line["yellow"], sample_frame=frame,
                      d_far=self.fh / z1, d_near=self.fh / z0)
        step = 1.0 / max(1.0, FADE_S * self.fps)
        for lid, st in list(self.state.items()):
            if junction or (lid not in seen and frame - st["sample_frame"] > KEEP_S * self.fps):
                st["support"] = 0.0  # at a junction, or no longer reported: fade out
            elif st["d_near"] - st["d_far"] > 10:
                found = find_paint(bright, float(st["ab"][0]), float(st["ab"][1]), (st["d_far"], bottom),
                                   self.horizon, self.lateral_h, boxes, contrast)
                if found is not None:
                    st["ab"] = st["ab"] + SNAP_GAIN * (np.array(found[:2]) - st["ab"])
                    st["d_near"] = max(st["d_near"], min(found[2], bottom))
                st["support"] = SUPPORT_DECAY * st["support"] + (1 - SUPPORT_DECAY) * (found is not None)
            target = 1.0 if st["support"] >= SHOW_SUPPORT else 0.0
            st["alpha"] = float(np.clip(st["alpha"] + np.sign(target - st["alpha"]) * step, 0.0, 1.0))
            st["frame"] = frame
            if st["alpha"] == 0.0 and target == 0.0:
                del self.state[lid]
        shown = sorted((st for st in self.state.values() if st["alpha"] > 0), key=lambda st: -st["support"])
        kept: list[dict] = []  # two ids that settled on one painted line are drawn once
        for st in shown:
            d_mid = (st["d_far"] + st["d_near"]) / 2
            u_mid = st["ab"][0] + st["ab"][1] * d_mid
            if all(abs(u_mid - (k["ab"][0] + k["ab"][1] * d_mid)) > 0.25 * d_mid / self.lateral_h for k in kept):
                kept.append(st)
        return kept

    def draw(self, image: np.ndarray, frame: int, in_bike_lane: bool, boxes: np.ndarray) -> None:
        """The lines as strips of paint in perspective, fading out towards where their paint was last seen, behind
        the vehicles."""
        lines = self.update(image, frame, boxes)
        if not lines:
            return
        height = image.shape[0]
        top = int(max(0, self.horizon + min(st["d_far"] for st in lines)))
        layer = np.zeros((height - top, image.shape[1], 3), np.uint8)
        alpha = np.zeros((height - top, image.shape[1]), np.float32)
        for st in lines:
            colour = LANE_COLOURS_BGR["yellow"] if st["yellow"] else LANE_COLOURS_BGR["solid" if st["solid"] else "broken"]
            if in_bike_lane and -2.4 <= st["x"] <= -0.3 and not st["yellow"]:
                colour = LANE_COLOURS_BGR["bike"]
            opacity = st["alpha"] * (0.85 if st["solid"] else 0.65)
            ds = np.linspace(st["d_far"], st["d_near"], 24)
            us = st["ab"][0] + st["ab"][1] * ds
            half = np.maximum(1.5, DRAW_W_M / 2 * ds / self.lateral_h)
            fade = np.clip((ds - st["d_far"]) / max(1.0, 0.35 * (st["d_near"] - st["d_far"])), 0.0, 1.0)
            vs = self.horizon + ds - top
            for i in range(len(ds) - 1):
                quad = np.array([[us[i] - half[i], vs[i]], [us[i] + half[i], vs[i]],
                                 [us[i + 1] + half[i + 1], vs[i + 1]], [us[i + 1] - half[i + 1], vs[i + 1]]], np.int32)
                cv2.fillConvexPoly(layer, quad, colour, cv2.LINE_AA)
                cv2.fillConvexPoly(alpha, quad, float(opacity * (fade[i] + fade[i + 1]) / 2), cv2.LINE_AA)
        for x1, y1, x2, y2 in boxes:
            alpha[max(0, int(y1) - top): max(0, int(y2) - top), max(0, int(x1)): max(0, int(x2))] = 0
        region = image[top:].astype(np.float32)
        a = alpha[..., None]
        image[top:] = (region * (1 - a) + layer.astype(np.float32) * a).astype(np.uint8)


# --------------------------------------------------------------------------------------------------------------------
# signs and signals: boxes interpolated between samples


def static_box(b0: np.ndarray, b1: np.ndarray, w: float) -> np.ndarray:
    """Box of a static object at fraction w of the way from box b0 to box b1 (w outside 0-1 extrapolates).

    Seen from the moving bike, a sign or signal head grows about a fixed point - the focus of expansion - by the same
    factor k = h1/h0 for its size and its offset from that point, so in between it is scaled by k**w about that point.
    A close sign doubles its size in half a second, which straight-line interpolation and holding the last box both
    get badly wrong. With no growth the box is interpolated linearly."""
    k = (b1[3] - b1[1]) / max(b0[3] - b0[1], 1e-6)
    if abs(k - 1) < 0.03:
        return b0 + w * (b1 - b0)
    c0, c1 = np.array([b0[0] + b0[2], b0[1] + b0[3]]) / 2, np.array([b1[0] + b1[2], b1[1] + b1[3]]) / 2
    centre = (c1 - k * c0) / (1 - k)
    scale = k ** w
    return np.concatenate([centre + scale * (b0[:2] - centre), centre + scale * (b0[2:] - centre)])


class ObjectBoxes:
    """Boxes of every light / sign around a frame: interpolated between the samples before and after it (static_box),
    and extrapolated for up to SAMPLE_HOLD_S before the first and after the last sample from the two nearest ones."""

    def __init__(self, objects: pd.DataFrame, fps: float, width: float, height: float) -> None:
        self.fps, self.width, self.height = fps, width, height
        self.objects = []
        for _, g in objects.sort_values("frame").groupby("object_id"):
            self.objects.append((g.frame.to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy(float),
                                 [row for _, row in g.iterrows()]))

    def __call__(self, frame: int) -> list[tuple[pd.Series, np.ndarray]]:
        out = []
        reach = MAX_INTERP_GAP_S * self.fps
        for frames, boxes, rows in self.objects:
            if frame < frames[0] - reach or frame > frames[-1] + reach:
                continue
            after = int(np.searchsorted(frames, frame))
            before = after - 1 if after == len(frames) or frames[after] != frame else after
            if 0 <= before and after < len(frames) and frames[after] - frames[before] <= reach:
                span = frames[after] - frames[before]
                w = 0.0 if span == 0 else (frame - frames[before]) / span
                out.append((rows[before] if w < 0.5 else rows[after], static_box(boxes[before], boxes[after], w)))
                continue
            near = [i for i in (before, after) if 0 <= i < len(frames)]
            nearest = min(near, key=lambda i: abs(frames[i] - frame))
            if abs(frames[nearest] - frame) > SAMPLE_HOLD_S * self.fps:
                continue
            box = boxes[nearest]
            pair = (nearest - 1, nearest) if nearest == len(frames) - 1 else (nearest, nearest + 1)
            if len(frames) >= 2 and frames[pair[1]] - frames[pair[0]] <= reach:
                a, b = pair
                box = static_box(boxes[a], boxes[b], (frame - frames[a]) / (frames[b] - frames[a]))
            if box[2] <= 0 or box[0] >= self.width or box[3] <= 0 or box[1] >= self.height:
                continue  # extrapolated out of the picture
            out.append((rows[nearest], box))
        return out


def readable(text: str) -> bool:
    """Text worth printing: at least one run of three letters or a time of day, not an OCR fragment ("-", "1", "BI")."""
    return bool(re.search(r"[A-Za-z]{3}|\d\s?[AP]M", text))


def sign_label(o: pd.Series) -> str:
    """The sign's type, and what it says where that is the point of the sign (names, directions, parking times)."""
    text = str(o.get("sign_text") or "")
    limit = o.get("speed_limit")
    if o.category == "speed_limit":
        return f"SPEED LIMIT {int(limit)}" if isinstance(limit, (int, float)) and np.isfinite(limit) else "speed limit"
    if o.category in ("street_name", "guide") and readable(text):
        return text[:28]
    name = SIGN_NAMES.get(o.category, o.category.replace("_", " "))
    if o.category in ("parking", "lane_use", "bus_stop") and readable(text) and len(text) <= 24:
        return f"{name}: {text}"
    return name


def draw_objects(image: np.ndarray, objects: list[tuple[pd.Series, np.ndarray]],
                 taken: list[tuple[int, int, int, int]] | None = None) -> None:
    """Signal heads with a read lamp state and signs with a known type get a box and a label; unlit or side-on heads
    and untyped signs only a thin box, so a junction full of heads does not bury the picture in labels. Labels are
    placed so they do not overlap one another or those in `taken` (place_label)."""
    taken = [] if taken is None else taken
    for o, (x1, y1, x2, y2) in sorted(objects, key=lambda item: -(item[1][3] - item[1][1])):  # largest first
        if o.kind == "light":
            lit = o.light_state in ("red", "yellow", "green")
            colour = LIGHT_COLOURS_BGR.get(o.light_state or "off", LIGHT_COLOURS_BGR["off"])
            label = f"signal {o.light_state}" if lit else ""
        elif o.kind == "sign":
            colour = SIGN_COLOURS_BGR.get(o.category, SIGN_COLOUR_BGR)
            label = sign_label(o) if o.category != "other_sign" else ""
        else:
            continue
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, colour, 3 if label else 1)
        if not label or y2 - y1 < 14:
            continue
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        spot = place_label(taken, p1[0] - 2, p1[1] - 2, p2[1], tw + 6, th + 10, image.shape[1], 60)
        if spot is None:
            continue
        lx, ly = spot
        cv2.rectangle(image, (lx, ly), (lx + tw + 6, ly + th + 10), (20, 20, 20), -1)
        cv2.putText(image, label, (lx + 3, ly + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)


def smoothed_scene(scene: pd.Series, window_s: int = SCENE_SMOOTH_S) -> pd.Series:
    """Per second, the most common riding context over window_s seconds around it (ties: the second's own)."""
    codes, names = pd.factorize(scene.astype(str))
    out = []
    for i in range(len(codes)):
        window = codes[max(0, i - window_s // 2): i + window_s // 2 + 1]
        counts = np.bincount(window, minlength=len(names))
        best = np.flatnonzero(counts == counts.max())
        out.append(names[codes[i] if codes[i] in best else best[0]])
    return pd.Series(out, index=scene.index)


def fill_box_gaps(kin: pd.DataFrame, stride: int, max_gap: int = MAX_BOX_GAP, step: int | None = None) -> pd.DataFrame:
    """Rows interpolated between a track's boxes that are at most `max_gap` processed frames apart, at every `step`-th
    frame: by default the processed frames the detector missed (it lost a close vehicle for a frame or two), so its box
    does not blink out of the clip; with step=1 also every source frame in between, for a clip played at the video's
    own frame rate. Other columns are the earlier row's."""
    step = step or stride
    moving = [c for c in ("x1", "y1", "x2", "y2", "box_h", "z_m", "time_s") if c in kin]
    added = []
    for _, t in kin.sort_values("frame").groupby("track_id"):
        f = t.frame.to_numpy()
        gaps = np.diff(f)
        spans = np.flatnonzero((gaps > step) & (gaps <= stride * (max_gap + 1)))
        if not len(spans):
            continue
        new = np.concatenate([np.arange(f[i] + step, f[i + 1], step) for i in spans])
        rows = t.iloc[np.searchsorted(f, new) - 1].copy()
        rows["frame"] = new
        for c in moving:
            rows[c] = np.interp(new, f, t[c].to_numpy(float))
        added.append(rows)
    return pd.concat([kin, *added], ignore_index=True) if added else kin


def known_plates(analysis_dir: Path, kin: pd.DataFrame) -> dict[int, str]:
    """Track id -> plate text from plates.parquet, shared by all tracks of one vehicle."""
    path = analysis_dir / "plates.parquet"
    if not path.exists():
        return {}
    from bikesafe.plates import plates_on_pieces, vehicle_plates

    plates = pd.read_parquet(path)
    if "plate_text" not in plates:  # written by the earlier hashed audit
        return {}
    plates = plates_on_pieces(plates, kin)
    vehicle_of = kin.groupby("track_id").vehicle_id.first().astype(int).to_dict()
    by_vehicle = vehicle_plates(plates, vehicle_of)
    return {int(tid): str(by_vehicle[vid]) for tid, vid in vehicle_of.items() if vid in by_vehicle.index}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--infrastructure", type=Path, default=None,
                        help="Folder with infrastructure.parquet / lanes.parquet (default: the video's analysis folder)")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results" / "corpus")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None, help="Output width in px (default: the video's own)")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality: lower is sharper and larger (18-28)")
    parser.add_argument("--max-mbps", type=float, default=None, help="Bit-rate cap, for clips played from a web page")
    parser.add_argument("--min-box-h", type=float, default=18)
    parser.add_argument("--no-infrastructure", action="store_true", help="Do not draw lane lines, signals and signs")
    parser.add_argument("--no-vehicles", action="store_true", help="Draw only the road infrastructure and the banner")
    parser.add_argument("--no-plates", action="store_true", help="Do not box and read licence plates")
    args = parser.parse_args()

    analysis_dir = args.analysis / args.video
    infra_dir = args.infrastructure / args.video if args.infrastructure else analysis_dir
    calib = read_json(analysis_dir / "calibration.json")
    fps = calib["source_fps"]
    kin = pd.read_parquet(analysis_dir / "kinematics.parquet")
    tracks = pd.read_csv(args.corpus / args.video / "tracks_typed.csv")
    line = pd.read_csv(args.corpus / args.video / "timeline_1s.csv").set_index("second")
    shown_scene = smoothed_scene(line.scene)
    gps_path = args.corpus / args.video / "gps_1s.csv"
    gps_speed = (pd.read_csv(gps_path).set_index("second").gps_speed_mps.rolling(3, center=True, min_periods=1).median()
                 if gps_path.exists() else pd.Series(dtype=float))
    if "vehicle_relation" in tracks:  # one colour per physical vehicle
        tracks = tracks.assign(relation=tracks.vehicle_relation)
        proba = [c for c in tracks if c.startswith("proba_")]
        if proba and "vehicle_id" in tracks:  # how sure the detection-weighted vote of the vehicle's tracks is
            votes = tracks[proba].mul(tracks.n_det, axis=0).groupby(tracks.vehicle_id).sum()
            confidence = votes.max(axis=1) / votes.sum(axis=1).replace(0, np.nan)
            tracks["relation_confidence"] = tracks.vehicle_id.map(confidence)
    extra = [c for c in ("n_det",) if c in tracks]
    if "vehicle_id" not in kin:
        kin["vehicle_id"] = kin.track_id
    plates = {} if args.no_plates else known_plates(analysis_dir, kin)
    kin = kin.merge(tracks[["track_id", "relation", "relation_confidence", *extra]], on="track_id", how="left")
    end_s = args.start_s + args.duration_s
    stride = int(calib.get("stride", 1))
    kin = kin[(kin.time_s >= args.start_s - 1) & (kin.time_s < end_s + 1)]
    detected = set(kin.frame)
    # the detector ran on every stride-th frame; the clip has every frame, the boxes in between interpolated
    kin = fill_box_gaps(kin, stride, step=1)
    kin = kin[(kin.time_s >= args.start_s) & (kin.time_s < end_s)]
    frames = kin.groupby("frame")
    present = set(frames.groups)

    lane_overlay = object_boxes = None
    lanes_path = infra_dir / "lanes.parquet"
    if lanes_path.exists() and not args.no_infrastructure:
        lanes = pd.read_parquet(lanes_path)
        lanes = lanes[(lanes.time_s >= args.start_s - 2) & (lanes.time_s < end_s + 2)].reset_index(drop=True)
        lane_overlay = LaneOverlay(lanes, calib, fps) if len(lanes) else None
        objects = pd.read_parquet(infra_dir / "infrastructure.parquet")
        signs_csv = args.corpus / args.video / "signs.csv"
        if signs_csv.exists():  # its sign text has partial street names completed
            text = pd.read_csv(signs_csv, keep_default_na=False).set_index("object_id").sign_text.astype(str)
            objects["sign_text"] = objects.object_id.map(text).fillna(objects.sign_text)
        object_boxes = ObjectBoxes(objects[(objects.time_s >= args.start_s - 2) & (objects.time_s < end_s + 2)
                                           & objects.kind.isin(["light", "sign"])], fps, calib["width"], calib["height"])

    out = args.out or ROOT / "demos" / f"typology_{args.video}_{int(args.start_s)}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    width = args.width or calib["width"]
    scale = width / calib["width"]
    size = (width, int(round(calib["height"] * scale)) // 2 * 2)
    writer = VideoOut(out, fps, size, crf=args.crf, max_mbps=args.max_mbps)
    cap = open_video(args.videos / f"{args.video}.mp4")
    first = int(np.ceil(args.start_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    annotator = None
    if not args.no_plates:
        from bikesafe.plates import PlateAnnotator

        annotator = PlateAnnotator(plates)
    # every frame of the window, with or without vehicles, so an empty road does not cut time from the clip
    for target in range(first, int(np.ceil(end_s * fps))):
        ok, image = cap.read()
        if not ok:
            break
        in_frame = frames.get_group(target) if target in present else kin.iloc[:0]
        rows = in_frame if not args.no_vehicles else in_frame.iloc[:0]
        if "n_det" in rows:  # a vehicle boxed twice is drawn once, with its longest track's box
            rows = rows.sort_values("n_det", ascending=False).drop_duplicates("vehicle_id")
        rows = rows[(rows.box_h >= args.min_box_h) & rows.relation.apply(lambda r: isinstance(r, str))]
        second = int(target / fps)
        info = line.loc[second] if second in line.index else None
        scene = shown_scene.get(second) if info is not None else None
        in_bike_lane = scene == "bike_lane"  # the riding context, not the raw paint call
        if lane_overlay is not None:
            lane_overlay.draw(image, target, in_bike_lane, in_frame[["x1", "y1", "x2", "y2"]].to_numpy(float))
        taken: list[tuple[int, int, int, int]] = []
        if object_boxes is not None:
            draw_objects(image, object_boxes(target), taken)
        if annotator is not None and len(rows):  # on the full-resolution frame, so plates are large enough to read
            annotator(image, rows[["x1", "y1", "x2", "y2"]].to_numpy(float), rows.track_id.tolist(), scale,
                      find=target in detected)
        for r in rows.sort_values("z_m").itertuples():  # nearest first: their labels get the place above the box
            colour = RELATION_COLOURS_BGR.get(r.relation, (255, 255, 255))
            far = not r.z_m <= LABEL_MAX_Z
            thick = 1 if far else 4 if r.relation in ("ego_lane", "adjacent_same", "oncoming") else 2
            cv2.rectangle(image, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, thick)
            if r.box_h >= 40 and not far:
                unsure = "?" if r.relation_confidence < UNSURE_CONFIDENCE else ""
                label = f"{BOX_LABELS[r.relation]}{unsure} {r.z_m:.0f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                spot = place_label(taken, int(r.x1), int(r.y1), int(r.y2), tw + 8, th + 10, image.shape[1], 56, steps=1)
                if spot is None:
                    continue  # nearer vehicles took the room around it
                lx, ly = spot
                cv2.rectangle(image, (lx, ly), (lx + tw + 8, ly + th + 10), colour, -1)
                cv2.putText(image, label, (lx + 4, ly + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2, cv2.LINE_AA)
        clock = target / fps
        banner = f"{int(clock // 60)}:{clock % 60:04.1f}"
        if info is not None:
            gps = float(gps_speed.get(second, np.nan))
            speed = gps if np.isfinite(gps) else max(0.0, float(info.ego_speed_mps))
            banner += (f" | {str(scene).replace('_', ' ')} | {speed:.1f} m/s ({speed * 2.237:.0f} mph"
                       f"{' GPS' if np.isfinite(gps) else ''})")
            if "in_bike_lane" in line:
                state = info.get("traffic_light_state")
                if isinstance(state, str):
                    banner += f" | signal {state}"
                limit = info.get("speed_limit_posted")
                if limit is not None and pd.notna(limit):
                    banner += f" | limit {int(limit)} mph"
        cv2.rectangle(image, (0, 0), (calib["width"], 54), (20, 20, 20), -1)
        cv2.putText(image, banner, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
        if not args.no_vehicles:
            draw_legend(image, lanes=lane_overlay is not None)
        writer.write(image if size == (image.shape[1], image.shape[0]) else
                     cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    writer.release()
    cap.release()
    plates_note = "" if annotator is None else f"; {annotator.plates_drawn} plate boxes drawn"
    print(f"wrote {out}{plates_note}")


if __name__ == "__main__":
    main()
