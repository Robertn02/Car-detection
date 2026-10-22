"""Render a typology overlay clip: boxes coloured by relation to the rider, scene and ego-speed banner, legend.

When bikesafe.infrastructure has run, the clip also shows the painted lane lines it found (green when the rider is in
a bike lane), traffic lights with their state, and traffic signs with their type and text ("SPEED LIMIT 35",
"Hoover St"). Licence plates are boxed and read out (bikesafe.plates: the track's consensus reading when
bikesafe.plates has run, otherwise a confident reading of the frame; --no-plates to switch off). A vehicle boxed twice
by the detector (car + truck) is drawn once, in the colour of its vehicle-level relation.

Lane lines and road furniture are measured on sampled frames (2 per second). In between, a sign's box and a lane
line's position are interpolated from the samples before and after, and a lane line is drawn only where it was found
on two of three consecutive samples, no further ahead than its paint was seen, not across a junction's crosswalk, and
behind the vehicles - so the overlay does not flicker or sweep across the picture when one sample is off.

The clip is written as H.264 when imageio-ffmpeg is installed (bikesafe.common.VideoOut), else as MPEG-4.

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

LABELS = RELATION_TEXT
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


def draw_legend(frame: np.ndarray) -> None:
    x, y = 16, frame.shape[0] - 16 - 30 * len(RELATIONS)
    cv2.rectangle(frame, (x - 8, y - 26), (x + 300, frame.shape[0] - 8), (20, 20, 20), -1)
    for i, rel in enumerate(RELATIONS):
        cy = y + 30 * i
        cv2.rectangle(frame, (x, cy - 14), (x + 22, cy + 6), RELATION_COLOURS_BGR[rel], -1)
        cv2.putText(frame, LABELS[rel], (x + 32, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)


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
    times = lanes.time_s.to_numpy()
    after = int(np.searchsorted(times, t))
    before = after - 1
    candidates = []
    if before >= 0 and t - times[before] <= MAX_INTERP_GAP_S:
        candidates.append(before)
    if after < len(times) and times[after] - t <= MAX_INTERP_GAP_S:
        candidates.append(after)
    if not candidates:
        return [], None
    if len(candidates) == 2:
        w = (t - times[before]) / max(times[after] - times[before], 1e-6)
    else:
        w = 0.0 if candidates[0] == before else 1.0
        if abs(times[candidates[0]] - t) > SAMPLE_HOLD_S:
            return [], None
    u_foe = float(np.interp(t, times[candidates], lanes.u_foe.to_numpy()[candidates])) if len(candidates) == 2 \
        else float(lanes.u_foe.iloc[candidates[0]])
    a = tracks[(tracks["sample"] == before) & tracks.confirmed] if before in candidates else tracks.iloc[:0]
    b = tracks[(tracks["sample"] == after) & tracks.confirmed] if after in candidates else tracks.iloc[:0]
    if (len(a) and a.junction.iloc[0]) or (len(b) and b.junction.iloc[0]):
        return [], u_foe  # crosswalk bars and turn paint at a junction are not lane structure
    out = []
    for line_id in set(a.line_id) | set(b.line_id):
        ra = a[a.line_id == line_id]
        rb = b[b.line_id == line_id]
        if len(ra) and len(rb):
            ra, rb = ra.iloc[0], rb.iloc[0]
            line = {k: (1 - w) * ra[k] + w * rb[k] for k in ("x", "shear", "z_near", "z_far")}
            line.update(solid=bool(ra.solid if w < 0.5 else rb.solid), yellow=bool(ra.yellow if w < 0.5 else rb.yellow),
                        both=True)
        else:  # seen on one side only: hold it briefly, do not invent motion
            only, when = (ra.iloc[0], times[before]) if len(ra) else (rb.iloc[0], times[after])
            if abs(t - when) > SAMPLE_HOLD_S:
                continue
            line = {k: only[k] for k in ("x", "shear", "z_near", "z_far", "solid", "yellow")}
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
    return left + right, u_foe


def draw_lane_lines(image: np.ndarray, lines: list[dict], u_foe: float, calib: dict, in_bike_lane: bool,
                    vehicle_boxes: np.ndarray) -> None:
    """Lines no further ahead than their paint was seen, hidden behind vehicles."""
    layer = np.zeros_like(image)
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
    drawn = layer.any(axis=2)
    for x1, y1, x2, y2 in vehicle_boxes:
        drawn[max(0, int(y1)): int(y2), max(0, int(x1)): int(x2)] = False
    image[drawn] = layer[drawn]


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


def draw_objects(image: np.ndarray, objects: list[tuple[pd.Series, np.ndarray]]) -> None:
    """Signal heads with a read lamp state and signs with a known type get a box and a label; unlit or side-on heads
    and untyped signs only a thin box, so a junction full of heads does not bury the picture in labels. Labels are
    placed so they do not overlap one another (the later one moves up)."""
    taken: list[tuple[int, int, int, int]] = []
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
        top = max(60 + th, p1[1] - 6)
        for _ in range(6):  # move up past labels already placed
            box = (p1[0] - 2, top - th - 6, p1[0] + tw + 4, top + 4)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken):
                break
            top = max(60 + th, top - th - 10)
        taken.append((p1[0] - 2, top - th - 6, p1[0] + tw + 4, top + 4))
        cv2.rectangle(image, (p1[0] - 2, top - th - 6), (p1[0] + tw + 4, top + 4), (20, 20, 20), -1)
        cv2.putText(image, label, (p1[0] + 1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)


def fill_box_gaps(kin: pd.DataFrame, stride: int, max_gap: int = MAX_BOX_GAP) -> pd.DataFrame:
    """Rows interpolated where a track misses up to `max_gap` processed frames (the detector lost a close vehicle for
    a frame or two), so its box does not blink out of the clip. Other columns are the earlier row's."""
    moving = [c for c in ("x1", "y1", "x2", "y2", "box_h", "z_m", "time_s") if c in kin]
    added = []
    for _, t in kin.sort_values("frame").groupby("track_id"):
        f = t.frame.to_numpy()
        steps = np.diff(f)
        for i in np.flatnonzero((steps > stride) & (steps <= stride * (max_gap + 1))):
            a, b = t.iloc[i], t.iloc[i + 1]
            for frame in range(int(f[i]) + stride, int(f[i + 1]), stride):
                w = (frame - f[i]) / (f[i + 1] - f[i])
                row = a.copy()
                row["frame"] = frame
                for c in moving:
                    row[c] = a[c] + w * (b[c] - a[c])
                added.append(row)
    return pd.concat([kin, pd.DataFrame(added)], ignore_index=True) if added else kin


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
    parser.add_argument("--width", type=int, default=1280)
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
    origin = int(kin.frame.min()) % stride if len(kin) else 0
    kin = kin[(kin.time_s >= args.start_s - 1) & (kin.time_s < end_s + 1)]
    kin = fill_box_gaps(kin, stride)
    kin = kin[(kin.time_s >= args.start_s) & (kin.time_s < end_s)]
    frames = kin.groupby("frame")
    present = set(frames.groups)
    # every processed frame of the window, with or without vehicles, so an empty road does not cut time from the clip
    first = origin + stride * int(np.ceil((args.start_s * fps - origin) / stride))
    processed = list(range(first, int(np.ceil(end_s * fps)), stride))

    lanes = object_boxes = line_tracks = None
    lanes_path = infra_dir / "lanes.parquet"
    if lanes_path.exists() and not args.no_infrastructure:
        lanes = pd.read_parquet(lanes_path)
        lanes = lanes[(lanes.time_s >= args.start_s - 2) & (lanes.time_s < end_s + 2)].reset_index(drop=True)
        line_tracks = lane_line_tracks(lanes, calib)
        objects = pd.read_parquet(infra_dir / "infrastructure.parquet")
        object_boxes = ObjectBoxes(objects[(objects.time_s >= args.start_s - 2) & (objects.time_s < end_s + 2)
                                           & objects.kind.isin(["light", "sign"])], fps, calib["width"], calib["height"])

    out = args.out or ROOT / "demos" / f"typology_{args.video}_{int(args.start_s)}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = args.width / calib["width"]
    size = (args.width, int(round(calib["height"] * scale)))
    writer = VideoOut(out, calib["fps"], size)
    cap = open_video(args.videos / f"{args.video}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_s * fps))
    idx = int(args.start_s * fps)
    annotator = None
    if not args.no_plates:
        from bikesafe.plates import PlateAnnotator

        annotator = PlateAnnotator(plates)
    targets = iter(processed)
    target = next(targets, None)
    while target is not None:
        ok = cap.grab()
        if not ok:
            break
        if idx < target:
            idx += 1
            continue
        _, image = cap.retrieve()
        idx += 1
        in_frame = frames.get_group(target) if target in present else kin.iloc[:0]
        rows = in_frame if not args.no_vehicles else in_frame.iloc[:0]
        if "n_det" in rows:  # a vehicle boxed twice is drawn once, with its longest track's box
            rows = rows.sort_values("n_det", ascending=False).drop_duplicates("vehicle_id")
        rows = rows[(rows.box_h >= args.min_box_h) & rows.relation.apply(lambda r: isinstance(r, str))]
        second = int(target / fps)
        info = line.loc[second] if second in line.index else None
        in_bike_lane = bool(info is not None and info.scene == "bike_lane")  # the riding context, not the raw paint call
        if lanes is not None and line_tracks is not None and object_boxes is not None and len(lanes):
            drawn, u_foe = lines_at(target / fps, lanes, line_tracks, calib)
            if u_foe is not None:
                draw_lane_lines(image, drawn, u_foe, calib, in_bike_lane, in_frame[["x1", "y1", "x2", "y2"]].to_numpy(float))
            draw_objects(image, object_boxes(target))
        if annotator is not None and len(rows):  # on the full-resolution frame, so plates are large enough to read
            annotator(image, rows[["x1", "y1", "x2", "y2"]].to_numpy(float), rows.track_id.tolist(), scale)
        for r in rows.itertuples():
            colour = RELATION_COLOURS_BGR.get(r.relation, (255, 255, 255))
            far = not r.z_m <= LABEL_MAX_Z
            thick = 1 if far else 4 if r.relation in ("ego_lane", "adjacent_same", "oncoming") else 2
            cv2.rectangle(image, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, thick)
            if r.box_h >= 40 and not far:
                unsure = "?" if r.relation_confidence < UNSURE_CONFIDENCE else ""
                label = f"{LABELS[r.relation]}{unsure} {r.z_m:.0f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                top = max(0, int(r.y1) - th - 10)
                cv2.rectangle(image, (int(r.x1), top), (int(r.x1) + tw + 8, top + th + 10), colour, -1)
                cv2.putText(image, label, (int(r.x1) + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2, cv2.LINE_AA)
        banner = f"t={target / fps:7.1f}s"
        if info is not None:
            banner += f" | scene: {str(info.scene).replace('_', ' ')} | ego {info.ego_speed_mps:.1f} m/s"
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
            draw_legend(image)
        writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        target = next(targets, None)
    writer.release()
    cap.release()
    plates_note = "" if annotator is None else f"; {annotator.plates_drawn} plate boxes drawn"
    print(f"wrote {out}{plates_note}")


if __name__ == "__main__":
    main()
