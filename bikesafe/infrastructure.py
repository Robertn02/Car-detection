"""Road infrastructure seen from the bike: traffic lights, traffic signs and painted bike lanes.

A sampled-frame pass (default 2 fps, like the depth pass). Every sample is looked at twice:

* **The camera image**, for lights and signs. An open-vocabulary detector (YOLOE-26 from ultralytics) is given a
  road-furniture vocabulary - traffic light, stop / yield / speed-limit / street-name / warning signs and so on - plus
  distractor classes (billboard, shop sign, licence plate, street lamp) that absorb look-alikes instead of letting them
  be forced into a sign class. A lit lamp's colour gives each signal head's state. The prompts' text features are
  cached in models/infrastructure_vocab.npz, so the text encoder is only needed to change the vocabulary.
* **A bird's-eye view of the road ahead**, warped with the same self-calibrated flat-road geometry the rest of the
  pipeline uses (bikesafe.geometry). In it lane lines run vertically and painted stencils look like upright icons,
  which the detector recognises far better than their foreshortened originals. It gives
    - every longitudinal paint line within 4.5 m of the rider's path: where it is, solid or broken, white or yellow,
    - bicycle stencils and their position relative to the rider,
    - the share of green paint in the rider's corridor.
  A painted bike lane shows up as a solid white line 0.3-2.4 m to the rider's left (broken near junctions), as a
  narrow lane between two lines, or as a stencil in the rider's path. Junctions interrupt the paint, so the per-sample
  evidence is smoothed over several seconds (per_second_summary).

Outputs in work/analysis/<video>/:
  infrastructure.parquet  one row per light / sign / stencil detection per sample, linked into object tracks
  lanes.parquet           one row per sample: lines found, the rider's lane edges, stencil and green-paint evidence

    python -m bikesafe.infrastructure work/perception/VID_... [--sample-fps 2] [--model models/yoloe-26l-seg.pt]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import ROOT, fp16_kwargs, is_cuda, open_video, read_json, resolve_video
from bikesafe.geometry import Calibration, ego_from_tracking, fit_horizon

# large: the medium model misses faint bicycle stencils in the bird's-eye view (demo clip: 0/3 vs 3/3 found)
DEFAULT_MODEL = str(ROOT / "models" / "yoloe-26l-seg.pt")  # downloaded there by ultralytics on first use
VOCAB_CACHE = ROOT / "models" / "infrastructure_vocab.npz"

# prompt -> (category, kind). "light" and "sign" count in the camera image, "marking" in the bird's-eye view;
# "distractor" classes count nowhere, they only absorb look-alikes.
VOCABULARY: dict[str, tuple[str, str]] = {
    "traffic light": ("traffic_light", "light"),
    "pedestrian crossing signal": ("pedestrian_signal", "light"),
    "stop sign": ("stop", "sign"),
    "yield sign": ("yield", "sign"),
    "speed limit sign": ("speed_limit", "sign"),
    "street name sign": ("street_name", "sign"),
    "one way sign": ("one_way", "sign"),
    "no parking sign": ("no_parking", "sign"),
    "no turn sign": ("no_turn", "sign"),
    "do not enter sign": ("do_not_enter", "sign"),
    "bike lane sign": ("bike_lane_sign", "sign"),
    "pedestrian crossing sign": ("pedestrian_crossing", "sign"),
    "yellow warning sign": ("warning", "sign"),
    "lane control sign": ("lane_control", "sign"),
    "green highway guide sign": ("guide", "sign"),
    "traffic sign": ("other_sign", "sign"),
    "bicycle symbol painted on the road": ("bike_stencil", "marking"),
    "arrow painted on the road": ("arrow", "distractor"),
    "manhole cover": ("manhole", "distractor"),
    "billboard": ("billboard", "distractor"),
    "store sign": ("store_sign", "distractor"),
    "license plate": ("license_plate", "distractor"),
    "street lamp": ("street_lamp", "distractor"),
    "car": ("car", "distractor"),
    "bus": ("bus", "distractor"),
    "truck": ("truck", "distractor"),
    "person": ("person", "distractor"),
    "bicycle": ("bicycle", "distractor"),
}
MIN_CONF = {"light": 0.25, "sign": 0.25, "marking": 0.08}
# stop signs are confirmed by a COCO-trained detector (class 11), which the repository already ships
STOP_VERIFIER, COCO_STOP_SIGN, STOP_CONFIRM_CONF = str(ROOT / "weights" / "yolo11m.pt"), 11, 0.4

# Bird's-eye view: X to the right of the rider's heading, Z ahead, flat road. Lines are judged 2.5-14 m ahead.
BEV_X, BEV_Z, BEV_RES = (-4.5, 4.5), (2.5, 18.0), 0.02
Z_BIN, X_BIN, REF_Z, LINE_Z_MAX = 0.5, 0.1, 5.0, 14.0
# Lateral drift per metre ahead: the lane's angle to the rider's heading, the heading estimate's error, and the fan
# a few pixels of horizon error give parallel lines (in proportion to their offset).
SHEARS = np.linspace(-0.35, 0.35, 29)
MIN_VISIBLE_BINS = 6  # a line needs at least 3 m of visible road to be judged
SOLID_COVER, LINE_COVER, MIN_SOLID_M, MAX_LINES, MIN_LINE_GAP = 0.6, 0.2, 5.0, 8, 0.35
PAINT_CONTRAST = (28, 14)  # white, yellow: grey levels above the local asphalt
YELLOW_HUE, YELLOW_MIN_SAT = (8, 32), 32  # OpenCV HLS
MAX_LINE_WIDTH, MIN_STROKE_LENGTH = 0.35, 0.5  # metres; lane lines are 0.1-0.3 m wide, dashes ~3 m long
CORRIDOR_X = 1.4  # half-width of the rider's own path for stencils and green paint (m)
JUNCTION_LINES = 6  # this many lines at once is a crosswalk or a junction's paint, not lane structure

# Lamp colour ranges in OpenCV hue (0-180). Lit reds read pink-magenta on this camera, LED greens cyan-green.
LAMP_HUES = {"red": [(0, 12), (140, 180)], "yellow": [(13, 35)], "green": [(40, 95)]}

# Per-second bike-lane decision: evidence averaged over a centred window, held through junctions without paint.
BIKE_LANE_WINDOW_S, BIKE_LANE_HOLD_S, BIKE_LANE_THRESHOLD = 7, 6, 0.35


# --------------------------------------------------------------------------------------------------------------------
# detector


def text_features(model, prompts: list[str], cache: Path = VOCAB_CACHE):
    """Raw text features (1, N, D) for the prompts, from the cache when it holds all of them for this text model.

    They are the same for every YOLOE-26 size; each model's head adapts them when the classes are set.
    """
    import torch

    text_model = str(getattr(model.model, "text_model", "unknown"))
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        names = [str(p) for p in stored["prompts"]]
        if str(stored["text_model"]) == text_model and all(p in names for p in prompts):
            feats = stored["features"].astype(np.float32)[[names.index(p) for p in prompts]]
            return torch.from_numpy(feats)[None]
    raw = model.model.get_text_pe(prompts, without_reprta=True).float().cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, prompts=np.array(prompts), features=raw[0].numpy().astype(np.float16), text_model=np.array(text_model))
    return raw


class InfrastructureDetector:
    """YOLOE with the road-furniture vocabulary; returns detections of the requested kinds."""

    def __init__(self, weights: str = DEFAULT_MODEL, device: str = "cpu", half: bool = False,
                 vocabulary: dict[str, tuple[str, str]] = VOCABULARY, verifier_weights: str = STOP_VERIFIER) -> None:
        import torch
        from ultralytics import YOLOE

        self.model = YOLOE(weights)
        self.prompts = list(vocabulary)
        raw = text_features(self.model, self.prompts)
        head = self.model.model.model[-1]
        with torch.inference_mode():
            embeddings = head.get_tpe(raw.to(next(head.parameters()).dtype))
        self.model.set_classes(self.prompts, embeddings)
        self.category = [vocabulary[p][0] for p in self.prompts]
        self.kind = [vocabulary[p][1] for p in self.prompts]
        self.device = device
        self.precision = fp16_kwargs(device) if half else {}
        self.verifier, self.verifier_weights = None, verifier_weights

    def confirm_stop_signs(self, image: np.ndarray, detections: list[dict], imgsz: int) -> None:
        """Keep the "stop" category only where a COCO-trained detector also sees a stop sign (IoU >= 0.3).

        The open vocabulary calls DO NOT ENTER, NO PARKING and red shop signs "stop sign" too (10 of 12 on the demo
        clips); COCO's stop-sign class rejected all of those and kept the real one facing the rider. Side-on stop signs
        for cross traffic are mostly dropped as well, which is what "stop signs the rider passed" should count.
        Unconfirmed candidates stay in the data as generic signs.
        """
        candidates = [d for d in detections if d["category"] == "stop"]
        if not candidates:
            return
        if self.verifier is None:
            from ultralytics import YOLO

            self.verifier = YOLO(self.verifier_weights)
        result = self.verifier.predict(image, imgsz=imgsz, conf=STOP_CONFIRM_CONF, classes=[COCO_STOP_SIGN],
                                       device=self.device, verbose=False, **self.precision)[0]
        confirmed = result.boxes.xyxy.tolist() if result.boxes is not None else []
        for d in candidates:
            box = np.array([d["x1"], d["y1"], d["x2"], d["y2"]])
            d["stop_confirmed"] = any(_iou(box, np.array(c)) >= 0.3 for c in confirmed)
            if not d["stop_confirmed"]:
                d["category"] = "other_sign"

    def __call__(self, image: np.ndarray, imgsz: int, kinds: tuple[str, ...]) -> list[dict]:
        conf = min(MIN_CONF[k] for k in kinds)
        result = self.model.predict(image, imgsz=imgsz, conf=conf, device=self.device, verbose=False,
                                    **self.precision)[0]
        found = []
        boxes = result.boxes
        if boxes is None or not len(boxes):
            return found
        for cls, score, (x1, y1, x2, y2) in zip(boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            kind = self.kind[cls]
            if kind in kinds and score >= MIN_CONF[kind]:
                found.append({"category": self.category[cls], "kind": kind, "conf": float(score),
                              "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
        return found


def light_state(crop: np.ndarray) -> tuple[str, float]:
    """State of a signal head from its lit lamp: red / yellow / green, or off when no lamp faces the camera.

    A lit lamp is a bright, saturated, roughly round blob inside the housing. Sky, walls and awnings behind a head
    seen side-on also fall in the colour ranges, but they run along the crop's top or bottom edge or form tall strips.
    """
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return "off", 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (val >= 120) & (sat >= 60)
    height = hue.shape[0]
    areas = {}
    for state, ranges in LAMP_HUES.items():
        colour = np.zeros(hue.shape, bool)
        for lo, hi in ranges:
            colour |= (hue >= lo) & (hue <= hi)
        n, _, stats, _ = cv2.connectedComponentsWithStats((colour & lit).astype(np.uint8), connectivity=8)
        area = 0
        for x, y, w, h, a in stats[1:n]:
            if y == 0 or y + h >= height or h > 0.5 * height:
                continue
            if not 0.4 <= w / h <= 2.5 or a < 0.35 * w * h:
                continue
            area += int(a)
        areas[state] = area
    state = max(areas, key=lambda k: areas[k])
    if areas[state] < max(2, int(0.003 * hue.size)):
        return "off", 0.0
    return state, areas[state] / max(1, sum(areas.values()))


# --------------------------------------------------------------------------------------------------------------------
# bird's-eye view and lane paint


class GroundView:
    """Flat-road bird's-eye view of the road ahead, from the self-calibrated camera geometry.

    Rows run from BEV_Z[1] (top) to BEV_Z[0] (bottom, just ahead of the wheel), columns from BEV_X[0] to BEV_X[1], at
    BEV_RES metres per pixel. A ground point at depth Z and lateral offset X projects to row v = v_h + f*h/Z and column
    u = u_foe + X*d/h_x (d = rows below the horizon), exactly as in bikesafe.geometry.
    """

    def __init__(self, calib: Calibration) -> None:
        self.calib = calib
        self.xs = np.arange(BEV_X[0], BEV_X[1] - 1e-9, BEV_RES) + BEV_RES / 2
        self.zs = np.arange(BEV_Z[1], BEV_Z[0] + 1e-9, -BEV_RES) - BEV_RES / 2
        grid_x, grid_z = np.meshgrid(self.xs, self.zs)
        self._rows_below = (calib.focal_px * calib.cam_height_m / grid_z).astype(np.float32)
        self._x_over_h = (grid_x / calib.lateral_height_m).astype(np.float32)
        self._map_y = (calib.horizon_v + self._rows_below).astype(np.float32)

    def warp(self, image: np.ndarray, u_foe: float, nearest: bool = False) -> np.ndarray:
        map_x = (u_foe + self._x_over_h * self._rows_below).astype(np.float32)
        return cv2.remap(image, map_x, self._map_y, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def ground(self, col: float, row: float) -> tuple[float, float]:
        """Lateral offset X and depth Z (m) of a bird's-eye pixel."""
        return BEV_X[0] + col * BEV_RES, BEV_Z[1] - row * BEV_RES


def road_mask(shape: tuple[int, int], horizon: float, boxes: np.ndarray) -> np.ndarray:
    """Image pixels that can show road paint: below the horizon and outside every detected vehicle (uint8 0/255)."""
    mask = np.zeros(shape, np.uint8)
    mask[int(max(0, horizon + 10)):, :] = 255
    for x1, y1, x2, y2 in boxes:
        mask[max(0, int(y1) - 4): int(y2) + 4, max(0, int(x1) - 4): int(x2) + 4] = 0
    return mask


def bev_paint(bev: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """White and yellow lane paint in the bird's-eye view.

    Paint is brighter than the asphalt beside it (lines run vertically here, so a wide horizontal opening estimates
    the local background), and it is a thin stroke along the road. Wider bright regions - kerb and gutter concrete,
    crosswalk bars, car bodies, sunlit patches - and short specks from cracks and shadow edges are removed.
    """
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    hue, lightness, saturation = hls[..., 0], hls[..., 1], hls[..., 2]
    blurred = cv2.GaussianBlur(lightness, (0, 0), 1.5)
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1)))
    contrast = cv2.subtract(blurred, background)
    # Faded yellow centre lines on this footage sit at hue ~15-17 with saturation ~35-70, while white paint in low sun
    # takes a warm hue but stays below ~30, so saturation splits them; bright white stays white.
    yellowish = (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1]) & (saturation >= YELLOW_MIN_SAT)
    yellow = valid & (contrast > PAINT_CONTRAST[1]) & yellowish
    white = valid & (contrast > PAINT_CONTRAST[0]) & ~yellowish
    paint = (white | yellow).astype(np.uint8)
    wide_px = int(round(MAX_LINE_WIDTH / BEV_RES)) + 1
    wide = cv2.morphologyEx(paint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (wide_px, 1)))
    thin = paint & (1 - wide)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    height, width = stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_WIDTH]
    keep = (height >= MIN_STROKE_LENGTH / BEV_RES) & (height >= 2 * width)
    keep[0] = False
    strokes = keep[labels]
    return white & strokes, yellow & strokes


def _shift_cols(a: np.ndarray, k: int) -> np.ndarray:
    """Shift a 1-D array left by k (right for negative k), filling with zeros instead of wrapping."""
    out = np.zeros_like(a)
    if k == 0:
        out[:] = a
    elif k > 0:
        out[:-k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


def lane_lines(white: np.ndarray, yellow: np.ndarray, valid: np.ndarray, zs: np.ndarray) -> list[dict]:
    """Longitudinal paint lines in the bird's-eye view, sorted by lateral position.

    The view is binned into 0.5 m (along) x 0.1 m (across) cells. For a candidate line the share of visible cells it
    passes through that hold paint is its coverage: ~1 for a solid line, ~0.25-0.5 for a broken one. Each line gets
    its own slant (SHEARS): the rider is rarely exactly parallel to the lane, and a few pixels of horizon error fan
    parallel lines out in proportion to their offset. Positions are reported at REF_Z metres ahead.
    """
    rows = np.flatnonzero(zs <= LINE_Z_MAX)
    rows_per_bin, cols_per_bin = int(round(Z_BIN / BEV_RES)), int(round(X_BIN / BEV_RES))
    n_zb, n_xb = len(rows) // rows_per_bin, white.shape[1] // cols_per_bin
    if n_zb < MIN_VISIBLE_BINS:
        return []
    sel = rows[len(rows) - n_zb * rows_per_bin:]
    paint = (white | yellow)[sel][:, : n_xb * cols_per_bin].reshape(n_zb, rows_per_bin, n_xb, cols_per_bin)
    yel = yellow[sel][:, : n_xb * cols_per_bin].reshape(n_zb, rows_per_bin, n_xb, cols_per_bin)
    vis = valid[sel][:, : n_xb * cols_per_bin].reshape(n_zb, rows_per_bin, n_xb, cols_per_bin)
    remaining = paint.any(axis=(1, 3))
    visible = vis.mean(axis=(1, 3)) >= 0.6
    paint_count, yellow_count = paint.sum(axis=(1, 3)), yel.sum(axis=(1, 3))
    z_centres = zs[sel].reshape(n_zb, rows_per_bin).mean(axis=1)
    shifts = [np.round(shear * (z_centres - REF_Z) / X_BIN).astype(int) for shear in SHEARS]
    vis_by_shear = [np.stack([_shift_cols(visible[z], s[z]) for z in range(n_zb)]) for s in shifts]
    n_visible = np.stack([v.sum(axis=0) for v in vis_by_shear])  # (shear, x)

    # Greedy extraction: take the line with the most paint along it, remove the paint it explains, repeat. Without
    # the removal, a slanted line also lends 30-60 % coverage to every candidate that crosses it at a shallow angle.
    lines = []
    for _ in range(MAX_LINES):
        support = np.zeros((len(SHEARS), n_xb))
        for k, s in enumerate(shifts):
            occ = np.stack([_shift_cols(remaining[z], s[z]) for z in range(n_zb)])
            occ = occ | np.pad(occ[:, 1:], ((0, 0), (0, 1))) | np.pad(occ[:, :-1], ((0, 0), (1, 0)))  # +-0.1 m
            support[k] = (occ & vis_by_shear[k]).sum(axis=0)
        cover = np.where(n_visible >= MIN_VISIBLE_BINS, support / np.maximum(n_visible, 1), 0.0)
        support[cover < LINE_COVER] = 0
        for line in lines:  # the rest of a stroke already taken is not a new line
            near = np.abs((np.arange(n_xb) + 0.5) * X_BIN + BEV_X[0] - line["x"]) < MIN_LINE_GAP
            support[:, near] = 0
        k, i = np.unravel_index(int(np.argmax(support)), support.shape)
        if support[k, i] <= 0:
            break
        cells = [(z, i + shifts[k][z] + c) for z in range(n_zb) for c in (-2, -1, 0, 1, 2)]
        cells = [(z, c) for z, c in cells if 0 <= c < n_xb]
        painted = sum(paint_count[z, c] for z, c in cells)
        yellow_px = sum(yellow_count[z, c] for z, c in cells)
        for z, c in cells:
            remaining[z, c] = False
        visible_m = float(n_visible[k, i] * Z_BIN)
        lines.append({
            "x": float(BEV_X[0] + (i + 0.5) * X_BIN), "shear": float(SHEARS[k]), "cover": float(cover[k, i]),
            "visible_m": visible_m, "solid": bool(cover[k, i] >= SOLID_COVER and visible_m >= MIN_SOLID_M),
            "yellow": float(yellow_px / painted) if painted else 0.0,
        })
    return sorted(lines, key=lambda line: line["x"])


def green_share(bev: np.ndarray, valid: np.ndarray, view: GroundView) -> float:
    """Share of the rider's corridor (|X| <= CORRIDOR_X, 2.5-10 m ahead) covered by green bike-lane paint."""
    cols = np.abs(view.xs) <= CORRIDOR_X
    rows = view.zs <= 10.0
    region = bev[np.ix_(rows, cols)]
    ok = valid[np.ix_(rows, cols)]
    if ok.sum() < 200:
        return float("nan")
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 60)
    return float((green & ok).sum() / ok.sum())


def vehicles_on_ground(boxes: np.ndarray, calib: Calibration, u_foe: float) -> tuple[np.ndarray, np.ndarray]:
    """Near-side lateral offset and depth of each vehicle whose wheels are visible (flat-road geometry)."""
    if not len(boxes):
        return np.zeros(0), np.zeros(0)
    x1, x2, y2 = boxes[:, 0], boxes[:, 2], boxes[:, 3]
    rows_below = y2 - calib.horizon_v
    ok = (rows_below > 4) & (y2 < calib.height - 3)
    inner = np.where(x1 > u_foe, x1, np.where(x2 < u_foe, x2, u_foe))
    x_inner = calib.lateral_height_m * (inner - u_foe) / np.maximum(rows_below, 1)
    z = calib.focal_px * calib.cam_height_m / np.maximum(rows_below, 1)
    return x_inner[ok], z[ok]


def lane_record(lines: list[dict], stencils: list[dict], green: float, vehicle_x: np.ndarray,
                vehicle_z: np.ndarray) -> dict:
    """The rider's lane from the lines beside it, plus the other bike-lane cues, as one row."""
    left = [line for line in lines if line["x"] <= -0.3]
    right = [line for line in lines if line["x"] >= 0.3]
    left_line = left[-1] if left else None
    right_line = right[0] if right else None
    # the lane line proper, even when a weaker edge (a car's shadow, a worn patch) sits between it and the rider
    solid_left = [line for line in left if line["solid"] and line["yellow"] < 0.5 and line["x"] >= -2.4]
    solid_line = max(solid_left, key=lambda line: line["cover"]) if solid_left else None
    near_right = vehicle_x[(vehicle_x > 0.3) & (vehicle_z < 15)]
    in_corridor = [s for s in stencils if abs(s["x"]) <= CORRIDOR_X]
    stencil = max(in_corridor, key=lambda s: s["conf"]) if in_corridor else None
    nan = float("nan")
    record = {
        "n_lines": len(lines),
        "left_x": left_line["x"] if left_line else nan,
        "left_solid": bool(left_line and left_line["solid"]),
        "left_cover": left_line["cover"] if left_line else nan,
        "left_yellow": left_line["yellow"] if left_line else nan,
        "right_x": right_line["x"] if right_line else nan,
        "right_solid": bool(right_line and right_line["solid"]),
        "right_cover": right_line["cover"] if right_line else nan,
        "right_yellow": right_line["yellow"] if right_line else nan,
        "lane_width": (right_line["x"] - left_line["x"]) if left_line and right_line else nan,
        "left_solid_white_x": solid_line["x"] if solid_line else nan,
        "right_vehicle_x": float(near_right.min()) if len(near_right) else nan,
        "stencil_conf": stencil["conf"] if stencil else 0.0,
        "stencil_x": stencil["x"] if stencil else nan,
        "stencil_z": stencil["z"] if stencil else nan,
        "green_share": green,
        # every line as x:solid:yellow:shear, x at REF_Z metres ahead and shear its lateral drift per metre
        "lines_json": ";".join(f"{line['x']:.2f}:{int(line['solid'])}:{line['yellow']:.2f}:{line['shear']:.3f}"
                               for line in lines),
    }
    record["bike_lane_evidence"] = bike_lane_evidence(record)
    return record


def bike_lane_evidence(r: dict) -> float:
    """Transparent per-sample evidence that the rider is in a painted bike lane, in [-1, 1]; NaN when the sample says
    nothing either way (no paint in view, or the clutter of a junction's crosswalk).

    For: a bicycle stencil in the rider's corridor (+0.9); a solid white line 0.3-2.4 m to the left (+0.7) - between
    same-direction traffic lanes US lines are broken, so a solid white line that close is a bike-lane or edge line;
    otherwise a narrow lane (1.0-2.8 m) between two lines with one solid (+0.6); a broken white line on the left with
    cars close on the right (+0.3: bike lanes turn broken before junctions, but so do traffic lanes); green paint
    (+0.5). Against: the rider between two well-painted lines 3 m or more apart with no solid line beside it - a
    traffic lane (-0.7); a yellow centre line on the left with no white line in between (-0.5). Something bright on the
    right (gutter, kerb, car doors) never vetoes a solid line on the left.
    """
    if r["n_lines"] >= JUNCTION_LINES and r["stencil_conf"] <= 0:
        return float("nan")
    left, width = r["left_x"], r["lane_width"]
    left_near = np.isfinite(left) and -2.4 <= left
    left_white = left_near and r["left_yellow"] < 0.5
    left_yellow = left_near and r["left_yellow"] >= 0.5 and r["left_cover"] >= 0.4
    solid_white_left = np.isfinite(r.get("left_solid_white_x", np.nan))
    parked_right = np.isfinite(r["right_vehicle_x"]) and r["right_vehicle_x"] <= 3.2
    evidence, informative = 0.0, False
    if r["stencil_conf"] > 0:
        evidence += 0.9
        informative = True
    if solid_white_left:
        evidence += 0.7
        informative = True
    elif np.isfinite(width) and 1.0 <= width <= 2.8 and (r["left_solid"] or r["right_solid"]):
        evidence += 0.6
        informative = True
    elif left_white and parked_right:
        evidence += 0.3
        informative = True
    if np.isfinite(r["green_share"]) and r["green_share"] >= 0.15:
        evidence += 0.5
        informative = True
    well_painted = min(r["left_cover"], r["right_cover"]) >= 0.3 if np.isfinite(width) else False
    if np.isfinite(width) and width >= 3.0 and well_painted and not solid_white_left:
        evidence -= 0.7
        informative = True
    if left_yellow and not solid_white_left:
        evidence -= 0.5
        informative = True
    return float(np.clip(evidence, -1.0, 1.0)) if informative else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# linking detections of the same sign or light across samples


def link_objects(objects: pd.DataFrame, u_foe: dict[int, float], horizon: float, max_gap: int = 2) -> pd.Series:
    """Give each physical sign / signal head one id across samples.

    A static object moves straight away from the focus of expansion as the rider advances and grows by
    Z_before/Z_after, so the previous box is scaled about the FOE by a few plausible factors and matched by IoU.
    """
    from scipy.optimize import linear_sum_assignment

    ids = pd.Series(-1, index=objects.index, dtype=int)
    if objects.empty:
        return ids
    scales = np.array([0.95, 1.0, 1.1, 1.25, 1.45, 1.7])
    next_id = 0
    frames = sorted(objects.frame.unique())
    frame_pos = {f: i for i, f in enumerate(frames)}
    for kind, group in objects.groupby("kind"):
        if kind == "marking":  # bird's-eye boxes: not in image coordinates, so not linked
            ids[group.index] = np.arange(next_id, next_id + len(group))
            next_id += len(group)
            continue
        active: dict[int, tuple[int, str, np.ndarray]] = {}  # id -> (last frame position, category, box)
        for frame, rows in group.groupby("frame", sort=True):
            pos = frame_pos[frame]
            active = {k: v for k, v in active.items() if pos - v[0] <= max_gap + 1}
            foe = np.array([u_foe.get(int(frame), np.nan), horizon])
            track_ids = list(active)
            det_idx = list(rows.index)
            if track_ids and det_idx:
                cost = np.ones((len(track_ids), len(det_idx)))
                for a, tid in enumerate(track_ids):
                    box = active[tid][2]
                    centre = foe if np.isfinite(foe[0]) else np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    for b, di in enumerate(det_idx):
                        # no category gate: the open vocabulary can name one sign differently from sample to sample,
                        # and the static-object prediction is specific enough; an instance takes its majority name
                        det = rows.loc[di, ["x1", "y1", "x2", "y2"]].to_numpy(float)
                        best = 0.0
                        for s in scales:
                            pred = np.concatenate([centre + (box[:2] - centre) * s, centre + (box[2:] - centre) * s])
                            best = max(best, _iou(pred, det))
                        cost[a, b] = 1.0 - best
                rows_i, cols_i = linear_sum_assignment(cost)
                matched = set()
                for a, b in zip(rows_i, cols_i):
                    if cost[a, b] <= 0.8:  # IoU >= 0.2 after the static-object prediction
                        tid, di = track_ids[a], det_idx[b]
                        ids[di] = tid
                        active[tid] = (pos, active[tid][1], rows.loc[di, ["x1", "y1", "x2", "y2"]].to_numpy(float))
                        matched.add(di)
                det_idx = [di for di in det_idx if di not in matched]
            for di in det_idx:
                ids[di] = next_id
                active[next_id] = (pos, rows.at[di, "category"], rows.loc[di, ["x1", "y1", "x2", "y2"]].to_numpy(float))
                next_id += 1
    return ids


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------------------------------------------------
# per-second summary used by bikesafe.exposure


def parse_lines(lines_json: str) -> list[tuple[float, bool, float, float]]:
    """(x, solid, yellow share, shear) for each line in a lanes.parquet `lines_json` cell."""
    out = []
    for item in str(lines_json).split(";"):
        parts = item.split(":")
        if len(parts) >= 3 and parts[0]:
            shear = float(parts[3]) if len(parts) > 3 else 0.0
            out.append((float(parts[0]), bool(int(parts[1])), float(parts[2]), shear))
    return out


def per_second_summary(lanes: pd.DataFrame, objects: pd.DataFrame, n_seconds: int) -> pd.DataFrame:
    """Per second: bike-lane score and decision, the state of the signal heads facing the rider, and signs in view.

    Bike lane: per-sample evidence is averaged per second, then over a centred BIKE_LANE_WINDOW_S window; seconds
    with no paint in view (junctions) hold the neighbouring decision for up to BIKE_LANE_HOLD_S.
    """
    out = pd.DataFrame(index=pd.RangeIndex(0, max(n_seconds, 0), name="second"))
    if len(lanes):
        per_s = lanes.assign(second=lanes.time_s.astype(int)).groupby("second").bike_lane_evidence.mean()
        per_s = per_s.reindex(out.index)
        score = per_s.rolling(BIKE_LANE_WINDOW_S, center=True, min_periods=1).mean()
        score = score.ffill(limit=BIKE_LANE_HOLD_S).bfill(limit=BIKE_LANE_HOLD_S)
        out["bike_lane_score"] = score.round(3)
        out["in_bike_lane"] = (score >= BIKE_LANE_THRESHOLD).where(score.notna())
        visible = lanes.assign(second=lanes.time_s.astype(int)).groupby("second").n_lines.max().reindex(out.index)
        out["lane_lines_visible"] = visible
    else:
        out["bike_lane_score"] = np.nan
        out["in_bike_lane"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out["lane_lines_visible"] = np.nan
    if len(objects):
        obj = objects.assign(second=objects.time_s.astype(int))
        lights = obj[(obj.category == "traffic_light") & obj.light_state.isin(["red", "yellow", "green"])]
        if len(lights):
            # the heads facing the rider are the lit ones; weight by apparent size, which favours the nearest junction
            area = (lights.x2 - lights.x1) * (lights.y2 - lights.y1)
            votes = lights.assign(area=area).groupby(["second", "light_state"]).area.sum().unstack(fill_value=0)
            state = votes.idxmax(axis=1)
            # a yellow reading must be confirmed by two heads: yellow awnings and walls behind a head can fool one
            n_yellow = lights[lights.light_state == "yellow"].groupby("second").size()
            state[(state == "yellow") & (n_yellow.reindex(state.index).fillna(0) < 2)] = np.nan
            out["traffic_light_state"] = state.reindex(out.index)
        else:
            out["traffic_light_state"] = np.nan
        out["n_signal_heads"] = obj[obj.kind == "light"].groupby("second").object_id.nunique().reindex(out.index).fillna(0).astype(int)
        signs = obj[obj.kind == "sign"]
        out["n_signs"] = signs.groupby("second").object_id.nunique().reindex(out.index).fillna(0).astype(int)
        out["stop_sign"] = (signs[signs.category == "stop"].groupby("second").size().reindex(out.index).fillna(0) > 0)
        out["bike_stencil"] = (obj[obj.category == "bike_stencil"].groupby("second").size().reindex(out.index).fillna(0) > 0)
    else:
        out["traffic_light_state"] = np.nan
        out["n_signal_heads"] = 0
        out["n_signs"] = 0
        out["stop_sign"] = False
        out["bike_stencil"] = False
    return out


def object_instances(objects: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct sign / signal head: category, first and last time seen, best confidence, lamp states."""
    if objects.empty:
        return pd.DataFrame(columns=["object_id", "kind", "category", "t_first", "t_last", "n_samples", "conf_max",
                                     "box_h_max", "states"])
    rows = []
    for object_id, g in objects.groupby("object_id"):
        # a stop sign confirmed by the COCO check in any sample is a stop sign; otherwise the majority name
        category = "stop" if (g.category == "stop").any() else g.category.mode().iloc[0]
        states = g.light_state[g.light_state.isin(["red", "yellow", "green"])]
        rows.append({"object_id": int(object_id), "kind": g.kind.iloc[0], "category": category,
                     "t_first": float(g.time_s.min()), "t_last": float(g.time_s.max()), "n_samples": int(len(g)),
                     "conf_max": float(g.conf.max()), "box_h_max": float((g.y2 - g.y1).max()),
                     "states": "/".join(dict.fromkeys(states)) if len(states) else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------------
# the pass over a video


def load_ego_foe(analysis_dir: Path, calib: Calibration) -> dict[int, float]:
    """Heading column (focus of expansion) per processed frame, from the ego-motion pass when it has run."""
    raw = analysis_dir / "egomotion_raw.parquet"
    if not raw.exists():
        return {}
    ego = ego_from_tracking(pd.read_parquet(raw), calib)
    return dict(zip(ego.frame.astype(int), ego.u_foe.astype(float)))


def run(perception_dir: Path, analysis_root: Path, videos: Path, sample_fps: float, device: str, weights: str,
        imgsz: int, bev_imgsz: int, verifier: str = STOP_VERIFIER) -> tuple[Path, Path]:
    meta = read_json(perception_dir / "meta.json")
    dets = pd.read_parquet(perception_dir / "detections.parquet")
    calib = fit_horizon(dets, meta["width"], meta["height"], meta["processed_fps"], meta["stride"])
    out_dir = analysis_root / meta["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    foe = load_ego_foe(out_dir, calib)
    boxes_by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in dets.groupby("frame")}
    stride = meta["stride"]
    step = max(stride, int(round(meta["source_fps"] / sample_fps / stride)) * stride)

    detector = InfrastructureDetector(weights, device, half=is_cuda(device), verifier_weights=verifier)
    view = GroundView(calib)
    cap = open_video(resolve_video(meta, videos))
    if meta["start_frame"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
    objects: list[dict] = []
    lanes: list[dict] = []
    idx, started = meta["start_frame"], time.perf_counter()
    while idx < meta["end_frame"]:
        if (idx - meta["start_frame"]) % step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / meta["source_fps"]
        u_foe = foe.get(idx, calib.width / 2)
        boxes = boxes_by_frame.get(idx, np.zeros((0, 4)))

        image_objects = detector(frame, imgsz, ("light", "sign"))
        detector.confirm_stop_signs(frame, image_objects, imgsz)
        for d in image_objects:
            d.pop("stop_confirmed", None)
            state, share = ("", 0.0)
            if d["kind"] == "light":
                crop = frame[int(d["y1"]): int(np.ceil(d["y2"])), int(d["x1"]): int(np.ceil(d["x2"]))]
                state, share = light_state(crop)
            objects.append({"frame": idx, "time_s": t, **d, "light_state": state, "state_share": share,
                            "ground_x": np.nan, "ground_z": np.nan})

        bev = view.warp(frame, u_foe)
        valid = view.warp(road_mask(frame.shape[:2], calib.horizon_v, boxes), u_foe, nearest=True) > 0
        stencils = []
        for d in detector(bev, bev_imgsz, ("marking",)):
            gx, gz = view.ground((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2)
            if valid[int((d["y1"] + d["y2"]) / 2), int((d["x1"] + d["x2"]) / 2)]:
                stencils.append({**d, "x": gx, "z": gz})
                objects.append({"frame": idx, "time_s": t, **d, "light_state": "", "state_share": 0.0,
                                "ground_x": gx, "ground_z": gz})
        white, yellow = bev_paint(bev, valid)
        lines = lane_lines(white, yellow, valid, view.zs)
        vehicle_x, vehicle_z = vehicles_on_ground(boxes, calib, u_foe)
        lanes.append({"frame": idx, "time_s": t, "u_foe": u_foe, "road_visible": float(valid.mean()),
                      **lane_record(lines, stencils, green_share(bev, valid, view), vehicle_x, vehicle_z)})
        idx += 1
        if len(lanes) % 200 == 0:
            done = (idx - meta["start_frame"]) / max(1, meta["end_frame"] - meta["start_frame"])
            print(f"{meta['stem']}: {done:5.1%} {len(lanes)} samples {len(objects)} objects "
                  f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    cap.release()

    obj = pd.DataFrame(objects, columns=["frame", "time_s", "category", "kind", "conf", "x1", "y1", "x2", "y2",
                                         "light_state", "state_share", "ground_x", "ground_z"])
    obj["object_id"] = link_objects(obj, foe, calib.horizon_v).to_numpy()
    lane_table = pd.DataFrame(lanes)
    obj_path, lane_path = out_dir / "infrastructure.parquet", out_dir / "lanes.parquet"
    obj.to_parquet(obj_path, index=False)
    lane_table.to_parquet(lane_path, index=False)
    in_lane = lane_table.bike_lane_evidence.gt(0).mean() if len(lane_table) else 0.0
    print(f"{meta['stem']}: {len(lane_table)} samples, {obj.object_id.nunique() if len(obj) else 0} lights/signs/stencils,"
          f" bike-lane evidence in {in_lane:.0%} of samples, {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return obj_path, lane_path


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("perception_dirs", nargs="*", type=Path)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLOE-26 weights (downloaded by ultralytics on first use)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detector input size for the camera image")
    parser.add_argument("--bev-imgsz", type=int, default=640, help="Detector input size for the bird's-eye view")
    parser.add_argument("--stop-verifier", default=STOP_VERIFIER, help="COCO-trained YOLO weights that confirm stop signs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--build-vocab", action="store_true",
                        help="Only (re)build models/infrastructure_vocab.npz; needs the text encoder download once")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.build_vocab:
        from ultralytics import YOLOE

        model = YOLOE(args.model)
        if VOCAB_CACHE.exists():
            VOCAB_CACHE.unlink()
        feats = text_features(model, list(VOCABULARY))
        print(f"wrote {VOCAB_CACHE} ({feats.shape[1]} prompts)")
        return
    for path in args.perception_dirs:
        stem = read_json(path / "meta.json")["stem"]
        if (args.analysis / stem / "lanes.parquet").exists() and not args.overwrite:
            print(f"skip {stem}: infrastructure exists")
            continue
        run(path, args.analysis, args.videos, args.sample_fps, args.device, args.model, args.imgsz, args.bev_imgsz,
            args.stop_verifier)


if __name__ == "__main__":
    main()
