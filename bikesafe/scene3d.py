"""Depth, bird's-eye position and lane structure for every tracked vehicle.

Two geometries, each used where it is valid:

* **Learned metric depth** (Depth Anything V2, metric outdoor) for the vehicles. It does not assume a flat road and
  keeps working past 10 m, where wheel-row geometry degrades. Its absolute scale is tuned to car-dashcam intrinsics,
  so it is rescaled per video against the one thing we know exactly - a car is about 1.55 m tall - using clean,
  untruncated car boxes. Only the scale is borrowed; the depth *shape* is the model's.
* **The same depth map for the lane paint.** White and yellow markings are detected in the image and projected into a
  bird's-eye grid with the depth they were measured at - not with a flat-road assumption, which fans out badly past
  15 m when the horizon estimate is a few pixels off. The grid is accumulated over a short window so dashed lines are
  not missed. Vehicles and paint therefore live in one consistent bird's-eye frame, which supplies the feature the
  first model was missing: how many lane lines, and whether a yellow centre line, lie between the rider and the vehicle.

Per-track headings (does it travel with the rider, against, or across?) are derived from these samples in
bikesafe.tracks, where the rider's own motion is available to subtract.

    python -m bikesafe.scene3d work/perception/VID_... [--sample-fps 2]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from bikesafe.common import ROOT, read_json
from bikesafe.geometry import CAR_HEIGHT_M, fit_horizon

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_W, DEPTH_H = 518, 294  # multiples of 14, 16:9 aspect preserved
BEV_X_RANGE, BEV_Z_RANGE, BEV_CELL = 12.0, 25.0, 0.25  # lane geometry is only trusted within 25 m
PAINT_WINDOW_S = 1.0
MIN_PAINT_HITS = 3.0


class DepthModel:
    def __init__(self, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), return_tensors="pt",
                                size={"height": DEPTH_H, "width": DEPTH_W}).to(self.device)
        return self.model(**inputs).predicted_depth[0].float().cpu().numpy()


def paint_mask(frame: np.ndarray, horizon: float, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """White and yellow road markings below the horizon, with vehicles masked out."""
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hue, lightness, saturation = hls[..., 0], hls[..., 1], hls[..., 2]
    road = np.zeros(lightness.shape, bool)
    road[int(max(0, horizon + 15)):, :] = True
    for x1, y1, x2, y2 in boxes:
        road[max(0, int(y1) - 4): int(y2) + 4, max(0, int(x1) - 4): int(x2) + 4] = False
    blurred = cv2.GaussianBlur(lightness, (0, 0), 2)
    # paint is brighter than the asphalt beside it, which a wide horizontal opening estimates
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 3)))
    contrast = cv2.subtract(blurred, background)
    yellow = road & (contrast > 10) & (hue > 12) & (hue < 38) & (saturation > 70)
    white = road & (contrast > 18) & (saturation < 90)
    return white | yellow, yellow


def depth_to_bev(u: np.ndarray, z: np.ndarray, cx: float, focal: float) -> np.ndarray:
    """Lateral offset from the direction of travel for points at measured depth z."""
    return (u - cx) * z / focal


def bev_accumulate(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], cx: float, focal: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Recent paint pixels projected into one bird's-eye grid using their measured depth (all paint, yellow only)."""
    nx, nz = int(2 * BEV_X_RANGE / BEV_CELL), int(BEV_Z_RANGE / BEV_CELL)
    grid = np.zeros((nz, nx), np.float32)
    grid_yellow = np.zeros((nz, nx), np.float32)
    for mask, yellow, depth_full in samples:
        vs, us = np.nonzero(mask)
        if not len(vs):
            continue
        if len(vs) > 60000:
            keep = rng.choice(len(vs), 60000, replace=False)
            vs, us = vs[keep], us[keep]
        z = depth_full[vs, us]
        x = depth_to_bev(us.astype(np.float32), z, cx, focal)
        ok = (np.abs(x) < BEV_X_RANGE) & (z > 1.5) & (z < BEV_Z_RANGE)
        if not ok.any():
            continue
        xi = np.clip(((x[ok] + BEV_X_RANGE) / BEV_CELL).astype(int), 0, nx - 1)
        zi = np.clip((z[ok] / BEV_CELL).astype(int), 0, nz - 1)
        np.add.at(grid, (zi, xi), 1.0)
        is_yellow = yellow[vs[ok], us[ok]]
        np.add.at(grid_yellow, (zi[is_yellow], xi[is_yellow]), 1.0)
    return grid, grid_yellow


def lines_between(grid: np.ndarray, grid_yellow: np.ndarray, x_obj: float, z_obj: float,
                  band_m: float = 6.0) -> tuple[int, int, float]:
    """Paint clusters between the rider's path (x = 0) and the vehicle, at the vehicle's distance."""
    if not (np.isfinite(x_obj) and np.isfinite(z_obj)) or not 1.5 < z_obj < BEV_Z_RANGE:
        return -1, -1, float("nan")
    z_lo = max(0, int((z_obj - band_m) / BEV_CELL))
    z_hi = min(grid.shape[0], int((z_obj + band_m) / BEV_CELL) + 1)
    profile = grid[z_lo:z_hi].sum(axis=0)
    profile_yellow = grid_yellow[z_lo:z_hi].sum(axis=0)
    centre = int(BEV_X_RANGE / BEV_CELL)
    target = int(np.clip((x_obj + BEV_X_RANGE) / BEV_CELL, 0, len(profile) - 1))
    lo, hi = sorted((centre, target))
    painted = profile >= MIN_PAINT_HITS
    segment = painted[lo:hi + 1]
    padded = np.concatenate([[False], segment, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    yellow_runs = int(sum(profile_yellow[lo + s: lo + e].sum() >= MIN_PAINT_HITS for s, e in zip(starts, ends)))
    columns = np.flatnonzero(painted)
    nearest = float(np.min(np.abs(columns - target)) * BEV_CELL) if len(columns) else float("nan")
    return int(len(starts)), yellow_runs, nearest


def run(perception_dir: Path, analysis_dir: Path, videos: Path, sample_fps: float, device: str) -> Path:
    meta = read_json(perception_dir / "meta.json")
    dets = pd.read_parquet(perception_dir / "detections.parquet")
    calib = fit_horizon(dets, meta["width"], meta["height"], meta["processed_fps"], meta["stride"])
    tracked = dets[dets.track_id >= 0]
    boxes_by_frame = {int(f): g for f, g in tracked.groupby("frame")}
    all_boxes_by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in dets.groupby("frame")}
    stride = meta["stride"]
    step = max(stride, int(round(meta["source_fps"] / sample_fps / stride)) * stride)
    video = Path(meta["video"])
    if not video.exists():
        video = videos / f"{meta['stem']}.mp4"

    depth_model = DepthModel(device)
    cap = cv2.VideoCapture(str(video))
    width, height = meta["width"], meta["height"]
    cx = width / 2
    sx, sy = DEPTH_W / width, DEPTH_H / height
    rng = np.random.default_rng(0)
    recent: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    window = max(1, int(round(PAINT_WINDOW_S * sample_fps)))
    calibration_pairs: list[tuple[float, float]] = []
    scale = float("nan")  # model depth -> metres, fixed once enough clean cars have been measured
    rows: list[dict] = []
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
        frame_dets = boxes_by_frame.get(idx)
        if frame_dets is None or not len(frame_dets):
            idx += 1
            continue
        depth_small = depth_model(frame)

        measured = []
        for det in frame_dets.itertuples():
            box_h, box_w = det.y2 - det.y1, det.x2 - det.x1
            # model depth just above the wheel line, across the middle of the vehicle
            y = int(np.clip((det.y2 - 0.10 * box_h) * sy, 0, DEPTH_H - 1))
            x0 = int(np.clip((det.x1 + 0.3 * box_w) * sx, 0, DEPTH_W - 2))
            x1 = int(np.clip((det.x2 - 0.3 * box_w) * sx, x0 + 1, DEPTH_W - 1))
            patch = depth_small[max(0, y - 3): y + 2, x0:x1]
            z_model = float(np.percentile(patch, 40)) if patch.size else float("nan")
            z_box = float(calib.focal_px * CAR_HEIGHT_M / box_h) if box_h > 0 else float("nan")
            clean_car = bool(det.class_id == 2 and det.confidence >= 0.5 and 40 <= box_h <= 0.45 * height
                             and det.x1 > 3 and det.x2 < width - 3 and det.y2 < height - 3 and det.y1 > 3)
            if clean_car and np.isfinite(z_model) and z_model > 0 and np.isfinite(z_box):
                calibration_pairs.append((z_box, z_model))
            measured.append((det, z_model, z_box, clean_car))
        if not np.isfinite(scale) and len(calibration_pairs) >= 80:
            box_depths, model_depths = np.array(calibration_pairs).T
            scale = float(np.median(box_depths / model_depths))

        grid = grid_yellow = None
        if np.isfinite(scale):
            depth_full = cv2.resize(depth_small, (width, height), interpolation=cv2.INTER_LINEAR) * scale
            mask, yellow = paint_mask(frame, calib.horizon_v, all_boxes_by_frame.get(idx, np.zeros((0, 4))))
            recent.append((mask, yellow, depth_full))
            recent[:] = recent[-window:]
            grid, grid_yellow = bev_accumulate(recent, cx, calib.focal_px, rng)

        for det, z_model, z_box, clean_car in measured:
            z_metric = z_model * scale
            u_c = float((det.x1 + det.x2) / 2)
            bev_x = float(depth_to_bev(np.array([u_c]), np.array([z_metric]), cx, calib.focal_px)[0])
            crossings, yellow_runs, nearest = ((-1, -1, float("nan")) if grid is None
                                               else lines_between(grid, grid_yellow, bev_x, z_metric))
            rows.append({
                "frame": idx, "track_id": int(det.track_id), "z_model_raw": z_model, "z_box_height": z_box,
                "clean_car": clean_car, "u_c": u_c, "z_metric": z_metric, "bev_x": bev_x,
                "lines_between": crossings, "yellow_between": yellow_runs, "dist_to_line_m": nearest,
            })
        idx += 1
        if len(rows) >= 5000 and (idx // step) % 400 == 0:
            done = (idx - meta["start_frame"]) / max(1, meta["end_frame"] - meta["start_frame"])
            print(f"{meta['stem']}: {done:5.1%} {len(rows)} samples {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    cap.release()

    table = pd.DataFrame(rows)
    out_dir = analysis_dir / meta["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "scene3d.parquet", index=False)
    with_lanes = float(np.mean(table.lines_between >= 0)) if len(table) else 0.0
    print(f"{meta['stem']}: {len(table)} samples, depth scale {scale:.3f}, lane geometry on {with_lanes:.0%}, "
          f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return out_dir / "scene3d.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("perception_dirs", nargs="+", type=Path)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in args.perception_dirs:
        if (args.analysis / path.name / "scene3d.parquet").exists() and not args.overwrite:
            print(f"skip {path.name}: scene3d exists")
            continue
        run(path, args.analysis, args.videos, args.sample_fps, args.device)


if __name__ == "__main__":
    main()
