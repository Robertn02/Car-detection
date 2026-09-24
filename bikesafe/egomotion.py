"""CPU ego-motion pass: rider speed, yaw and heading from road-surface feature tracking.

Dense flow at the perception resolution saturates on fast-moving asphalt just ahead of the wheel, so ego motion
is measured separately with forward-backward-checked pyramidal Lucas-Kanade on 960x540 frames, sampling features
in a road band 70-170 px below the horizon and outside detected vehicles. Per processed frame the flat-road model

    dv = g*d^2 + p + w*x*y/f^2          du = g*d*(u - u_foe) + w*(1 + x^2/f^2)

(d = rows below horizon, x/y = offsets from the image centre, w = yaw shift at the centre, p = pitch shift)
is fitted with iteratively reweighted least squares. Requires the perception outputs and the horizon calibration.

    python -m bikesafe.egomotion work/perception/VID_... [--analysis work/analysis]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import ROOT, open_video, read_json, resolve_video
from bikesafe.geometry import fit_horizon

SCALE = 0.5
BAND_MIN_D, BAND_MAX_D = 70, 170  # full-res rows below the horizon (~5-12 m ahead) used for ego-motion features
LK_PARAMS = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def fit_frame(u: np.ndarray, v: np.ndarray, du: np.ndarray, dv: np.ndarray, horizon: float, cx: float, cy: float,
              focal: float) -> tuple[float, float, float, float, int, float]:
    """Return g, pitch shift p, u_foe, yaw shift w (all per step, full-res px), inliers, residual scale."""
    d = v - horizon
    x, y = u - cx, v - cy
    n = len(u)
    a = np.zeros((2 * n, 4))
    b = np.concatenate([dv, du])
    a[:n, 0] = d * d
    a[:n, 1] = 1.0
    a[:n, 3] = x * y / focal ** 2
    a[n:, 0] = d * u
    a[n:, 2] = -d
    a[n:, 3] = 1.0 + (x / focal) ** 2
    w = np.ones(2 * n)
    coef = np.zeros(4)
    scale = 1.0
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)
        res = b - a @ coef
        point_res = np.hypot(res[:n], res[n:])
        scale = 1.4826 * float(np.median(point_res)) + 0.05
        pw = np.clip(2.0 * scale / (point_res + 1e-9), 0, 1)
        w = np.concatenate([pw, pw])
    g, p, g_foe, yaw = coef
    inliers = int((point_res < 3 * scale).sum())
    u_foe = g_foe / g if abs(g) > 1e-7 else np.nan
    return float(g), float(p), float(u_foe), float(yaw), inliers, scale


def run(perception_dir: Path, analysis_dir: Path, videos: Path, max_corners: int) -> Path:
    meta = read_json(perception_dir / "meta.json")
    dets = pd.read_parquet(perception_dir / "detections.parquet")
    calib = fit_horizon(dets, meta["width"], meta["height"], meta["processed_fps"], meta["stride"])
    boxes_by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in dets.groupby("frame")}
    cap = open_video(resolve_video(meta, videos))
    start, end, stride = meta["start_frame"], meta["end_frame"], meta["stride"]
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w_small, h_small = int(meta["width"] * SCALE), int(meta["height"] * SCALE)
    horizon_small = calib.horizon_v * SCALE
    # Road roughly 5-12 m ahead. Nearer asphalt is motion-blurred at riding speed: features there are sensor/compression
    # texture that stays fixed in the frame and would drag the speed estimate toward zero.
    band_top = int(max(0, horizon_small + BAND_MIN_D * SCALE))
    band_bottom = int(min(h_small, horizon_small + BAND_MAX_D * SCALE))
    rows = []
    prev = None
    prev_idx = -1
    started = time.perf_counter()
    idx = start
    while idx < end:
        if (idx - start) % stride:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (w_small, h_small), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        record = {"frame": idx, "g": np.nan, "pitch": np.nan, "u_foe": np.nan, "yaw": np.nan, "n_points": 0, "inliers": 0,
                  "residual": np.nan}
        if prev is not None:
            mask = np.zeros_like(prev)
            mask[band_top:band_bottom, :] = 255
            for x1, y1, x2, y2 in boxes_by_frame.get(prev_idx, []):
                mask[max(0, int(y1 * SCALE) - 4): int(y2 * SCALE) + 6, max(0, int(x1 * SCALE) - 4): int(x2 * SCALE) + 4] = 0
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners, qualityLevel=0.005, minDistance=7, mask=mask, blockSize=7)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **LK_PARAMS)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, **LK_PARAMS)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 0.7)
                if good.sum() >= 12:
                    p0 = pts.reshape(-1, 2)[good] / SCALE
                    p1 = nxt.reshape(-1, 2)[good] / SCALE
                    g, pitch, u_foe, yaw, inliers, residual = fit_frame(
                        p0[:, 0], p0[:, 1], p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1],
                        calib.horizon_v, meta["width"] / 2, meta["height"] / 2, calib.focal_px)
                    record.update(g=g, pitch=pitch, u_foe=u_foe, yaw=yaw, n_points=int(good.sum()), inliers=inliers,
                                  residual=residual)
        rows.append(record)
        prev, prev_idx = gray, idx
        idx += 1
        if len(rows) % 3000 == 0:
            print(f"{meta['stem']}: {len(rows)}/{meta['processed_frames']} frames {len(rows) / (time.perf_counter() - started):.1f} fps", flush=True)
    cap.release()
    out = analysis_dir / meta["stem"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "egomotion_raw.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"{meta['stem']}: ego-motion for {len(rows)} frames in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("perception_dirs", nargs="+", type=Path)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--max-corners", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in args.perception_dirs:
        if (args.analysis / path.name / "egomotion_raw.parquet").exists() and not args.overwrite:
            print(f"skip {path.name}: ego-motion exists")
            continue
        run(path, args.analysis, args.videos, args.max_corners)


if __name__ == "__main__":
    main()
