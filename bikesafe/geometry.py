"""Monocular ground-plane geometry for a forward bike camera, self-calibrated from the footage.

Pinhole + flat-road model (x right, z forward, camera height h above the road, horizon row v_h):
  * a ground point at depth Z projects to row v with  d = v - v_h = f*h/Z,
  * a car of height H with its wheels at that row has box height  box_h = f*H/Z = d*H/h,
    so regressing wheel row on box height over many cars gives the horizon v_h and h/H (camera height),
  * lateral offset from the direction of travel is  X = h*(u - u_foe)/d  (no focal length needed),
  * pure forward ego motion V moves a static ground point by  dv = d^2 * V/(f*h)  per second, so the
    background flow grid gives g = V/(f*h) per frame; yaw adds a horizontal shift r shared by all points.

For a tracked vehicle, the static-world prediction of its box expansion rate is g*d; the measured expansion
rate s gives its own along-road speed relative to the ground:  V_obj/(f*h) = g - s/d  (positive = same
direction as the rider). Lateral displacement over a step is corrected for yaw by subtracting r/d.
Quantities are kept in camera-height units ("_rel") and converted to metres only with the assumed focal
length, which can be recalibrated later against Strava speed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

CAR_HEIGHT_M = 1.55
CAR_BOX_WIDTH_M = 1.95  # rear/front view detection box, mirrors included
DEFAULT_HFOV_DEG = 110.0


@dataclass
class Calibration:
    width: int
    height: int
    fps: float  # processed frames per second
    stride: int  # source frames per processed frame
    horizon_v: float
    height_ratio: float  # camera height / car height, the slope k in  d = k * box_h
    cam_height_m: float
    lateral_height_m: float  # effective camera height for lateral offsets, from rear-view car widths
    focal_px: float
    inliers: int
    residual_px: float

    def as_dict(self) -> dict:
        return asdict(self)


def fit_horizon(dets: pd.DataFrame, width: int, height: int, fps: float, stride: int,
                hfov_deg: float = DEFAULT_HFOV_DEG, seed: int = 0) -> Calibration:
    """Robustly fit wheel row = v_h + k * box height over confident, untruncated cars."""
    box_h = dets.y2 - dets.y1
    keep = (
        (dets.class_id == 2) & (dets.confidence >= 0.5) & (box_h >= 24) & (box_h <= 0.45 * height)
        & (dets.x1 > 3) & (dets.x2 < width - 3) & (dets.y2 < height - 3) & (dets.y1 > 3)
    )
    x = box_h[keep].to_numpy(float)
    y = dets.y2[keep].to_numpy(float)
    rng = np.random.default_rng(seed)
    if len(x) > 20000:
        idx = rng.choice(len(x), 20000, replace=False)
        x, y = x[idx], y[idx]
    best_inliers = np.zeros(len(x), bool)
    for _ in range(400):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 10:
            continue
        k = (y[i] - y[j]) / (x[i] - x[j])
        if not 0.2 < k < 2.0:
            continue
        v0 = y[i] - k * x[i]
        inliers = np.abs(y - (v0 + k * x)) < 6 + 0.04 * x
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    k, v0 = np.polyfit(x[best_inliers], y[best_inliers], 1)
    residual = float(np.median(np.abs(y[best_inliers] - (v0 + k * x[best_inliers]))))
    focal = (width / 2) / np.tan(np.radians(hfov_deg) / 2)

    # Lateral scale: the reframed 360 image is not square-pixel pinhole, so calibrate X = h_x*(u-u0)/d separately
    # from cars seen square-on near the image centre, where box width ~ CAR_BOX_WIDTH_M.
    box_w = dets.x2 - dets.x1
    centre = keep & ((dets.x1 + dets.x2) / 2 - width / 2).abs().lt(0.08 * width) & (box_h >= 40)
    centre &= (box_w / box_h).between(1.0, 1.7) & (dets.confidence >= 0.6)
    d_centre = dets.y2[centre] - v0
    ratios = (CAR_BOX_WIDTH_M * d_centre / box_w[centre]).to_numpy(float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    lateral_h = float(np.median(ratios)) if len(ratios) >= 20 else float(k * CAR_HEIGHT_M)
    return Calibration(width, height, fps, stride, float(v0), float(k), float(k * CAR_HEIGHT_M), lateral_h, float(focal),
                       int(best_inliers.sum()), residual)


def ego_motion(frames: np.ndarray, grid: np.ndarray, calib: Calibration, smooth_s: float = 1.0) -> pd.DataFrame:
    """Per processed frame: forward-motion rate g (1/(px*s)), yaw shift r (px/s), heading column u_foe.

    Uses flow-grid cells whose centres lie well below the horizon (mostly road surface; vehicles were masked).
    """
    n_frames, gh, gw, _ = grid.shape
    cell_u = (np.arange(gw) + 0.5) * calib.width / gw
    cell_v = (np.arange(gh) + 0.5) * calib.height / gh
    uu, vv = np.meshgrid(cell_u, cell_v)
    d = vv - calib.horizon_v
    ground = d > 0.12 * calib.height
    du_all = grid[..., 0][:, ground] * calib.fps
    dv_all = grid[..., 1][:, ground] * calib.fps
    d_g, u_g = d[ground], uu[ground]

    g = np.full(n_frames, np.nan)
    r = np.full(n_frames, np.nan)
    foe = np.full(n_frames, np.nan)
    quality = np.zeros(n_frames)
    for i in range(n_frames):
        ok = np.isfinite(du_all[i]) & np.isfinite(dv_all[i])
        if ok.sum() < 8:
            continue
        dd, uu_i, du, dv = d_g[ok], u_g[ok], du_all[i][ok], dv_all[i][ok]
        # dv = g*d^2 + p  (Huber-style reweighting)
        a = np.column_stack([dd * dd, np.ones_like(dd)])
        w = np.ones_like(dd)
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(a * w[:, None], dv * w, rcond=None)
            res = dv - a @ coef
            scale = 1.4826 * np.median(np.abs(res)) + 1e-6
            w = np.clip(1.5 * scale / (np.abs(res) + 1e-9), 0, 1)
        g[i] = coef[0]
        quality[i] = float((np.abs(res) < 3 * scale).mean())
        # du = g*d*u - (g*u_foe)*d + r
        y = du - g[i] * dd * uu_i
        b = np.column_stack([-dd, np.ones_like(dd)])
        w = np.ones_like(dd)
        for _ in range(4):
            cb, *_ = np.linalg.lstsq(b * w[:, None], y * w, rcond=None)
            res = y - b @ cb
            scale = 1.4826 * np.median(np.abs(res)) + 1e-6
            w = np.clip(1.5 * scale / (np.abs(res) + 1e-9), 0, 1)
        r[i] = cb[1]
        if abs(g[i]) > 2e-6:
            foe[i] = cb[0] / g[i]

    ego = pd.DataFrame({"frame": frames, "g": g, "yaw_px_s": r, "u_foe_raw": foe, "fit_quality": quality})
    window = max(3, int(round(smooth_s * calib.fps)) | 1)
    ego["g_smooth"] = ego.g.rolling(window, center=True, min_periods=2).median()
    moving = ego.g_smooth > ego_moving_threshold(calib)
    plausible = ego.u_foe_raw.between(0.25 * calib.width, 0.75 * calib.width) & moving
    heading = ego.u_foe_raw.where(plausible)
    long_window = max(window, int(round(4 * calib.fps)) | 1)
    ego["u_foe"] = heading.rolling(long_window, center=True, min_periods=3).median().ffill().bfill().fillna(calib.width / 2)
    ego["yaw_smooth"] = ego.yaw_px_s.rolling(window, center=True, min_periods=2).median()
    ego["ego_moving"] = moving
    ego["ego_speed_mps"] = ego.g_smooth * calib.focal_px * calib.cam_height_m
    # cumulative rider pose: how far it has advanced and how far it has turned (radians), for world-frame trajectories
    dt = 1.0 / calib.fps
    ego["advance_m"] = (ego.ego_speed_mps.fillna(0) * dt).cumsum()
    ego["turn_rad"] = (ego.yaw_smooth.fillna(0) * dt).cumsum() / calib.focal_px
    return ego


def ego_from_tracking(raw: pd.DataFrame, calib: Calibration, smooth_s: float = 1.0) -> pd.DataFrame:
    """Smooth the per-frame Lucas-Kanade fits from bikesafe.egomotion into the columns used downstream."""
    ego = pd.DataFrame({
        "frame": raw.frame,
        "g": raw.g * calib.fps,
        "yaw_px_s": raw.yaw * calib.fps,
        "u_foe_raw": raw.u_foe,
        "fit_quality": (raw.inliers / raw.n_points.where(raw.n_points > 0)).fillna(0.0),
    })
    reliable = (raw.inliers >= 20) & (ego.fit_quality >= 0.4)
    ego.loc[~reliable, ["g", "yaw_px_s", "u_foe_raw"]] = np.nan
    window = max(3, int(round(smooth_s * calib.fps)) | 1)
    ego["g_smooth"] = ego.g.rolling(window, center=True, min_periods=2).median().interpolate(limit=window, limit_area="inside")
    moving = ego.g_smooth > ego_moving_threshold(calib)
    plausible = ego.u_foe_raw.between(0.25 * calib.width, 0.75 * calib.width) & moving
    long_window = max(window, int(round(4 * calib.fps)) | 1)
    ego["u_foe"] = (ego.u_foe_raw.where(plausible).rolling(long_window, center=True, min_periods=3).median()
                    .ffill().bfill().fillna(calib.width / 2))
    ego["yaw_smooth"] = ego.yaw_px_s.rolling(window, center=True, min_periods=2).median().fillna(0.0)
    ego["ego_moving"] = moving
    ego["ego_speed_mps"] = ego.g_smooth * calib.focal_px * calib.cam_height_m
    # cumulative rider pose: how far it has advanced and how far it has turned (radians), for world-frame trajectories
    dt = 1.0 / calib.fps
    ego["advance_m"] = (ego.ego_speed_mps.fillna(0) * dt).cumsum()
    ego["turn_rad"] = (ego.yaw_smooth.fillna(0) * dt).cumsum() / calib.focal_px
    return ego


def ego_moving_threshold(calib: Calibration, speed_mps: float = 1.0) -> float:
    return speed_mps / (calib.focal_px * calib.cam_height_m)


def rolling_slope(df: pd.DataFrame, key: str, t: str, y: str, window: int, min_periods: int = 4) -> pd.Series:
    """Centred least-squares slope dy/dt over `window` rows within each group, ignoring NaNs."""
    ok = df[y].notna() & df[t].notna()
    tt = df[t].where(ok, 0.0)
    t0 = df.groupby(key, sort=False)[t].transform("first")
    tt = (tt - t0).where(ok, 0.0)
    yy = df[y].where(ok, 0.0)
    parts = pd.DataFrame({key: df[key], "n": ok.astype(float), "t": tt, "y": yy, "tt": tt * tt, "ty": tt * yy})
    sums = parts.groupby(key, sort=False)[["n", "t", "y", "tt", "ty"]].transform(
        lambda c: c.rolling(window, center=True, min_periods=1).sum())
    denom = sums.n * sums.tt - sums.t ** 2
    slope = (sums.n * sums.ty - sums.t * sums.y) / denom.where(denom.abs() > 1e-9)
    return slope.where(sums.n >= min_periods)


def detection_kinematics(dets: pd.DataFrame, ego: pd.DataFrame, calib: Calibration, window_s: float = 1.5) -> pd.DataFrame:
    """Add per-detection geometry and world-frame motion estimates (windowed regressions within each track)."""
    df = dets[dets.track_id >= 0].merge(
        ego[["frame", "g_smooth", "yaw_smooth", "u_foe", "ego_moving", "ego_speed_mps", "advance_m", "turn_rad"]],
        on="frame", how="left"
    ).sort_values(["track_id", "frame"]).reset_index(drop=True)
    W, H, fps = calib.width, calib.height, calib.fps
    df["box_w"] = df.x2 - df.x1
    df["box_h"] = df.y2 - df.y1
    df["u_c"] = (df.x1 + df.x2) / 2
    df["trunc_left"] = df.x1 <= 3
    df["trunc_right"] = df.x2 >= W - 3
    df["trunc_bottom"] = df.y2 >= H - 3
    df["trunc_top"] = df.y1 <= 3
    d_bottom = (df.y2 - calib.horizon_v).where(~df.trunc_bottom & (df.y2 - calib.horizon_v > 4))
    d_height = (df.box_h * calib.height_ratio).where(~df.trunc_top & ~df.trunc_bottom)
    # Wheel row and box height give two independent inverse-depth readings; average them when both are usable.
    df["d_eff"] = pd.concat([d_bottom, d_height], axis=1).mean(axis=1).fillna(df.box_h * calib.height_ratio).clip(lower=4.0)
    inner_u = np.where(df.x1 > df.u_foe, df.x1, np.where(df.x2 < df.u_foe, df.x2, df.u_foe))
    df["x_center_rel"] = (df.u_c - df.u_foe) / df.d_eff
    df["x_inner_rel"] = (inner_u - df.u_foe) / df.d_eff
    df["x_center_m"] = df.x_center_rel * calib.lateral_height_m
    df["x_inner_m"] = df.x_inner_rel * calib.lateral_height_m
    df["z_m"] = calib.focal_px * calib.cam_height_m / df.d_eff
    # Inverse depth is only trustworthy while the whole vehicle is inside the frame: once a box is clipped its
    # apparent size stops growing, which otherwise makes a closely passed parked car look like it moves with the rider.
    df["geom_ok"] = (
        ~df.trunc_top & ~(df.trunc_bottom & (df.trunc_left | df.trunc_right)) & (df.box_h < 0.75 * H)
        & ~(df.trunc_bottom & (df.box_h > 0.45 * H))
    )
    df["inv_d"] = (1.0 / df.d_eff).where(df.geom_ok)

    window = max(5, int(round(window_s * fps)) | 1)
    # Along-road speed over ground in (f*h)/s:  d(1/d)/dt = (V_obj - V_ego)/(f*h)  and  g = V_ego/(f*h).
    df["v_along_rel"] = df.g_smooth + rolling_slope(df, "track_id", "time_s", "inv_d", window)
    df["v_along_mps"] = (df.v_along_rel * calib.focal_px * calib.cam_height_m).where(df.geom_ok)
    # Independent estimate from the dense-flow expansion of the vehicle body (large boxes only).
    scale_rate = (df.obj_scale * fps).where(df.obj_n >= 36)
    scale_rate = scale_rate.groupby(df.track_id, sort=False).transform(
        lambda s: s.rolling(window, center=True, min_periods=3).median())
    df["v_along_flow_mps"] = (df.g_smooth - scale_rate / df.d_eff) * calib.focal_px * calib.cam_height_m

    # Lateral speed from the image-angle rate, which is insensitive to depth jitter: a static point drifts at
    # du/dt = g*d*(u - u_foe) plus the wide-angle yaw shift; the residual converts to metres with h_x/d.
    u_ok = df.u_c.where(~(df.trunc_left | df.trunc_right))
    du_dt = rolling_slope(df.assign(u_ok=u_ok), "track_id", "time_s", "u_ok", window)
    x_norm = (df.u_c - W / 2) / calib.focal_px
    static_du = df.g_smooth * df.d_eff * (df.u_c - df.u_foe) + df.yaw_smooth * (1 + x_norm ** 2)
    df["v_lat_mps"] = (du_dt - static_du) * calib.lateral_height_m / df.d_eff
    df["closing_mps"] = df.ego_speed_mps - df.v_along_mps
    return df
