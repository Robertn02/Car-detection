"""Per-track features and the transparent rule-based rider-relation baseline.

    python -m bikesafe.tracks work/perception/VID_... [--out work/analysis]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bikesafe.common import ROOT, SCENES, read_json, write_json
from bikesafe.geometry import DEFAULT_HFOV_DEG, detection_kinematics, ego_from_tracking, ego_motion, fit_horizon
from bikesafe.scene import crop_scores, scene_probabilities

STATIC_MPS = 1.5
CROSS_MPS = 1.5


def q(series: pd.Series, p: float) -> float:
    s = series.dropna()
    return float(s.quantile(p)) if len(s) else float("nan")


def world_trajectory(t: pd.DataFrame) -> dict:
    """Least-squares velocity of the vehicle in the world, from the rider-frame track.

    The rider's own pose is removed first: positions are rotated back by how far the rider has turned since the
    track started and shifted by how far it has advanced. Fitting the whole track is far steadier than differencing
    consecutive frames. Heading is 0 deg travelling with the rider, 180 deg oncoming, 90 deg crossing.
    """
    usable = t[t.geom_ok & t.x_center_m.notna() & t.z_m.notna() & t.advance_m.notna()]
    out = {"world_speed": np.nan, "world_heading_deg": np.nan, "world_heading_abs": np.nan,
           "world_speed_ratio": np.nan, "world_fit_span_s": 0.0}
    if len(usable) < 5:
        return out
    turn = usable.turn_rad - usable.turn_rad.iloc[0]
    advance = usable.advance_m - usable.advance_m.iloc[0]
    x_world = usable.x_center_m + turn * usable.z_m
    z_world = usable.z_m + advance - turn * usable.x_center_m
    times = usable.time_s - usable.time_s.mean()
    denom = float((times ** 2).sum())
    span = float(usable.time_s.max() - usable.time_s.min())
    if denom <= 0 or span < 0.4:
        return out
    vx = float((times * (x_world - x_world.mean())).sum() / denom)
    vz = float((times * (z_world - z_world.mean())).sum() / denom)
    heading = float(np.degrees(np.arctan2(vx, vz)))
    ego_speed = float(usable.ego_speed_mps.median())
    out.update(world_speed=float(np.hypot(vx, vz)), world_heading_deg=heading, world_heading_abs=abs(heading),
               world_speed_ratio=float(np.hypot(vx, vz) / max(ego_speed, 0.5)), world_fit_span_s=span)
    return out


def aggregate_track(t: pd.DataFrame, width: int, height: int) -> dict:
    va, vl = t.v_along_mps, t.v_lat_mps
    valid = va.notna() & vl.notna()
    static = (va.abs() < STATIC_MPS) & (vl.abs() < CROSS_MPS)
    crossing = (vl.abs() >= CROSS_MPS) & (vl.abs() > va.abs())
    first, last = t.iloc[0], t.iloc[-1]
    return {
        "class_id": int(t.class_id.mode().iloc[0]),
        "n_det": len(t),
        "t_start": float(first.time_s),
        "t_end": float(last.time_s),
        "duration_s": float(last.time_s - first.time_s),
        "conf_median": float(t.confidence.median()),
        "box_h_median": float(t.box_h.median()),
        "box_h_max": float(t.box_h.max()),
        "aspect_median": float((t.box_w / t.box_h).median()),
        "u_rel_median": float(((t.u_c - t.u_foe) / width).median()),
        "u_rel_first": float((first.u_c - first.u_foe) / width),
        "u_rel_last": float((last.u_c - last.u_foe) / width),
        "d_rel_median": float((t.d_eff / height).median()),
        "x_center_med": q(t.x_center_m, 0.5),
        "x_center_p10": q(t.x_center_m, 0.1),
        "x_center_p90": q(t.x_center_m, 0.9),
        "x_inner_med": q(t.x_inner_m, 0.5),
        "x_inner_absmin": float(t.x_inner_m.abs().min()),
        "z_med": q(t.z_m, 0.5),
        "z_min": float(t.z_m.min()),
        "z_first": float(first.z_m),
        "z_last": float(last.z_m),
        "v_along_med": q(va, 0.5),
        "v_along_p10": q(va, 0.1),
        "v_along_p90": q(va, 0.9),
        "v_lat_med": q(vl, 0.5),
        "v_lat_absmed": q(vl.abs(), 0.5),
        "frac_static": float(static[valid].mean()) if valid.any() else float("nan"),
        "frac_cross": float(crossing[valid].mean()) if valid.any() else float("nan"),
        "frac_same": float((va[valid] > STATIC_MPS).mean()) if valid.any() else float("nan"),
        "frac_opposite": float((va[valid] < -STATIC_MPS).mean()) if valid.any() else float("nan"),
        "ego_speed_med": q(t.ego_speed_mps, 0.5),
        "ego_moving_frac": float(t.ego_moving.fillna(False).astype(bool).mean()),
        "closing_med": q(t.closing_mps, 0.5),
        "frac_trunc_bottom": float(t.trunc_bottom.mean()),
        "frac_trunc_side": float((t.trunc_left | t.trunc_right).mean()),
        "enter_edge": bool(first.trunc_left or first.trunc_right or first.trunc_bottom),
        "exit_edge": bool(last.trunc_left or last.trunc_right or last.trunc_bottom),
        "log_growth": float(np.log(max(last.box_h, 1) / max(first.box_h, 1))),
        "x_at_min_z": float(t.x_center_m.iloc[int(np.argmin(t.z_m.to_numpy()))]),
        "x_inner_at_min_z": float(t.x_inner_m.iloc[int(np.argmin(t.z_m.to_numpy()))]),
        "frac_close_pass": float((t.x_inner_m.abs() < 1.5).mean()),
        "v_along_ratio": float(q(va, 0.5) / max(q(t.ego_speed_mps, 0.5), 0.5)),
        "v_lat_ratio": float(q(vl.abs(), 0.5) / max(q(t.ego_speed_mps, 0.5), 0.5)),
        "det_rate": float(len(t) / max(last.time_s - first.time_s, 1e-3)),
        "aspect_p90": float((t.box_w / t.box_h).quantile(0.9)),
        "geom_valid_frac": float(t.geom_ok.mean()),
        "abs_x_at_min_z": float(abs(t.x_center_m.iloc[int(np.argmin(t.z_m.to_numpy()))])),
        **world_trajectory(t),
    }


def feature(row: pd.Series, key: str) -> float:
    value = row.get(key, np.nan)
    return float(value) if value is not None else float("nan")


def rule_relation(row: pd.Series) -> str:
    """Transparent baseline: what the vehicle does in the world, then where it sits relative to the rider.

    With the rider's own motion removed, a vehicle's world heading is nearly the typology by itself - 0 deg travels
    with the rider, 180 deg comes at it, 90 deg crosses - and a vehicle that is not moving at all is parked. Lane
    lines between rider and vehicle separate "my lane" from "the next lane". Thresholds come from the class-conditional
    statistics of the labelled tracks (results/typology), not from hand guesses.
    """
    speed = feature(row, "world_speed")
    heading = feature(row, "world_heading_abs")
    lines = feature(row, "lines_between_med")
    x = feature(row, "x_at_min_z")
    if np.isfinite(speed) and np.isfinite(heading):
        if speed < STATIC_MPS:
            return "parked"
        if heading > 120:
            return "oncoming"
        if heading > 50:
            return "cross_side"
        if (not np.isfinite(lines) or lines < 1) and abs(x) < 2.5:
            return "ego_lane"
        return "adjacent_same"

    # short tracks with no usable trajectory fit: fall back to per-frame rates
    along = feature(row, "v_along_ratio")
    lateral = feature(row, "v_lat_ratio")
    if not np.isfinite(along) or not np.isfinite(x):
        return "parked"
    rear, front = feature(row, "view_rear"), feature(row, "view_front")
    if abs(along) < 0.3 and (not np.isfinite(lateral) or lateral < 0.8):
        return "parked"
    if along < -0.4 and (np.isfinite(front) and front > rear or x < -6):
        return "oncoming"
    if np.isfinite(lateral) and lateral > 1.0 and abs(along) < 0.6:
        return "cross_side"
    if along > 0.3:
        if abs(x) < 1.8 and (not np.isfinite(rear) or rear >= front):
            return "ego_lane"
        return "adjacent_same"
    return "cross_side"


def neighbour_features(tracks: pd.DataFrame, window_s: float = 10.0, lateral_m: float = 2.0) -> pd.DataFrame:
    """How many other stationary vehicles sit at the same lateral offset at the same time.

    Parked cars come in rows, so a parked car has parked neighbours; a car stopped in a traffic lane usually does not.
    """
    if not len(tracks):
        return pd.DataFrame()
    x = tracks.x_at_min_z.to_numpy(float)
    t0, t1 = tracks.t_start.to_numpy(float), tracks.t_end.to_numpy(float)
    static = (tracks.world_speed < STATIC_MPS).fillna(False).to_numpy() if "world_speed" in tracks else np.zeros(len(tracks), bool)
    counts = np.zeros(len(tracks))
    for i in range(len(tracks)):
        if not np.isfinite(x[i]):
            continue
        overlapping = (t0 < t1[i] + window_s) & (t1 > t0[i] - window_s)
        beside = np.abs(x - x[i]) < lateral_m
        same_side = np.sign(x) == np.sign(x[i])
        counts[i] = int((overlapping & beside & same_side & static).sum()) - (1 if static[i] else 0)
    return pd.DataFrame({"track_id": tracks.track_id.to_numpy(), "static_neighbours": np.maximum(counts, 0)})


def scene3d_features(scene3d: pd.DataFrame, ego: pd.DataFrame, source_fps: float, focal_px: float,
                     max_gap_s: float = 1.5, moving_mps: float = 1.5) -> pd.DataFrame:
    """Per-track bird's-eye geometry: distance, lateral offset, lane lines crossed, and world heading.

    Heading comes from consecutive depth samples with the rider's own motion removed: the rider advances by the
    integral of its speed and turns by the integral of its yaw, so what is left is the vehicle's own displacement.
    0 deg travels with the rider, 180 deg is oncoming, 90 deg crosses.
    """
    if scene3d.empty:
        return pd.DataFrame()
    ego = ego.sort_values("frame")
    dt_ego = np.gradient(ego.time_s.to_numpy()) if len(ego) > 1 else np.array([1.0 / source_fps])
    advance = np.cumsum(np.nan_to_num(ego.ego_speed_mps.to_numpy()) * dt_ego)
    turn = np.cumsum(np.nan_to_num(ego.yaw_smooth.to_numpy()) * dt_ego) / focal_px  # radians
    frames = ego.frame.to_numpy()

    df = scene3d.sort_values(["track_id", "frame"]).copy()
    pos = np.clip(np.searchsorted(frames, df.frame.to_numpy()), 0, len(frames) - 1)
    df["advance_m"] = advance[pos]
    df["turn_rad"] = turn[pos]
    df["time_s"] = df.frame / source_fps

    group = df.groupby("track_id", sort=False)
    dt = group.time_s.diff()
    d_turn = group.turn_rad.diff()
    d_advance = group.advance_m.diff()
    x_prev, z_prev = group.bev_x.shift(), group.z_metric.shift()
    # express the current measurement in the rider's earlier frame (small-angle rotation), then remove the rider's advance
    x_now = df.bev_x + d_turn * df.z_metric
    z_now = df.z_metric - d_turn * df.bev_x
    dx = x_now - x_prev
    dz = z_now - z_prev + d_advance
    valid = dt.between(1e-3, max_gap_s) & dx.notna() & dz.notna()
    df["world_speed"] = (np.hypot(dx, dz) / dt).where(valid)
    df["heading_deg"] = np.degrees(np.arctan2(dx, dz)).where(valid)
    moving = df.world_speed > moving_mps

    # world-frame trajectory of each vehicle, expressed in the rider's pose at the track's first sample
    # (turn and advance must be measured from that moment, not from the start of the video)
    turn_rel = df.turn_rad - group.turn_rad.transform("first")
    advance_rel = df.advance_m - group.advance_m.transform("first")
    df["x_world"] = df.bev_x + turn_rel * df.z_metric
    df["z_world"] = df.z_metric + advance_rel - turn_rel * df.bev_x

    rows = []
    for track_id, t in df.groupby("track_id", sort=True):
        heading = t.heading_deg[moving.loc[t.index]].dropna()
        lanes = t.lines_between[t.lines_between >= 0]
        closest = t.z_metric.idxmin() if t.z_metric.notna().any() else None
        # least-squares velocity over the whole track: much steadier than differencing consecutive samples
        fit = t[["time_s", "x_world", "z_world"]].dropna()
        speed_fit = heading_fit = np.nan
        if len(fit) >= 3 and fit.time_s.max() - fit.time_s.min() >= 0.4:
            times = fit.time_s - fit.time_s.mean()
            denom = float((times ** 2).sum())
            if denom > 0:
                vx = float((times * (fit.x_world - fit.x_world.mean())).sum() / denom)
                vz = float((times * (fit.z_world - fit.z_world.mean())).sum() / denom)
                speed_fit = float(np.hypot(vx, vz))
                heading_fit = float(np.degrees(np.arctan2(vx, vz)))
        rows.append({
            "track_id": int(track_id),
            "world_speed_fit": speed_fit,
            "heading_fit_deg": heading_fit,
            "heading_fit_abs": abs(heading_fit) if np.isfinite(heading_fit) else np.nan,
            "depth_z_min": float(t.z_metric.min()) if t.z_metric.notna().any() else np.nan,
            "depth_z_med": float(t.z_metric.median()),
            "depth_x_med": float(t.bev_x.median()),
            "depth_x_at_min_z": float(t.loc[closest, "bev_x"]) if closest is not None else np.nan,
            "world_speed_med": float(t.world_speed.median()) if t.world_speed.notna().any() else np.nan,
            "world_speed_p90": q(t.world_speed, 0.9),
            "frac_world_static": float((t.world_speed <= moving_mps).mean()) if t.world_speed.notna().any() else np.nan,
            "heading_abs_med": float(heading.abs().median()) if len(heading) else np.nan,
            "frac_heading_same": float((heading.abs() < 45).mean()) if len(heading) else np.nan,
            "frac_heading_opposite": float((heading.abs() > 135).mean()) if len(heading) else np.nan,
            "frac_heading_crossing": float(heading.abs().between(45, 135).mean()) if len(heading) else np.nan,
            "lines_between_med": float(lanes.median()) if len(lanes) else np.nan,
            "lines_between_min": float(lanes.min()) if len(lanes) else np.nan,
            "yellow_between_max": float(t.yellow_between[t.yellow_between >= 0].max()) if (t.yellow_between >= 0).any() else np.nan,
            "dist_to_line_med": q(t.dist_to_line_m, 0.5),
            "n_depth_samples": int(len(t)),
        })
    return pd.DataFrame(rows)


def analyze_video(perception_dir: Path, out_root: Path, scene_probe=None, hfov_deg: float | None = None) -> pd.DataFrame:
    meta = read_json(perception_dir / "meta.json")
    stem = meta["stem"]
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dets = pd.read_parquet(perception_dir / "detections.parquet")
    calib = fit_horizon(dets, meta["width"], meta["height"], meta["processed_fps"], meta["stride"],
                        hfov_deg=DEFAULT_HFOV_DEG if hfov_deg is None else hfov_deg)
    raw_ego = out_dir / "egomotion_raw.parquet"
    if raw_ego.exists():
        ego = ego_from_tracking(pd.read_parquet(raw_ego), calib)
    else:
        print(f"{stem}: no egomotion_raw.parquet (run bikesafe.egomotion); falling back to the coarse flow grid")
        flow = np.load(perception_dir / "flow_grid.npz")
        ego = ego_motion(flow["frame"], flow["grid"], calib)
    ego["time_s"] = ego.frame / meta["source_fps"]
    kin = detection_kinematics(dets, ego, calib)

    clip_path = perception_dir / "clip.npz"
    scene = scene_probabilities(clip_path, meta["source_fps"], probe=scene_probe) if clip_path.exists() else None
    rows = []
    for _, t in kin.groupby("track_id", sort=True):
        row = {"video": stem, "track_id": int(t.track_id.iloc[0]), **aggregate_track(t, calib.width, calib.height)}
        rows.append(row)
    tracks = pd.DataFrame(rows)
    if clip_path.exists():
        tracks = tracks.merge(crop_scores(clip_path), on="track_id", how="left")
    scene3d_path = out_dir / "scene3d.parquet"
    if scene3d_path.exists() and len(tracks):
        depth_features = scene3d_features(pd.read_parquet(scene3d_path), ego, meta["source_fps"], calib.focal_px)
        if len(depth_features):
            tracks = tracks.merge(depth_features, on="track_id", how="left")
    if scene is not None and len(tracks):
        mid = (tracks.t_start + tracks.t_end) / 2
        pos = np.clip(np.searchsorted(scene.time_s.to_numpy(), mid.to_numpy()), 0, len(scene) - 1)
        for s in SCENES:
            tracks[f"p_{s}"] = scene[f"p_{s}"].to_numpy()[pos]
        tracks["scene"] = scene.scene.to_numpy()[pos]
    if len(tracks):
        tracks = tracks.merge(neighbour_features(tracks), on="track_id", how="left")
    tracks["rule_relation"] = tracks.apply(rule_relation, axis=1) if len(tracks) else []

    kin.to_parquet(out_dir / "kinematics.parquet", index=False)
    ego.to_parquet(out_dir / "ego.parquet", index=False)
    tracks.to_parquet(out_dir / "tracks.parquet", index=False)
    if scene is not None:
        scene.to_parquet(out_dir / "scene.parquet", index=False)
    write_json(out_dir / "calibration.json", {**calib.as_dict(), "start_local": meta.get("start_local"),
                                              "source_fps": meta["source_fps"]})
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("perception_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--hfov-deg", type=float, default=None)
    args = parser.parse_args()
    for path in args.perception_dirs:
        tracks = analyze_video(path, args.out, hfov_deg=args.hfov_deg)
        print(path.name, len(tracks), "tracks")
        print(tracks.rule_relation.value_counts().to_string())


if __name__ == "__main__":
    main()
