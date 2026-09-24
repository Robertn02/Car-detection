"""Apply the trained typology models and compute rider-exposure outputs per video.

Outputs per video in results/corpus/<video>/:
  tracks_typed.csv     one row per vehicle track with relation, group, confidence and key measurements
  timeline_1s.csv      per-second scene, ego speed, vehicles visible by relation, closest distances, and - when
                       bikesafe.infrastructure has run - bike lane, signal state and signs in view (Strava-joinable)
  events.csv           overtakes, close ego-lane vehicles, near crossing traffic, close oncoming, emergency candidates,
                       stop signs passed and waits at red lights
  signs.csv            one row per distinct traffic sign / signal head seen
and results/corpus/corpus_summary.csv across videos.

Riding context: CLIP separates paths from roads well but confuses painted bike lanes with shared roads (bike-lane F1
0.49). Where the lane-paint detector has evidence, it decides bike lane vs shared road; CLIP keeps the separated-path
call and fills seconds without paint evidence. The CLIP-only context is kept as `scene_clip`.

    python -m bikesafe.exposure
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from bikesafe.common import RELATION_GROUP, RELATIONS, ROOT, SCENES, read_json
from bikesafe.infrastructure import object_instances, per_second_summary
from bikesafe.scene import scene_probabilities
from bikesafe.tracks import rule_relation

MIN_VISIBLE_BOX_H = 20
RELIABLE_Z_M = 15.0  # relations are validated within this range (results/typology/relation_accuracy_by_distance.csv)
OVERTAKE_MAX_Z = 8.0  # metres; a passing vehicle is alongside the rider
EMERGENCY_SHORTLIST = 10
SEPARATED_PATH_KEEP = 0.5  # CLIP probability above which the separated-path call stands whatever the paint says
RED_WAIT_MIN_S = 3  # a stop of at least this long with a red signal ahead counts as a wait at a red light
MIN_SIGN_BOX_H = 18  # px; smaller sign detections are too far away to count as passed


def unknown_scene(n_seconds: int) -> pd.DataFrame:
    """Scene table for a ride processed without CLIP (perceive --no-clip): no riding-context probabilities, so the
    context is 'unknown' except where the lane-paint detector decides it."""
    frame = pd.DataFrame({"time_s": np.arange(n_seconds, dtype=float)})
    for s in SCENES:
        frame[f"p_{s}"] = np.nan
    frame["scene"] = "unknown"
    return frame


def load_models(models: Path) -> tuple[dict | None, object | None]:
    relation = pickle.loads((models / "relation_model.pkl").read_bytes()) if (models / "relation_model.pkl").exists() else None
    scene = pickle.loads((models / "scene_probe.pkl").read_bytes()) if (models / "scene_probe.pkl").exists() else None
    return relation, scene


def type_tracks(tracks: pd.DataFrame, scene: pd.DataFrame, relation_model: dict | None) -> pd.DataFrame:
    """Model features keep the zero-shot scene scores stored by bikesafe.tracks (as in training); the trained scene
    probe only supplies the reported riding context."""
    tracks = tracks.copy()
    mid = ((tracks.t_start + tracks.t_end) / 2).to_numpy()
    pos = np.clip(np.searchsorted(scene.time_s.to_numpy(), mid), 0, len(scene) - 1)
    tracks["ride_scene"] = scene.scene.to_numpy()[pos]
    tracks["rule_relation"] = tracks.apply(rule_relation, axis=1)
    for flag in ("enter_edge", "exit_edge"):
        tracks[flag] = tracks[flag].astype(float)
    if relation_model is None:
        tracks["relation"] = tracks.rule_relation
        tracks["relation_confidence"] = np.nan
        tracks["relation_source"] = "rules"
    else:
        X = tracks.reindex(columns=relation_model["features"]).astype(float).to_numpy()
        proba = relation_model["model"].predict_proba(X)
        classes = np.array(relation_model["model"].classes_)
        tracks["relation"] = classes[proba.argmax(axis=1)]
        tracks["relation_confidence"] = proba.max(axis=1)
        for i, c in enumerate(classes):
            tracks[f"proba_{c}"] = proba[:, i]
        tracks["relation_source"] = relation_model["name"]
    tracks["group"] = tracks.relation.map(RELATION_GROUP)
    return tracks


def fuse_scene(line: pd.DataFrame) -> pd.Series:
    """Riding context per second: CLIP decides separated path, the lane-paint detector decides bike lane vs shared
    road wherever it has evidence, and CLIP fills the rest."""
    scene = line.scene_clip.copy()
    separated = (line.p_separated_path >= SEPARATED_PATH_KEEP).fillna(False).to_numpy(bool)
    in_lane = pd.to_numeric(line.in_bike_lane, errors="coerce").to_numpy(float)  # True 1, False 0, unknown NaN
    scene[~separated & (in_lane == 1.0)] = "bike_lane"
    scene[~separated & (in_lane == 0.0)] = "shared_road"
    return scene


def timeline(kin: pd.DataFrame, tracks: pd.DataFrame, ego: pd.DataFrame, scene: pd.DataFrame, start_local: str | None,
             infra: pd.DataFrame | None = None) -> pd.DataFrame:
    df = kin.merge(tracks[["track_id", "relation"]], on="track_id", how="left")
    df = df[df.box_h >= MIN_VISIBLE_BOX_H].copy()
    df["second"] = df.time_s.astype(int)
    seconds = pd.RangeIndex(0, int(max(ego.time_s.max(), 0)) + 1, name="second")
    out = pd.DataFrame(index=seconds)
    ego_s = ego.assign(second=ego.time_s.astype(int)).groupby("second")
    out["ego_speed_mps"] = ego_s.ego_speed_mps.median()
    out["ego_moving"] = ego_s.ego_moving.mean() > 0.5
    scene_s = scene.assign(second=scene.time_s.astype(int)).groupby("second")
    for s in SCENES:
        out[f"p_{s}"] = scene_s[f"p_{s}"].mean()
    probs = out[[f"p_{s}" for s in SCENES]]
    out["scene"] = "unknown"
    known = probs.notna().any(axis=1)
    out.loc[known, "scene"] = probs[known].idxmax(axis=1).str.removeprefix("p_")
    if infra is not None:
        out["scene_clip"] = out["scene"]
        out = out.join(infra, how="left")
        out["scene"] = fuse_scene(out)
    counts = df.groupby(["second", "relation"]).track_id.nunique().unstack(fill_value=0)
    for rel in RELATIONS:
        out[f"n_{rel}"] = counts[rel] if rel in counts else 0
    out = out.fillna({f"n_{rel}": 0 for rel in RELATIONS})
    near = df[df.z_m < 20]
    out["min_z_ego_lane_m"] = df[df.relation == "ego_lane"].groupby("second").z_m.min()
    out["min_lateral_adjacent_m"] = near[near.relation == "adjacent_same"].groupby("second").x_inner_m.apply(lambda s: s.abs().min())
    out["min_lateral_oncoming_m"] = near[near.relation == "oncoming"].groupby("second").x_inner_m.apply(lambda s: s.abs().min())
    door = df[(df.relation == "parked") & (df.x_inner_m > 0) & (df.x_inner_m < 1.5) & (df.z_m < 10)]
    out["door_zone_parked"] = door.groupby("second").track_id.nunique()
    out["door_zone_parked"] = out.door_zone_parked.fillna(0).astype(int)
    if start_local:
        t0 = datetime.fromisoformat(start_local)
        out.insert(0, "timestamp_local", [(t0 + timedelta(seconds=int(s))).isoformat() for s in out.index])
    return out.reset_index()


def events(tracks: pd.DataFrame, kin: pd.DataFrame) -> pd.DataFrame:
    """Safety events. Overtakes are detected from geometry (a vehicle that appears close, moves away faster than the
    rider and passes on one side), so they do not depend on the relation label being right."""
    rows = []
    for t in tracks.itertuples():
        base = {"track_id": t.track_id, "relation": t.relation, "t_start": round(t.t_start, 2), "t_end": round(t.t_end, 2),
                "z_min_m": round(t.z_min, 1), "x_inner_absmin_m": round(t.x_inner_absmin, 2)}
        # A real overtake passes beside the rider, so it must come close; otherwise "clearance" is just distance.
        overtaking = (t.z_first < 15 and t.z_min <= OVERTAKE_MAX_Z and t.log_growth < -0.15 and t.n_det >= 4
                      and t.ego_speed_med > 1.0 and t.v_along_med > t.ego_speed_med + 0.5)
        if overtaking:
            clearance = abs(t.x_inner_at_min_z)
            rows.append({**base, "event": "overtaken_by_vehicle", "clearance_m": round(clearance, 2),
                         "side": "left" if t.x_at_min_z < 0 else "right",
                         "speed_difference_mps": round(t.v_along_med - t.ego_speed_med, 1)})
            if clearance < 1.5:
                rows.append({**base, "event": "close_pass_under_1p5m", "clearance_m": round(clearance, 2),
                             "side": "left" if t.x_at_min_z < 0 else "right"})
        if t.relation == "ego_lane" and t.z_min < 6:
            rows.append({**base, "event": "vehicle_close_in_my_lane"})
        if t.relation == "cross_side" and t.z_min < 12:
            rows.append({**base, "event": "crossing_traffic_near"})
        if t.relation == "oncoming" and t.x_inner_absmin < 1.5 and t.z_min < 15:
            rows.append({**base, "event": "oncoming_close_pass"})
    frame = pd.DataFrame(rows)
    # The zero-shot emergency score is not comparable across videos or lighting, so flag a short review list
    # (the highest-scoring large vehicles) instead of thresholding it as a detection.
    if "emergency" not in tracks.columns:  # ride processed without CLIP
        return frame
    candidates = tracks[(tracks.emergency.fillna(0) >= 0.9) & (tracks.box_h_max >= 120)]
    candidates = candidates.nlargest(min(EMERGENCY_SHORTLIST, len(candidates)), "emergency")
    if len(candidates):
        extra = pd.DataFrame({
            "track_id": candidates.track_id, "relation": candidates.relation,
            "t_start": candidates.t_start.round(2), "t_end": candidates.t_end.round(2),
            "z_min_m": candidates.z_min.round(1), "x_inner_absmin_m": candidates.x_inner_absmin.round(2),
            "event": "emergency_vehicle_review", "score": candidates.emergency.round(2)})
        frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def infrastructure_events(signs: pd.DataFrame, line: pd.DataFrame) -> pd.DataFrame:
    """Stop signs passed (one row per distinct sign large enough to be close) and waits at red lights.

    A wait is a stop of RED_WAIT_MIN_S or more during which a red signal facing the rider was read for at least two
    seconds. It is defined by the stop, not by the signal readings, because buses and trucks crossing in front hide the
    signal head for seconds at a time and would otherwise split one wait into several.
    """
    rows = []
    if len(signs):
        stops = signs[(signs.category == "stop") & (signs.box_h_max >= MIN_SIGN_BOX_H)]
        for s in stops.itertuples():
            rows.append({"event": "stop_sign", "t_start": round(s.t_first, 2), "t_end": round(s.t_last, 2),
                         "score": round(s.conf_max, 2), "object_id": int(s.object_id)})
    if "traffic_light_state" in line:
        stopped = ~line.ego_moving.astype(bool).to_numpy()
        bridged = stopped.copy()
        bridged[1:-1] |= stopped[:-2] & stopped[2:]  # a single "moving" second while creeping forward
        red = (line.traffic_light_state == "red").to_numpy()
        start = None
        for i, flag in enumerate(np.append(bridged, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                red_seconds = int(red[start:i].sum())
                if i - start >= RED_WAIT_MIN_S and red_seconds >= 2:
                    rows.append({"event": "red_light_wait", "t_start": float(line.second.iloc[start]),
                                 "t_end": float(line.second.iloc[i - 1]) + 1.0, "duration_s": float(i - start),
                                 "red_seconds": red_seconds})
                start = None
    return pd.DataFrame(rows)


def summarise(video: str, tracks: pd.DataFrame, line: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """One row per video. Counts cover vehicles that came within the range where the relation is validated and that
    lasted at least a second, so neither distant guesses nor brief fragments inflate them."""
    relevant = tracks[(tracks.box_h_max >= 35) & (tracks.duration_s >= 1.0) & (tracks.z_min <= RELIABLE_Z_M)]
    minutes = len(line) / 60
    row = {"video": video, "minutes": round(minutes, 2), "moving_minutes": round(line.ego_moving.sum() / 60, 2),
           "median_moving_speed_mps": round(line.loc[line.ego_moving, "ego_speed_mps"].median(), 2)}
    for s in SCENES:
        row[f"minutes_{s}"] = round((line.scene == s).sum() / 60, 2)
    for rel in RELATIONS:
        n = int((relevant.relation == rel).sum())
        row[f"tracks_{rel}"] = n
        row[f"per_min_{rel}"] = round(n / max(minutes, 1e-6), 2)
    for name in ["overtaken_by_vehicle", "close_pass_under_1p5m", "vehicle_close_in_my_lane", "crossing_traffic_near",
                 "oncoming_close_pass", "emergency_vehicle_review"]:
        row[f"events_{name}"] = int((ev.event == name).sum()) if len(ev) else 0
    row["door_zone_seconds"] = int((line.door_zone_parked > 0).sum())
    overtakes = ev[ev.event == "overtaken_by_vehicle"] if len(ev) else ev
    row["median_overtake_clearance_m"] = round(float(overtakes.clearance_m.median()), 2) if len(overtakes) else float("nan")
    if "in_bike_lane" in line:
        in_lane = pd.to_numeric(line.in_bike_lane, errors="coerce")
        row["minutes_in_painted_bike_lane"] = round(float((in_lane == 1).sum()) / 60, 2)
        row["minutes_lane_paint_unknown"] = round(float(in_lane.isna().sum()) / 60, 2)
        row["minutes_scene_changed_by_paint"] = round(float((line.scene != line.scene_clip).sum()) / 60, 2)
        waits = ev[ev.event == "red_light_wait"] if len(ev) else ev
        row["events_stop_sign"] = int((ev.event == "stop_sign").sum()) if len(ev) else 0
        row["events_red_light_wait"] = int(len(waits))
        row["red_light_wait_seconds"] = int(waits.duration_s.sum()) if len(waits) else 0
        row["seconds_with_signal_ahead"] = int((line.n_signal_heads > 0).sum())
    return row


def summarise_by_scene(video: str, tracks: pd.DataFrame, line: pd.DataFrame) -> pd.DataFrame:
    """Exposure per riding context: how much traffic of each kind the rider meets per minute of bike lane,
    shared road and separated path."""
    relevant = tracks[(tracks.box_h_max >= 35) & (tracks.duration_s >= 1.0) & (tracks.z_min <= RELIABLE_Z_M)].copy()
    seconds = line.set_index("second").scene
    mid = ((relevant.t_start + relevant.t_end) / 2).round().astype(int).clip(0, int(seconds.index.max()))
    relevant["scene"] = seconds.reindex(mid).to_numpy()
    rows = []
    for scene in SCENES:
        minutes = (line.scene == scene).sum() / 60
        if minutes <= 0:
            continue
        subset = relevant[relevant.scene == scene]
        row = {"video": video, "scene": scene, "minutes": round(minutes, 2),
               "median_speed_mps": round(line.loc[(line.scene == scene) & line.ego_moving, "ego_speed_mps"].median(), 2)}
        for rel in RELATIONS:
            row[f"per_min_{rel}"] = round(int((subset.relation == rel).sum()) / minutes, 2)
        row["door_zone_share"] = round(float((line.loc[line.scene == scene, "door_zone_parked"] > 0).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "typology")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "corpus")
    args = parser.parse_args()
    relation_model, scene_probe = load_models(args.models)
    summaries, scene_rows = [], []
    for analysis_dir in sorted(p for p in args.analysis.iterdir() if (p / "tracks.parquet").exists()):
        video = analysis_dir.name
        calib = read_json(analysis_dir / "calibration.json")
        kin = pd.read_parquet(analysis_dir / "kinematics.parquet")
        ego = pd.read_parquet(analysis_dir / "ego.parquet")
        clip_path = args.perception / video / "clip.npz"
        scene = (scene_probabilities(clip_path, calib["source_fps"], probe=scene_probe) if clip_path.exists()
                 else unknown_scene(int(max(ego.time_s.max(), 0)) + 1))
        tracks = type_tracks(pd.read_parquet(analysis_dir / "tracks.parquet"), scene, relation_model)
        infra, signs = None, pd.DataFrame()
        if (analysis_dir / "lanes.parquet").exists():
            objects = pd.read_parquet(analysis_dir / "infrastructure.parquet")
            infra = per_second_summary(pd.read_parquet(analysis_dir / "lanes.parquet"), objects,
                                       int(max(ego.time_s.max(), 0)) + 1)
            signs = object_instances(objects[objects.kind.isin(["light", "sign"])])
        line = timeline(kin, tracks, ego, scene, calib.get("start_local"), infra)
        ev = events(tracks, kin)
        if infra is not None:
            ev = pd.concat([ev, infrastructure_events(signs, line)], ignore_index=True)
        out = args.out / video
        out.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(out / "tracks_typed.csv", index=False)
        line.to_csv(out / "timeline_1s.csv", index=False)
        ev.to_csv(out / "events.csv", index=False)
        if infra is not None:
            signs.to_csv(out / "signs.csv", index=False)
        summaries.append(summarise(video, tracks, line, ev))
        scene_rows.append(summarise_by_scene(video, tracks, line))
        print(video, tracks.relation.value_counts().to_dict())
    pd.DataFrame(summaries).to_csv(args.out / "corpus_summary.csv", index=False)
    pd.concat(scene_rows, ignore_index=True).to_csv(args.out / "exposure_by_scene.csv", index=False)
    print(pd.DataFrame(summaries).T.to_string())


if __name__ == "__main__":
    main()
