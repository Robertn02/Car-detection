"""Apply the trained typology models and compute rider-exposure outputs per video.

Outputs per video in results/corpus/<video>/:
  tracks_typed.csv     one row per vehicle track with relation, group, confidence and key measurements
  timeline_1s.csv      per-second scene, ego speed, vehicles visible by relation, closest distances, and - when
                       bikesafe.infrastructure has run - bike lane, signal state, the types of the signs in view and
                       the posted speed limit (Strava-joinable)
  events.csv           overtakes, close ego-lane vehicles, near crossing traffic, close oncoming, emergency candidates,
                       stop signs passed and waits at red lights
  signs.csv            one row per distinct traffic sign / signal head seen, with the sign's type and text (partial
                       street names completed from those read in full elsewhere in the corpus, bikesafe.signs)
  vehicles.csv         one row per physical vehicle: its relation in words and how sure the model is, when it was
                       seen (local time), how close it came and on which side, its plate, and whether to check it -
                       the table to read first (tracks_typed.csv holds every track and all ~130 model features)
and results/corpus/corpus_summary.csv across videos.

Counts are per physical vehicle: bikesafe.stitch merges the duplicate boxes the detector puts on one pickup or van
(car + truck) and joins tracks the tracker split, and every vehicle gets one relation - the detection-weighted vote
of its tracks. `tracks_<relation>` keeps the per-track count for comparison with earlier results; `per_min_*`,
`vehicles_*`, the per-second counts and the events are per vehicle. When bikesafe.plates has run, every track carries
its vehicle's plate (`plate`) and the summary reports how the vehicle ids agree with the plates.

Riding context: CLIP's (the reviewed-frame probe). CLIP separates paths from roads well but often takes a painted
bike lane for a shared road (bike-lane F1 0.49). The lane-paint detector's call is reported beside it
(`in_bike_lane`) but does not override it: on the 307 labelled frames, letting the paint decide bike lane vs shared
road took accuracy from 0.886 to 0.50 (it called 136 of 232 shared-road frames bike lane - mostly the solid line of a
parking lane or a wide kerb lane), and no combination tried, a learned one included, beat CLIP alone.

    python -m bikesafe.exposure
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from bikesafe.common import (RELATION_GROUP, RELATION_TEXT, RELATIONS, ROOT, SCENES, VEHICLE_CLASS_TEXT, read_json,
                              write_json)
from bikesafe.infrastructure import object_instances, pass_complete, per_second_summary
from bikesafe.plates import identity_audit, plates_on_pieces, vehicle_plates
from bikesafe.scene import scene_probabilities
from bikesafe.signs import NAMED_SIGNS, corpus_street_names
from bikesafe.tracks import rule_relation

MIN_VISIBLE_BOX_H = 20
# Distances and speeds are in metres calibrated against GPS (bikesafe.gps --calibrate); these thresholds were set
# on the earlier, uncalibrated scale and are the same ones in true metres (x 1.6).
RELIABLE_Z_M = 25.0  # relations are validated within this range (results/typology/relation_accuracy_by_distance.csv)
OVERTAKE_MAX_Z = 13.0  # metres; a passing vehicle is alongside the rider
# An overtake is followed for at least a second (passing parked cars gave sub-second fragments that seemed to pull
# away), passes beside the rider within about a lane's width (a box straddling the rider's heading, side gap 0, is in
# the rider's path, not beside it) and no faster than any traffic here could (fragments gave up to 130 m/s).
OVERTAKE_MIN_DET, OVERTAKE_MAX_GAP_M, OVERTAKE_MAX_DV = 12, 3.5, 24.0
EGO_MOVING_MPS = 1.6  # the rider is moving above this speed
MIN_SIGN_SAMPLES = 2  # a sign counted in the summary was seen on at least this many samples (half a second apart)
SAME_STOP_S = 3.0  # stop-sign objects this close in time are one sign the object linker split
EMERGENCY_SHORTLIST = 10
RED_WAIT_MIN_S = 3  # a stop of at least this long with a red signal ahead counts as a wait at a red light
MIN_SIGN_BOX_H = 18  # px; smaller sign detections are too far away to count as passed
SIGN_SUMMARY = ["stop", "yield", "speed_limit", "bike", "parking", "pedestrian", "warning", "street_name", "bus_stop",
                "roadwork"]


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
    if "vehicle_id" not in tracks:  # tracks written before vehicle ids existed
        tracks["vehicle_id"] = tracks.track_id
    tracks["vehicle_relation"] = tracks.vehicle_id.map(vehicle_relations(tracks))
    return tracks


def vehicle_relations(tracks: pd.DataFrame) -> pd.Series:
    """One relation per vehicle: the detection-weighted mean of its tracks' class probabilities, or the relation of
    its longest track when the rule baseline typed them."""
    proba = [f"proba_{rel}" for rel in RELATIONS if f"proba_{rel}" in tracks]
    if proba:
        weighted = tracks[proba].mul(tracks.n_det, axis=0).groupby(tracks.vehicle_id).sum()
        return weighted.idxmax(axis=1).str.removeprefix("proba_")
    return tracks.sort_values("n_det").groupby("vehicle_id").relation.last()


def vehicle_table(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle: span, largest box, closest approach, number of tracks and its relation."""
    table = pd.DataFrame(tracks.groupby("vehicle_id").agg(
        t_start=("t_start", "min"), t_end=("t_end", "max"), box_h_max=("box_h_max", "max"), z_min=("z_min", "min"),
        n_tracks=("track_id", "size"), relation=("vehicle_relation", "first")))
    table["duration_s"] = table.t_end - table.t_start
    return table


CHECK_CONFIDENCE = 0.45  # below this vote share the relation is flagged for review (as the overlay's "?")


def vehicle_summary(tracks: pd.DataFrame, start_local: str | None) -> pd.DataFrame:
    """One row per physical vehicle, for people reading the results: what it was to the rider, how sure, when, how
    close, and whether a person should check it. Confidence is the share of the detection-weighted class vote that
    went to the relation (bikesafe.exposure.vehicle_relations)."""
    t = tracks.copy()
    proba = [f"proba_{r}" for r in RELATIONS if f"proba_{r}" in t]
    if proba:
        votes = t[proba].mul(t.n_det, axis=0).groupby(t.vehicle_id).sum()
        confidence = votes.max(axis=1) / votes.sum(axis=1).replace(0, np.nan)
    else:
        confidence = pd.Series(np.nan, index=pd.Index(t.vehicle_id.unique(), name="vehicle_id"))  # rule-typed ride
    weight = t.n_det.astype(float)
    # the row of its closest approach (a vehicle never placed on the road, all z_min empty, keeps its first row)
    closest = (t.sort_values("z_min", na_position="last", kind="stable").drop_duplicates("vehicle_id")
               .set_index("vehicle_id").sort_index())  # in vehicle order, as the columns beside it
    g = t.groupby("vehicle_id")
    out = pd.DataFrame({
        "relation": g.vehicle_relation.first(),
        "confidence": confidence.round(3),
        "vehicle_class": t.groupby(["vehicle_id", "class_id"]).n_det.sum().unstack(fill_value=0).idxmax(axis=1).map(
            lambda c: VEHICLE_CLASS_TEXT.get(int(c), "vehicle")),
        "t_start": g.t_start.min().round(2), "t_end": g.t_end.max().round(2),
        "closest_m": g.z_min.min().round(1),
        "lateral_at_closest_m": closest.x_inner_at_min_z.round(2),
        "side": np.where(closest.x_at_min_z < 0, "left", "right"),
        "box_h_max_px": g.box_h_max.max().round(0),
        "n_tracks": g.size(),
        "track_ids": g.track_id.agg(lambda ids: " ".join(str(int(i)) for i in sorted(ids))),
    })
    out.insert(1, "relation_text", out.relation.map(RELATION_TEXT))
    out.insert(2, "group", out.relation.map(RELATION_GROUP))
    out["duration_s"] = (out.t_end - out.t_start).round(2)
    if "frac_rider_stopped" in t:  # share of its detections seen while the rider was stopped
        out["seen_while_rider_stopped"] = (t.frac_rider_stopped * weight).groupby(t.vehicle_id).sum().div(
            weight.groupby(t.vehicle_id).sum()).round(2)
    if "not_vehicle" in t:
        out["not_vehicle_score"] = (t.not_vehicle * weight).groupby(t.vehicle_id).sum().div(
            weight.groupby(t.vehicle_id).sum()).round(2)
    if "plate" in t:
        out["plate"] = g.plate.first()
    out["counted"] = out.index.isin(counted(vehicle_table(t)).index)  # as the summary counts them
    reasons = np.where(out.confidence < CHECK_CONFIDENCE, "model unsure", "")
    if "not_vehicle_score" in out:
        reasons = np.where(out.not_vehicle_score >= 0.5, np.char.add(reasons.astype(str), "; may not be a vehicle"), reasons)
    out["check"] = pd.Series(reasons, index=out.index).str.strip("; ")
    if start_local:
        t0 = datetime.fromisoformat(start_local)
        out.insert(out.columns.get_loc("t_start"), "time_local",
                   [(t0 + timedelta(seconds=float(s))).isoformat(timespec="seconds") for s in out.t_start])
    order = ["vehicle_id", "time_local", "t_start", "t_end", "duration_s", "relation_text", "relation", "group",
             "confidence", "check", "counted", "vehicle_class", "closest_m", "lateral_at_closest_m", "side",
             "seen_while_rider_stopped", "not_vehicle_score", "plate", "box_h_max_px", "n_tracks", "track_ids"]
    out = out.reset_index().sort_values("t_start").reset_index(drop=True)
    return pd.DataFrame(out[[c for c in order if c in out]])


def counted(table: pd.DataFrame) -> pd.DataFrame:
    """Vehicles (or tracks) that came within the range where the relation is validated and lasted at least a second,
    so neither distant guesses nor brief fragments inflate the counts."""
    return pd.DataFrame(table[(table.box_h_max >= 35) & (table.duration_s >= 1.0) & (table.z_min <= RELIABLE_Z_M)])


def timeline(kin: pd.DataFrame, tracks: pd.DataFrame, ego: pd.DataFrame, scene: pd.DataFrame, start_local: str | None,
             infra: pd.DataFrame | None = None) -> pd.DataFrame:
    df = kin.drop(columns=["vehicle_id"], errors="ignore").merge(
        tracks[["track_id", "vehicle_id", "vehicle_relation"]].rename(columns={"vehicle_relation": "relation"}),
        on="track_id", how="left")
    df = df[df.box_h >= MIN_VISIBLE_BOX_H].copy()
    df["second"] = df.time_s.astype(int)
    seconds = pd.RangeIndex(0, int(max(ego.time_s.max(), 0)) + 1, name="second")
    out = pd.DataFrame(index=seconds)
    ego_s = ego.assign(second=ego.time_s.astype(int)).groupby("second")
    # a failed fit can come out below zero at a standstill; moving is judged from the same speed it reports
    out["ego_speed_mps"] = ego_s.ego_speed_mps.median().clip(lower=0.0)
    out["ego_moving"] = out.ego_speed_mps > EGO_MOVING_MPS
    scene_s = scene.assign(second=scene.time_s.astype(int)).groupby("second")
    for s in SCENES:
        out[f"p_{s}"] = scene_s[f"p_{s}"].mean()
    probs = out[[f"p_{s}" for s in SCENES]]
    out["scene"] = "unknown"
    known = probs.notna().any(axis=1)
    out.loc[known, "scene"] = probs[known].idxmax(axis=1).str.removeprefix("p_")
    if infra is not None:
        out = out.join(infra, how="left")
    counts = df.groupby(["second", "relation"]).vehicle_id.nunique().unstack(fill_value=0)
    for rel in RELATIONS:
        out[f"n_{rel}"] = counts[rel] if rel in counts else 0
    out = out.fillna({f"n_{rel}": 0 for rel in RELATIONS}).astype({f"n_{rel}": int for rel in RELATIONS})
    near = df[df.z_m < 32]
    out["min_z_ego_lane_m"] = df[df.relation == "ego_lane"].groupby("second").z_m.min()
    out["min_lateral_adjacent_m"] = near[near.relation == "adjacent_same"].groupby("second").x_inner_m.apply(lambda s: s.abs().min())
    out["min_lateral_oncoming_m"] = near[near.relation == "oncoming"].groupby("second").x_inner_m.apply(lambda s: s.abs().min())
    door = df[(df.relation == "parked") & (df.x_inner_m > 0) & (df.x_inner_m < 1.5) & (df.z_m < 16)]
    out["door_zone_parked"] = door.groupby("second").vehicle_id.nunique()
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
        base = {"track_id": t.track_id, "vehicle_id": t.vehicle_id, "relation": t.vehicle_relation,
                "t_start": round(t.t_start, 2), "t_end": round(t.t_end, 2), "z_min_m": round(t.z_min, 1),
                "x_inner_absmin_m": round(t.x_inner_absmin, 2)}
        # A real overtake passes beside the rider, so it must come close; otherwise "clearance" is just distance.
        clearance = abs(t.x_inner_at_min_z)
        overtaking = (t.z_first < RELIABLE_Z_M and t.z_min <= OVERTAKE_MAX_Z and t.log_growth < -0.15
                      and t.n_det >= OVERTAKE_MIN_DET and t.ego_speed_med > EGO_MOVING_MPS
                      and t.ego_speed_med + 0.8 < t.v_along_med <= t.ego_speed_med + OVERTAKE_MAX_DV
                      and 0 < clearance <= OVERTAKE_MAX_GAP_M)
        if overtaking:
            rows.append({**base, "event": "overtaken_by_vehicle", "clearance_m": round(clearance, 2),
                         "side": "left" if t.x_at_min_z < 0 else "right",
                         "speed_difference_mps": round(t.v_along_med - t.ego_speed_med, 1)})
            if clearance < 1.5:
                rows.append({**base, "event": "close_pass_under_1p5m", "clearance_m": round(clearance, 2),
                             "side": "left" if t.x_at_min_z < 0 else "right"})
        if t.vehicle_relation == "ego_lane" and t.z_min < 10:
            rows.append({**base, "event": "vehicle_close_in_my_lane"})
        if t.vehicle_relation == "cross_side" and t.z_min < 20:
            rows.append({**base, "event": "crossing_traffic_near"})
        if t.vehicle_relation == "oncoming" and t.x_inner_absmin < 1.5 and t.z_min < RELIABLE_Z_M:
            rows.append({**base, "event": "oncoming_close_pass"})
    frame = pd.DataFrame(rows)
    # The zero-shot emergency score is not comparable across videos or lighting, so flag a short review list
    # (the highest-scoring large vehicles) instead of thresholding it as a detection.
    if "emergency" not in tracks.columns:  # ride processed without CLIP
        return one_per_vehicle(frame)
    candidates = tracks[(tracks.emergency.fillna(0) >= 0.9) & (tracks.box_h_max >= 120)]
    candidates = candidates.nlargest(min(EMERGENCY_SHORTLIST, len(candidates)), "emergency")
    if len(candidates):
        extra = pd.DataFrame({
            "track_id": candidates.track_id, "vehicle_id": candidates.vehicle_id, "relation": candidates.vehicle_relation,
            "t_start": candidates.t_start.round(2), "t_end": candidates.t_end.round(2),
            "z_min_m": candidates.z_min.round(1), "x_inner_absmin_m": candidates.x_inner_absmin.round(2),
            "event": "emergency_vehicle_review", "score": candidates.emergency.round(2)})
        frame = pd.concat([frame, extra], ignore_index=True)
    return one_per_vehicle(frame)


def one_per_vehicle(frame: pd.DataFrame) -> pd.DataFrame:
    """A vehicle boxed twice, or split into several tracks, must not count twice: one row per event and vehicle,
    the one with the smallest clearance / distance."""
    if frame.empty or "vehicle_id" not in frame:
        return frame
    order = [c for c in ("clearance_m", "z_min_m") if c in frame]
    frame = frame.sort_values(order, na_position="last").drop_duplicates(["event", "vehicle_id"])
    return frame.sort_values(["t_start", "event"]).reset_index(drop=True)


def infrastructure_events(signs: pd.DataFrame, line: pd.DataFrame) -> pd.DataFrame:
    """Stop signs passed (one row per distinct sign large enough to be close) and waits at red lights.

    A wait is a stop of RED_WAIT_MIN_S or more during which a red signal facing the rider was read for at least two
    seconds. It is defined by the stop, not by the signal readings, because buses and trucks crossing in front hide the
    signal head for seconds at a time and would otherwise split one wait into several.
    """
    rows = []
    if len(signs):
        stops = signs[(signs.category == "stop") & (signs.box_h_max >= MIN_SIGN_BOX_H)].sort_values("t_first")
        for s in stops.itertuples():  # one event per sign: the object linker splits a sign seen at a junction
            if rows and s.t_first - rows[-1]["t_end"] <= SAME_STOP_S:
                rows[-1]["t_end"] = round(max(rows[-1]["t_end"], s.t_last), 2)
                rows[-1]["score"] = round(max(rows[-1]["score"], s.conf_max), 2)
                continue
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


def summarise(video: str, tracks: pd.DataFrame, line: pd.DataFrame, ev: pd.DataFrame,
              links: pd.DataFrame | None = None, audit: dict | None = None, signs: pd.DataFrame | None = None) -> dict:
    """One row per video. Counts cover vehicles that came within the range where the relation is validated and that
    lasted at least a second (see `counted`); per_min_* are vehicles, tracks_* the per-track counts of earlier runs."""
    relevant = counted(tracks)
    vehicles = counted(vehicle_table(tracks))
    minutes = len(line) / 60
    row = {"video": video, "minutes": round(minutes, 2), "moving_minutes": round(line.ego_moving.sum() / 60, 2),
           "median_moving_speed_mps": round(line.loc[line.ego_moving, "ego_speed_mps"].median(), 2)}
    for s in SCENES:
        row[f"minutes_{s}"] = round((line.scene == s).sum() / 60, 2)
    for rel in RELATIONS:
        n = int((vehicles.relation == rel).sum())
        row[f"vehicles_{rel}"] = n
        row[f"per_min_{rel}"] = round(n / max(minutes, 1e-6), 2)
        row[f"tracks_{rel}"] = int((relevant.relation == rel).sum())
    row["vehicles_counted"], row["tracks_counted"] = int(len(vehicles)), int(len(relevant))
    if links is not None:
        row["duplicate_links"] = int((links.kind == "duplicate").sum())
        row["break_links"] = int((links.kind == "gap").sum())
    for key, value in (audit or {}).items():
        row[f"plate_{key}"] = value
    for name in ["overtaken_by_vehicle", "close_pass_under_1p5m", "vehicle_close_in_my_lane", "crossing_traffic_near",
                 "oncoming_close_pass", "emergency_vehicle_review"]:
        row[f"events_{name}"] = int((ev.event == name).sum()) if len(ev) else 0
    row["door_zone_seconds"] = int((line.door_zone_parked > 0).sum())
    overtakes = ev[ev.event == "overtaken_by_vehicle"] if len(ev) else ev
    row["median_overtake_clearance_m"] = round(float(overtakes.clearance_m.median()), 2) if len(overtakes) else float("nan")
    if "in_bike_lane" in line:  # the infrastructure pass ran (the paint call itself is not summarised: see above)
        waits = ev[ev.event == "red_light_wait"] if len(ev) else ev
        if "speed_limit_posted" in line:
            limit = pd.to_numeric(line.speed_limit_posted, errors="coerce")
            row["posted_speed_limit_median"] = float(limit.median()) if limit.notna().any() else float("nan")
            row["minutes_posted_limit_known"] = round(float(limit.notna().sum()) / 60, 2)
        row["events_stop_sign"] = int((ev.event == "stop_sign").sum()) if len(ev) else 0
        row["events_red_light_wait"] = int(len(waits))
        row["red_light_wait_seconds"] = int(waits.duration_s.sum()) if len(waits) else 0
        row["seconds_with_signal_ahead"] = int((line.n_signal_heads > 0).sum())
    if signs is not None and len(signs):
        # seen on two samples or more: two in three signs seen once are part of a sign the linker split
        passed = signs[(signs.kind == "sign") & (signs.box_h_max >= MIN_SIGN_BOX_H) & (signs.n_samples >= MIN_SIGN_SAMPLES)]
        row["signs_total"] = int(len(passed))
        for category in SIGN_SUMMARY:
            row[f"signs_{category}"] = int((passed.category == category).sum())
    return row


def summarise_by_scene(video: str, tracks: pd.DataFrame, line: pd.DataFrame) -> pd.DataFrame:
    """Exposure per riding context: how much traffic of each kind the rider meets per minute of bike lane,
    shared road and separated path."""
    relevant = counted(vehicle_table(tracks)).copy()
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
    street_names = corpus_street_names(args.analysis)
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
        if (analysis_dir / "lanes.parquet").exists() and not pass_complete(analysis_dir):
            # a test window (--start-s / --duration-s) would give signals, signs and their events for part of the ride
            print(f"{video}: the infrastructure pass covered only part of the ride - left out of the tables")
        if pass_complete(analysis_dir):
            objects = pd.read_parquet(analysis_dir / "infrastructure.parquet")
            infra = per_second_summary(pd.read_parquet(analysis_dir / "lanes.parquet"), objects,
                                       int(max(ego.time_s.max(), 0)) + 1)
            signs = object_instances(objects[objects.kind.isin(["light", "sign"])])
            signs["sign_text_read"] = signs.sign_text  # what OCR read; sign_text completes partial street names
            named = signs.category.isin(NAMED_SIGNS) & (signs.sign_text.astype(str).str.len() > 0)
            signs.loc[named, "sign_text"] = signs.loc[named, "sign_text"].astype(str).map(street_names)
        line = timeline(kin, tracks, ego, scene, calib.get("start_local"), infra)
        ev = events(tracks, kin)
        if infra is not None:
            ev = pd.concat([ev, infrastructure_events(signs, line)], ignore_index=True)
        out = args.out / video
        out.mkdir(parents=True, exist_ok=True)
        links_path = analysis_dir / "vehicle_links.parquet"
        links = pd.read_parquet(links_path) if links_path.exists() else None
        audit = None
        plates = pd.read_parquet(analysis_dir / "plates.parquet") if (analysis_dir / "plates.parquet").exists() else None
        if plates is not None and "plate_text" in plates:  # tables from the earlier hashed audit carry no text
            plates = plates_on_pieces(plates, kin)
            spans = kin.groupby("track_id").time_s.agg(t_start="min", t_end="max")
            vehicle_of = (kin.groupby("track_id").vehicle_id.first() if "vehicle_id" in kin
                          else pd.Series(spans.index, index=spans.index)).astype(int).to_dict()
            audit = identity_audit(plates[plates.track_id.isin(spans.index)], vehicle_of, spans)
            write_json(out / "identity_audit.json", audit)
            tracks["plate"] = tracks.vehicle_id.map(vehicle_plates(plates, vehicle_of))
        tracks.to_csv(out / "tracks_typed.csv", index=False)
        vehicle_summary(tracks, calib.get("start_local")).to_csv(out / "vehicles.csv", index=False)
        line.to_csv(out / "timeline_1s.csv", index=False)
        ev.to_csv(out / "events.csv", index=False)
        if infra is not None:
            signs.to_csv(out / "signs.csv", index=False)
        summaries.append(summarise(video, tracks, line, ev, links, audit, signs if infra is not None else None))
        scene_rows.append(summarise_by_scene(video, tracks, line))
        print(video, tracks.relation.value_counts().to_dict())
    pd.DataFrame(summaries).to_csv(args.out / "corpus_summary.csv", index=False)
    pd.concat(scene_rows, ignore_index=True).to_csv(args.out / "exposure_by_scene.csv", index=False)
    print(pd.DataFrame(summaries).T.to_string())


if __name__ == "__main__":
    main()
