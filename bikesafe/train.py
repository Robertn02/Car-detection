"""Train and evaluate the rider-relation classifier and the scene probe with leave-one-video-out validation.

Compares (1) the hand-written rules, (2) a depth-limited decision tree on physical features (readable rules
learned from labels), (3) gradient boosting and (4) extra trees on the physics-defined features. The best model by
macro-F1 within NEAR_Z_M across held-out videos is refit on all labels and saved for bikesafe.exposure.
Also writes feature_ablation.csv (the feature sets of feature_sets(): version 2, + flow motion, version 3 = + lane
paint, + flow consistency, version 5 = + kerb context, + path, version 7 = + motion while the rider is stopped, and
version 8 - geometry, flow and stopped-rider motion - which the model uses) and scores the riding context for the
CLIP probe alone and with the lane-paint detector deciding bike lane vs shared road, when bikesafe.infrastructure has
run. Feature sets are compared over several seeds with scripts/compare_features.py, not on one ablation run.

    python -m bikesafe.train
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from bikesafe.common import RELATION_GROUP, RELATIONS, ROOT, VIDEOS, SCENES, is_cuda, open_video, read_json, resolve_video
from bikesafe.infrastructure import per_second_summary
from bikesafe.labels import align_track_labels, alignment_summary
from bikesafe.scene import scene_features, zero_shot
from bikesafe.tracks import rule_relation

PHYSICAL_FEATURES_V2 = [
    "x_center_med", "x_center_p10", "x_center_p90", "x_inner_med", "z_min", "z_med", "v_along_med", "v_along_p10",
    "v_along_p90", "v_lat_med", "v_lat_absmed", "frac_static", "frac_cross", "frac_same", "frac_opposite",
    "ego_speed_med", "ego_moving_frac", "x_at_min_z", "frac_close_pass", "v_along_ratio", "v_lat_ratio",
    "p_separated_path", "p_bike_lane", "p_shared_road", "view_rear", "view_front", "view_side",
    # bird's-eye geometry from learned depth, and the lane structure between rider and vehicle
    "world_speed", "world_heading_abs", "world_speed_ratio", "abs_x_at_min_z", "static_neighbours",
    "depth_z_min", "depth_x_at_min_z", "world_speed_fit", "heading_fit_abs", "frac_world_static",
    "frac_heading_same", "frac_heading_opposite", "frac_heading_crossing", "lines_between_med", "yellow_between_max",
]
# Version 3: depth-free motion from the stored optical flow (bikesafe.geometry.add_flow_motion). The most frequent
# v2 error was a moving vehicle called parked: 12 of 23 oncoming and 9 of 25 same-direction tracks within 15 m.
# Metres calibrated against GPS (bikesafe.gps --calibrate): the range where relations are judged and its bins are the
# earlier 15 m and 5/10/15/25 m on the uncalibrated scale, in true metres
NEAR_Z_M = 25.0
DISTANCE_BINS = [0, 8, 16, 25, 40, 1000]
FLOW_FEATURES = [
    "closing_ratio_med", "closing_ratio_p10", "closing_ratio_p90", "v_lat_flow_absmed", "v_lat_flow_absp90",
    "v_lat_flow_out_med", "frac_lat_flow_moving", "v_along_flow_med", "v_along_flow_ratio", "grid_proj_med",
    "grid_proj_p10",
]
# ... and where the vehicle sits relative to the painted lines and the rider's bike lane (bikesafe.infrastructure)
LANE_FEATURES = ["paint_lines_between_med", "paint_solid_between_max", "paint_yellow_between_max", "in_rider_lane_frac",
                 "rider_bike_lane_frac"]
PHYSICAL_FEATURES_V3 = PHYSICAL_FEATURES_V2 + FLOW_FEATURES + LANE_FEATURES
# Version 5: how consistently the flow says static / co-moving / approaching, and where the vehicle stands relative to
# the kerb (bikesafe.streetseg.vehicle_context). The remaining errors after v4 were mostly stationary vehicles that are
# not parked - queued beside the rider or waiting on a side street - which motion cannot tell from parked cars.
CONSISTENCY_FEATURES = ["frac_closing_static", "frac_closing_with", "frac_closing_against", "frac_grid_static",
                        "frac_grid_with", "frac_grid_against"]
CONTEXT_FEATURES = ["kerb_gap_med", "frac_kerb_close", "frac_outer_vehicle", "inner_road_med", "under_offroad_med"]
# Version 6: time in the rider's path ahead (and there while the rider waits) - queued traffic, not parked. Scored in
# the ablation but not used: with 13 near in-my-lane labels it lowered near macro-F1 (0.59 vs 0.64 for v5). The review
# queue (bikesafe.review) uses it to find candidates instead.
PATH_FEATURES = ["frac_in_path", "frac_in_path_rider_stopped", "in_path_z_min"]
# Version 7: the vehicle's own motion while the rider is stopped - queued, crossing and turning traffic moves then,
# parked cars do not (bikesafe.tracks.own_motion_while_stopped)
STOPPED_FEATURES = ["frac_rider_stopped", "stopped_own_speed_med", "stopped_own_speed_p90", "frac_stopped_own_moving"]
PHYSICAL_FEATURES_V5 = PHYSICAL_FEATURES_V3 + CONSISTENCY_FEATURES + CONTEXT_FEATURES
PHYSICAL_FEATURES_V7 = PHYSICAL_FEATURES_V5 + STOPPED_FEATURES
# Version 8 (used): geometry, flow and the vehicle's motion while the rider is stopped, without the lane-paint and kerb
# features. With the lane and kerb pass over whole rides (not a window), those lowered near macro-F1 (5 seeds, 321 near
# labels: v2+flow 0.631, v3 0.612, v5 0.600, v7 0.607) - the same noisy paint that calls shared roads bike lanes - while
# the stopped-rider features still help where they were made for (stopped-rider accuracy 0.818 -> 0.853). v8: accuracy
# 0.730, macro-F1 0.627, 39.6 moving vehicles called parked against 43.0 for v7.
PHYSICAL_FEATURES = PHYSICAL_FEATURES_V2 + FLOW_FEATURES + STOPPED_FEATURES
NON_FEATURES = {"video", "track_id", "t_start", "t_end", "scene", "ride_scene", "rule_relation", "label", "labeler", "notes",
                "card_id", "vehicle_id", "vehicle_tracks", "vehicle_relation", "labelled_track_id", "alignment"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def as_float(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).to_numpy()


def load_labelled_tracks(analysis: Path, labels_csv: Path, perception: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    labels = labels[labels.label.isin(RELATIONS)]
    # Labels name tracker ids of the perception run they were made on; follow each vehicle to its current id.
    labels = align_track_labels(labels, perception, analysis=analysis)
    print(f"track labels: {alignment_summary(labels)}")
    labels = labels[labels.alignment != "lost"]
    tracks = pd.concat([pd.read_parquet(p) for p in sorted(analysis.glob("*/tracks.parquet"))], ignore_index=True)
    merged = labels.merge(tracks, on=["video", "track_id"], how="inner")
    for flag in ("enter_edge", "exit_edge"):
        merged[flag] = merged[flag].astype(float)
    # Recompute the rule baseline so it always reflects the current rules, not whatever was stored earlier.
    merged["rule_relation"] = merged.apply(rule_relation, axis=1)
    return merged


def make_models() -> dict:
    """Candidates compared leave-one-video-out. Extra trees were added after a 5-seed check on the version-2
    features: a tie with gradient boosting within 15 m (68.3% / macro-F1 0.510 vs 68.1% / 0.503) and better over all
    distances on every seed (65.9% / 0.482 vs 63.0% / 0.432)."""
    return {
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=0),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0,
            class_weight="balanced", random_state=0),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                 random_state=0, n_jobs=-1)),
    }


def model_features(name: str, all_features: list[str], feature_list: list[str] | None = None) -> list[str]:
    """Both models use the physics-defined list. Throwing all ~80 columns at gradient boosting scored the same or
    worse in leave-one-video-out ablation, which is what 378 labels should be expected to support."""
    del name
    return [f for f in (feature_list or PHYSICAL_FEATURES) if f in all_features]


def lovo_predict(data: pd.DataFrame, model, cols: list[str]) -> np.ndarray:
    pred = np.empty(len(data), dtype=object)
    for video in sorted(data.video.unique()):
        test = (data.video == video).to_numpy()
        if (~test).sum() == 0:
            continue
        model.fit(as_float(data[~test], cols), data.label[~test])
        pred[test] = model.predict(as_float(data[test], cols))
    return pred


def feature_sets() -> dict[str, list[str]]:
    """The feature sets compared in the ablation (and by scripts/compare_features.py), by name."""
    return {"v2": PHYSICAL_FEATURES_V2, "v2+flow": PHYSICAL_FEATURES_V2 + FLOW_FEATURES, "v3": PHYSICAL_FEATURES_V3,
            "v3+consistency": PHYSICAL_FEATURES_V3 + CONSISTENCY_FEATURES, "v5": PHYSICAL_FEATURES_V5,
            "v5+path": PHYSICAL_FEATURES_V5 + PATH_FEATURES, "v7": PHYSICAL_FEATURES_V7, "v8": PHYSICAL_FEATURES}


def feature_ablation(data: pd.DataFrame, near: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Leave-one-video-out scores for each feature set (feature_sets()), for each model, so a re-run shows directly
    what the features are worth on these labels (one seed: compare sets over several with scripts/compare_features.py)."""
    available = set(feature_columns(data))
    sets = feature_sets()
    rows = []
    for subset_name, subset in (("all", data), ("near", near)):
        y = subset.label.to_numpy()
        seen: set[tuple[str, ...]] = set()
        for set_name, feature_list in sets.items():
            cols = [f for f in feature_list if f in available]
            if tuple(cols) in seen:
                continue  # adds no columns: tracks from an older pipeline, or no infrastructure pass (no lane paint)
            seen.add(tuple(cols))
            for name, model in make_models().items():
                pred = lovo_predict(subset, model, cols)
                rows.append({"subset": subset_name, "features": set_name, "n_features": len(cols), "model": name,
                             "accuracy": round(accuracy_score(y, pred), 4),
                             "macro_f1": round(f1_score(y, pred, labels=RELATIONS, average="macro", zero_division=0), 4),
                             "group_accuracy": round(accuracy_score(pd.Series(y).map(RELATION_GROUP),
                                                                    pd.Series(pred).map(RELATION_GROUP)), 4)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "feature_ablation.csv", index=False)
    return table


def evaluate_relations(data: pd.DataFrame, out: Path, tag: str = "") -> dict:
    features = feature_columns(data)
    videos = sorted(data.video.unique())
    predictions = {"rules": data.rule_relation.to_numpy()}
    for name, model in make_models().items():
        predictions[name] = lovo_predict(data, model, model_features(name, features))

    summary, per_video_rows = {}, []
    y = data.label.to_numpy()
    groups_true = pd.Series(y).map(RELATION_GROUP).to_numpy()
    for name, pred in predictions.items():
        groups_pred = pd.Series(pred).map(RELATION_GROUP).to_numpy()
        summary[name] = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=RELATIONS, average="macro", zero_division=0),
            "group_accuracy": accuracy_score(groups_true, groups_pred),
            "group_macro_f1": f1_score(groups_true, groups_pred, average="macro", zero_division=0),
            "report": classification_report(y, pred, labels=RELATIONS, output_dict=True, zero_division=0),
        }
        for video in videos:
            m = (data.video == video).to_numpy()
            per_video_rows.append({"model": name, "video": video, "n": int(m.sum()),
                                   "accuracy": accuracy_score(y[m], pred[m]),
                                   "macro_f1": f1_score(y[m], pred[m], labels=RELATIONS, average="macro", zero_division=0)})
        cm = confusion_matrix(y, pred, labels=RELATIONS)
        pd.DataFrame(cm, index=RELATIONS, columns=RELATIONS).to_csv(out / f"confusion_{name}{tag}.csv")
    pd.DataFrame(per_video_rows).to_csv(out / f"relation_per_video{tag}.csv", index=False)
    data.assign(**{f"pred_{k}": v for k, v in predictions.items()}).to_csv(out / f"relation_lovo_predictions{tag}.csv", index=False)
    plot_confusions(predictions, y, out / f"relation_confusion{tag}.png")
    return summary


def plot_confusions(predictions: dict, y: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.2 * len(predictions), 4.8))
    for ax, (name, pred) in zip(np.atleast_1d(axes), predictions.items()):
        cm = confusion_matrix(y, pred, labels=RELATIONS).astype(float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(RELATIONS)):
            for j in range(len(RELATIONS)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(RELATIONS)), RELATIONS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(RELATIONS)), RELATIONS, fontsize=8)
        ax.set_title(f"{name} (held-out videos)")
        ax.set_xlabel("predicted")
    np.atleast_1d(axes)[0].set_ylabel("labelled")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def lane_paint_at(analysis: Path, perception: Path, keys: list[tuple[str, int]]) -> np.ndarray:
    """The lane-paint detector's per-second bike-lane decision (1 / 0 / NaN unknown) at each (video, frame)."""
    out = np.full(len(keys), np.nan)
    cache: dict[str, tuple[pd.Series, float] | None] = {}
    for i, (video, frame) in enumerate(keys):
        if video not in cache:
            lanes_path = analysis / video / "lanes.parquet"
            if not lanes_path.exists():
                cache[video] = None
            else:
                meta = read_json(perception / video / "meta.json")
                objects_path = analysis / video / "infrastructure.parquet"
                objects = pd.read_parquet(objects_path) if objects_path.exists() else pd.DataFrame()
                n_seconds = int(meta["source_frames"] / meta["source_fps"]) + 1
                decision = per_second_summary(pd.read_parquet(lanes_path), objects, n_seconds).in_bike_lane
                cache[video] = (pd.to_numeric(decision, errors="coerce"), meta["source_fps"])
        entry = cache[video]
        if entry is not None:
            decision, fps = entry
            second = int(frame / fps)
            if second in decision.index:
                out[i] = decision.loc[second]
    return out


def evaluate_scenes(perception: Path, labels_csv: Path, out: Path, analysis: Path | None = None) -> tuple[dict, object | None]:
    labels = pd.read_csv(labels_csv)
    labels = labels[labels.label.isin(SCENES)]
    xs, ys, vids, keys = [], [], [], []
    for video, group in labels.groupby("video"):
        if not (perception / video / "clip.npz").exists():
            print(f"scene labels for {video} skipped: no CLIP embeddings in {perception / video}")
            continue
        frames, feats = scene_features(perception / video / "clip.npz")
        # nearest embedded frame: a perception run at another frame rate samples slightly different frames
        tolerance = 0.5 * read_json(perception / video / "meta.json")["source_fps"]
        for row in group.itertuples():
            i = int(np.abs(frames.astype(int) - int(row.frame)).argmin())
            if abs(int(frames[i]) - int(row.frame)) <= tolerance:
                xs.append(feats[i])
                ys.append(row.label)
                vids.append(video)
                keys.append((str(video), int(row.frame)))
    if not xs:
        return {}, None
    X, y, vids = np.stack(xs), np.array(ys), np.array(vids)
    half = X.shape[1] // 2
    zs = 0.5 * (zero_shot(X[:, :half], SCENES) + zero_shot(X[:, half:], SCENES))
    zs_pred = np.array(SCENES)[zs.argmax(axis=1)]
    # Pick the probe's regularisation by leave-one-video-out macro-F1 rather than fixing it blind.
    best_c, best_score, probe_pred = 0.05, -1.0, np.empty(len(y), dtype=object)
    probe_sep = np.full(len(y), np.nan)
    for c in (0.01, 0.03, 0.1, 0.3, 1.0):
        pred = np.empty(len(y), dtype=object)
        p_sep = np.full(len(y), np.nan)
        for video in np.unique(vids):
            test = vids == video
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
            probe.fit(X[~test], y[~test])
            pred[test] = probe.predict(X[test])
            classes = list(probe.classes_)
            if "separated_path" in classes:
                p_sep[test] = probe.predict_proba(X[test])[:, classes.index("separated_path")]
        score = f1_score(y, pred, labels=SCENES, average="macro", zero_division=0)
        if score > best_score:
            best_c, best_score, probe_pred, probe_sep = c, score, pred, p_sep
    predictions = {"zero_shot": zs_pred, "linear_probe": probe_pred}
    paint = lane_paint_at(analysis, perception, keys) if analysis is not None else np.full(len(y), np.nan)
    if np.isfinite(paint).any():
        # the paint deciding bike lane vs shared road where CLIP does not see a separated path: kept as the check
        # that showed it does not help (bikesafe.exposure reports CLIP's context and the paint call beside it)
        fused = probe_pred.copy()
        road = ~(probe_sep >= 0.5)
        fused[road & (paint == 1.0)] = "bike_lane"
        fused[road & (paint == 0.0)] = "shared_road"
        predictions["probe_plus_lane_paint"] = fused
    summary = {}
    for name, pred in predictions.items():
        summary[name] = {"accuracy": accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, labels=SCENES, average="macro", zero_division=0),
                         "report": classification_report(y, pred, labels=SCENES, output_dict=True, zero_division=0)}
        pd.DataFrame(confusion_matrix(y, pred, labels=SCENES), index=SCENES, columns=SCENES).to_csv(out / f"scene_confusion_{name}.csv")
    summary["linear_probe"]["C"] = best_c
    if "probe_plus_lane_paint" in summary:
        summary["probe_plus_lane_paint"]["frames_with_paint_evidence"] = round(float(np.isfinite(paint).mean()), 3)
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced")).fit(X, y)
    return summary, final


def train_sign_probe(labels_csv: Path, perception: Path, videos: Path, models: Path, out: Path) -> dict:
    """Fit the sign-type probe (bikesafe.signs.train_probe) on the signs typed in bikesafe.review, cropping each one
    from its video again, and write models/typology/sign_probe.pkl and results/typology/sign_evaluation.json."""
    import torch

    from bikesafe.signs import MIN_PROBE_LABELS, SignTyper, sign_crop, train_probe

    labels = pd.read_csv(labels_csv) if labels_csv.exists() else pd.DataFrame()
    if len(labels) < MIN_PROBE_LABELS or labels.video.nunique() < 2:  # nothing to validate a probe on: no GPU needed
        (models / "sign_probe.pkl").unlink(missing_ok=True)
        report = {"trained": False, "n_labels": int(len(labels)),
                  "reason": "fewer than 20 reviewed signs" if len(labels) < MIN_PROBE_LABELS
                  else "reviewed signs from fewer than two rides"}
        (out / "sign_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    crops, kept = [], []
    for video, rows in labels.groupby("video"):
        meta_path = perception / str(video) / "meta.json"
        if not meta_path.exists():
            continue
        path = resolve_video(read_json(meta_path), videos)
        if not path.exists():  # e.g. a Colab run imported without its video
            print(f"{video}: video not found ({path}); its {len(rows)} reviewed signs are left out of the probe")
            continue
        cap = open_video(path)
        for r in rows.sort_values("frame").itertuples():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, image = cap.read()
            if ok:
                crops.append(sign_crop(image, (r.x1, r.y1, r.x2, r.y2)))
                kept.append(r.Index)
        cap.release()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer = SignTyper(device, fp16=is_cuda(device))
    probe, report = train_probe(labels.loc[kept, "label"], crops, typer, labels.loc[kept, "video"])
    if probe is not None:
        (models / "sign_probe.pkl").write_bytes(pickle.dumps(probe))
    else:  # an earlier probe that no longer validates must not keep typing the signs
        (models / "sign_probe.pkl").unlink(missing_ok=True)
    (out / "sign_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--track-labels", type=Path, default=ROOT / "data" / "typology" / "track_labels.csv")
    parser.add_argument("--scene-labels", type=Path, default=ROOT / "data" / "typology" / "scene_labels.csv")
    parser.add_argument("--sign-labels", type=Path, default=ROOT / "data" / "typology" / "sign_labels.csv",
                        help="Signs typed in bikesafe.review; trains the sign-type probe")
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "typology")
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "typology")
    parser.add_argument("--near-z", type=float, default=NEAR_Z_M,
                        help="Vehicles whose closest approach is within this distance (m) are the deployment domain")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    sign_report = train_sign_probe(args.sign_labels, args.perception, args.videos, args.models, args.out)
    print(f"sign probe: {sign_report}")
    scene_summary, scene_probe = evaluate_scenes(args.perception, args.scene_labels, args.out, args.analysis)
    if scene_probe is not None:
        (args.models / "scene_probe.pkl").write_bytes(pickle.dumps(scene_probe))

    data = load_labelled_tracks(args.analysis, args.track_labels, args.perception)
    if data.label.nunique() < len(RELATIONS) or data.video.nunique() < 2:
        print(f"{len(data)} labelled tracks from {data.video.nunique()} processed videos: the relation model is not "
              f"retrained (it needs every relation and at least two videos); keeping {args.models / 'relation_model.pkl'}")
        return
    near = data[data.z_min <= args.near_z]
    ablation = feature_ablation(data, near, args.out)
    print("feature ablation (leave-one-video-out):")
    print(ablation.to_string(index=False))
    relation_summary = evaluate_relations(data, args.out)
    near_summary = evaluate_relations(near, args.out, tag="_near")
    # The near subset is where the geometry is trustworthy and where safety matters, so the model is selected there.
    best = max((k for k in near_summary if k != "rules"), key=lambda k: near_summary[k]["macro_f1"])
    features = feature_columns(data)
    cols = model_features(best, features)
    final = make_models()[best].fit(as_float(data, cols), data.label)
    (args.models / "relation_model.pkl").write_bytes(pickle.dumps({"name": best, "features": cols, "model": final}))
    tree = make_models()["decision_tree"]
    tree_cols = model_features("decision_tree", features)
    tree.fit(as_float(data, tree_cols), data.label)
    (args.out / "decision_tree_rules.txt").write_text(export_text(tree, feature_names=tree_cols, decimals=2), encoding="utf-8")

    def strip(summary: dict) -> dict:
        return {k: {m: v for m, v in s.items() if m != "report"} | {"per_class_f1": {c: round(s["report"][c]["f1-score"], 3)
                for c in s["report"] if c in RELATIONS + SCENES}} for k, s in summary.items()}

    accuracy_by_distance = (data.assign(correct=data.label == pd.read_csv(args.out / "relation_lovo_predictions.csv")[f"pred_{best}"],
                                        bin=pd.cut(data.z_min, DISTANCE_BINS))
                            .groupby("bin", observed=True).correct.agg(["size", "mean"]).round(3))
    accuracy_by_distance.to_csv(args.out / "relation_accuracy_by_distance.csv")
    result = {"n_track_labels": int(len(data)), "label_counts": data.label.value_counts().to_dict(),
              "videos": sorted(data.video.unique()), "relation": strip(relation_summary),
              "near_z_m": args.near_z, "n_near_labels": int(len(near)),
              "relation_within_near_z": strip(near_summary), "selected_model": best,
              "accuracy_by_closest_approach": {str(k): {"n": int(v["size"]), "accuracy": float(v["mean"])}
                                               for k, v in accuracy_by_distance.iterrows()},
              "feature_ablation": ablation.to_dict(orient="records"),
              "scene": strip(scene_summary)}
    (args.out / "evaluation_summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
