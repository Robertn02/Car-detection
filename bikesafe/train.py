"""Train and evaluate the rider-relation classifier and the scene probe with leave-one-video-out validation.

Compares (1) the hand-written rules, (2) a depth-limited decision tree on physical features (readable rules
learned from labels) and (3) gradient boosting on all track features. The best model by macro-F1 across held-out
videos is refit on all labels and saved for bikesafe.apply.

    python -m bikesafe.train
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from bikesafe.common import RELATION_GROUP, RELATIONS, ROOT, SCENES
from bikesafe.scene import scene_features, zero_shot
from bikesafe.tracks import rule_relation

PHYSICAL_FEATURES = [
    "x_center_med", "x_center_p10", "x_center_p90", "x_inner_med", "z_min", "z_med", "v_along_med", "v_along_p10",
    "v_along_p90", "v_lat_med", "v_lat_absmed", "frac_static", "frac_cross", "frac_same", "frac_opposite",
    "ego_speed_med", "ego_moving_frac", "x_at_min_z", "frac_close_pass", "v_along_ratio", "v_lat_ratio",
    "p_separated_path", "p_bike_lane", "p_shared_road", "view_rear", "view_front", "view_side",
    # bird's-eye geometry from learned depth, and the lane structure between rider and vehicle
    "world_speed", "world_heading_abs", "world_speed_ratio", "abs_x_at_min_z", "static_neighbours",
    "depth_z_min", "depth_x_at_min_z", "world_speed_fit", "heading_fit_abs", "frac_world_static",
    "frac_heading_same", "frac_heading_opposite", "frac_heading_crossing", "lines_between_med", "yellow_between_max",
]
NON_FEATURES = {"video", "track_id", "t_start", "t_end", "scene", "ride_scene", "rule_relation", "label", "labeler", "notes", "card_id"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def as_float(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).to_numpy()


def load_labelled_tracks(analysis: Path, labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    labels = labels[labels.label.isin(RELATIONS)]
    tracks = pd.concat([pd.read_parquet(p) for p in sorted(analysis.glob("*/tracks.parquet"))], ignore_index=True)
    merged = labels.merge(tracks, on=["video", "track_id"], how="inner")
    for flag in ("enter_edge", "exit_edge"):
        merged[flag] = merged[flag].astype(float)
    # Recompute the rule baseline so it always reflects the current rules, not whatever was stored earlier.
    merged["rule_relation"] = merged.apply(rule_relation, axis=1)
    return merged


def make_models() -> dict:
    return {
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=0),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0,
            class_weight="balanced", random_state=0),
    }


def model_features(name: str, all_features: list[str]) -> list[str]:
    """Both models use the physics-defined list. Throwing all ~80 columns at gradient boosting scored the same or
    worse in leave-one-video-out ablation, which is what 378 labels should be expected to support."""
    del name
    return [f for f in PHYSICAL_FEATURES if f in all_features]


def evaluate_relations(data: pd.DataFrame, out: Path, tag: str = "") -> dict:
    features = feature_columns(data)
    videos = sorted(data.video.unique())
    predictions = {"rules": data.rule_relation.to_numpy()}
    for name, model in make_models().items():
        cols = model_features(name, features)
        pred = np.empty(len(data), dtype=object)
        for video in videos:
            test = (data.video == video).to_numpy()
            if (~test).sum() == 0:
                continue
            model.fit(as_float(data[~test], cols), data.label[~test])
            pred[test] = model.predict(as_float(data[test], cols))
        predictions[name] = pred

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


def evaluate_scenes(perception: Path, labels_csv: Path, out: Path) -> tuple[dict, object | None]:
    labels = pd.read_csv(labels_csv)
    labels = labels[labels.label.isin(SCENES)]
    xs, ys, vids = [], [], []
    for video, group in labels.groupby("video"):
        frames, feats = scene_features(perception / video / "clip.npz")
        index = {int(f): i for i, f in enumerate(frames)}
        for row in group.itertuples():
            if int(row.frame) in index:
                xs.append(feats[index[int(row.frame)]])
                ys.append(row.label)
                vids.append(video)
    if not xs:
        return {}, None
    X, y, vids = np.stack(xs), np.array(ys), np.array(vids)
    half = X.shape[1] // 2
    zs = 0.5 * (zero_shot(X[:, :half], SCENES) + zero_shot(X[:, half:], SCENES))
    zs_pred = np.array(SCENES)[zs.argmax(axis=1)]
    # Pick the probe's regularisation by leave-one-video-out macro-F1 rather than fixing it blind.
    best_c, best_score, probe_pred = 0.05, -1.0, np.empty(len(y), dtype=object)
    for c in (0.01, 0.03, 0.1, 0.3, 1.0):
        pred = np.empty(len(y), dtype=object)
        for video in np.unique(vids):
            test = vids == video
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
            probe.fit(X[~test], y[~test])
            pred[test] = probe.predict(X[test])
        score = f1_score(y, pred, labels=SCENES, average="macro", zero_division=0)
        if score > best_score:
            best_c, best_score, probe_pred = c, score, pred
    summary = {}
    for name, pred in {"zero_shot": zs_pred, "linear_probe": probe_pred}.items():
        summary[name] = {"accuracy": accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, labels=SCENES, average="macro", zero_division=0),
                         "report": classification_report(y, pred, labels=SCENES, output_dict=True, zero_division=0)}
        pd.DataFrame(confusion_matrix(y, pred, labels=SCENES), index=SCENES, columns=SCENES).to_csv(out / f"scene_confusion_{name}.csv")
    summary["linear_probe"]["C"] = best_c
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced")).fit(X, y)
    return summary, final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--track-labels", type=Path, default=ROOT / "data" / "typology" / "track_labels.csv")
    parser.add_argument("--scene-labels", type=Path, default=ROOT / "data" / "typology" / "scene_labels.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "typology")
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "typology")
    parser.add_argument("--near-z", type=float, default=15.0,
                        help="Vehicles whose closest approach is within this distance (m) are the deployment domain")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    scene_summary, scene_probe = evaluate_scenes(args.perception, args.scene_labels, args.out)
    if scene_probe is not None:
        (args.models / "scene_probe.pkl").write_bytes(pickle.dumps(scene_probe))

    data = load_labelled_tracks(args.analysis, args.track_labels)
    near = data[data.z_min <= args.near_z]
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
                                        bin=pd.cut(data.z_min, [0, 5, 10, 15, 25, 1000]))
                            .groupby("bin", observed=True).correct.agg(["size", "mean"]).round(3))
    accuracy_by_distance.to_csv(args.out / "relation_accuracy_by_distance.csv")
    result = {"n_track_labels": int(len(data)), "label_counts": data.label.value_counts().to_dict(),
              "videos": sorted(data.video.unique()), "relation": strip(relation_summary),
              "near_z_m": args.near_z, "n_near_labels": int(len(near)),
              "relation_within_near_z": strip(near_summary), "selected_model": best,
              "accuracy_by_closest_approach": {str(k): {"n": int(v["size"]), "accuracy": float(v["mean"])}
                                               for k, v in accuracy_by_distance.iterrows()},
              "scene": strip(scene_summary)}
    (args.out / "evaluation_summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
