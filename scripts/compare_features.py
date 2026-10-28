"""Compare relation-model feature sets on the near labels, robustly: extra trees with several seeds, leave-one-video-
out, reporting mean and spread of accuracy and macro-F1, the per-class F1, moving vehicles called parked, and
accuracy on the vehicles seen mostly while the rider was stopped.

    python scripts/compare_features.py [--seeds 5] [--sets v5 v7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikesafe import train  # noqa: E402
from bikesafe.common import RELATIONS, ROOT  # noqa: E402

SETS = train.feature_sets()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "typology" / "track_labels.csv")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sets", nargs="+", default=["v5", "v7"])
    parser.add_argument("--out", type=Path, default=None, help="CSV of the per-seed scores")
    args = parser.parse_args()
    data = train.load_labelled_tracks(args.analysis, args.labels, args.perception)
    data = data[data.z_min <= train.NEAR_Z_M].reset_index(drop=True)
    y, videos = data.label.to_numpy(), data.video.to_numpy()
    stopped = (data.ego_moving_frac < 0.5).to_numpy()
    available = set(train.feature_columns(data))
    rows = []
    for name in args.sets:
        cols = [c for c in SETS[name] if c in available]
        X = train.as_float(data, cols)
        for seed in range(args.seeds):
            pred = np.empty(len(data), dtype=object)
            for v in np.unique(videos):
                test = videos == v
                model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                      ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                                           class_weight="balanced_subsample", random_state=seed,
                                                           n_jobs=-1))
                model.fit(X[~test], y[~test])
                pred[test] = model.predict(X[test])
            f1 = f1_score(y, pred, labels=RELATIONS, average=None, zero_division=0)
            rows.append({"set": name, "n_features": len(cols), "seed": seed, "accuracy": accuracy_score(y, pred),
                         "macro_f1": float(f1.mean()), **{f"f1_{r}": f for r, f in zip(RELATIONS, f1)},
                         "moving_called_parked": int(((pred == "parked") & (y != "parked")).sum()),
                         "stopped_accuracy": accuracy_score(y[stopped], pred[stopped]) if stopped.any() else np.nan})
    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
    print(f"{len(data)} near labels, {int(stopped.sum())} seen mostly while the rider was stopped")
    summary = table.groupby("set").agg(["mean", "std"]).drop(columns=["seed", "n_features"], level=0)
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
