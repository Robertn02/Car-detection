"""Does the relation model still improve with more labels? Leave-one-video-out scores for the selected model trained
on growing random fractions of the labels of the other rides (the held-out ride is always scored in full).

    python scripts/learning_curve.py [--repeats 5]

Writes results/typology/learning_curve.csv and .png. A curve still rising at 100% means more reviewed labels
(bikesafe.review) are the most direct way to raise accuracy; a flat one means the features are the limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikesafe.common import RELATIONS, ROOT  # noqa: E402
from bikesafe.train import as_float, feature_columns, load_labelled_tracks, make_models, model_features  # noqa: E402

FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "typology" / "track_labels.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "typology")
    parser.add_argument("--model", default="extra_trees")
    parser.add_argument("--near-z", type=float, default=15.0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    data = load_labelled_tracks(args.analysis, args.labels, args.perception)
    data = data[data.z_min <= args.near_z].reset_index(drop=True)
    cols = model_features(args.model, feature_columns(data))
    X, y, videos = as_float(data, cols), data.label.to_numpy(), data.video.to_numpy()
    rows = []
    for fraction in FRACTIONS:
        for seed in range(args.repeats if fraction < 1 else 1):
            rng = np.random.default_rng(seed)
            pred = np.empty(len(data), dtype=object)
            n_train = []
            for video in np.unique(videos):
                test = videos == video
                train = np.flatnonzero(~test)
                train = rng.choice(train, int(round(fraction * len(train))), replace=False)
                model = make_models()[args.model]
                model.fit(X[train], y[train])
                pred[test] = model.predict(X[test])
                n_train.append(len(train))
            rows.append({"fraction": fraction, "seed": seed, "train_labels": int(np.mean(n_train)),
                         "accuracy": accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, labels=RELATIONS, average="macro", zero_division=0)})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "learning_curve.csv", index=False)
    summary = table.groupby("fraction").agg(train_labels=("train_labels", "mean"), accuracy=("accuracy", "mean"),
                                            macro_f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"))
    print(summary.round(3).to_string())
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(summary.train_labels, summary.macro_f1, yerr=summary.f1_sd.fillna(0), marker="o", label="macro-F1")
    ax.plot(summary.train_labels, summary.accuracy, marker="s", label="accuracy")
    ax.set_xlabel("training labels (per held-out ride)")
    ax.set_ylabel(f"held-out score within {args.near_z:.0f} m")
    ax.set_title("Relation model vs number of labels")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "learning_curve.png", dpi=150)


if __name__ == "__main__":
    main()
