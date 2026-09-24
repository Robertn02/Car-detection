"""Evaluate tracker CSVs against an approved MOT-style reference sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.x.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda value: np.asarray(value, dtype=float)  # type: ignore[attr-defined]

import motmetrics as mm


def xywh(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        frame["x1"].to_numpy(), frame["y1"].to_numpy(),
        (frame["x2"] - frame["x1"]).to_numpy(), (frame["y2"] - frame["y1"]).to_numpy(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("predictions", nargs="+", help="name=tracks.csv")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(args.reference_csv)
    reference = reference[reference["track_id"] >= 0].copy()
    reference["frame"] = reference["frame"] - reference["frame"].min() + 1
    frame_count = int(reference["frame"].max())
    accumulators, names = [], []
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        prediction = pd.read_csv(path)
        prediction = prediction[prediction["track_id"] >= 0].copy()
        accumulator = mm.MOTAccumulator(auto_id=False)
        for frame_no in range(1, frame_count + 1):
            gt = reference[reference["frame"] == frame_no]
            pred = prediction[prediction["frame"] == frame_no]
            distances = mm.distances.iou_matrix(xywh(gt), xywh(pred), max_iou=args.iou)
            if distances.size:
                class_mismatch = gt["class_id"].to_numpy()[:, None] != pred["class_id"].to_numpy()[None, :]
                distances[class_mismatch] = np.nan
            accumulator.update(gt["track_id"].astype(int).tolist(), pred["track_id"].astype(int).tolist(), distances, frameid=frame_no)
        accumulators.append(accumulator)
        names.append(name)

    metrics = [
        "num_frames", "num_objects", "mota", "motp", "idf1", "idp", "idr",
        "precision", "recall", "num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost",
    ]
    summary = mm.metrics.create().compute_many(accumulators, names=names, metrics=metrics, generate_overall=False)
    summary["mean_matched_iou"] = 1 - summary["motp"]
    summary = summary.sort_values(["idf1", "mota"], ascending=False)
    summary.to_csv(args.output_dir / "tracker_mot_metrics.csv")
    rendered = mm.io.render_summary(summary, formatters=mm.metrics.create().formatters)
    (args.output_dir / "tracker_mot_metrics.txt").write_text(rendered, encoding="utf-8")

    labels = summary.index
    if len(labels) > 6:
        fig, axes = plt.subplots(1, 3, figsize=(15, max(5.2, 0.52 * len(labels))), sharey=True)
        positions = np.arange(len(labels))
        axes[0].barh(positions, 100 * summary["idf1"], color="#2678a5")
        axes[0].set(title="IDF1", xlabel="Percent", yticks=positions, yticklabels=labels)
        axes[1].barh(positions, 100 * summary["mota"], color="#4b9a62")
        axes[1].set(title="MOTA", xlabel="Percent")
        axes[2].barh(positions, summary["num_switches"], color="#d37843")
        axes[2].set(title="Identity switches", xlabel="Count")
        axes[0].invert_yaxis()
        for ax in axes:
            ax.grid(axis="x", alpha=0.2)
    else:
        figure_width = max(12, 1.65 * len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(figure_width, 5.2))
        axes[0].bar(labels, 100 * summary["idf1"], color="#2678a5")
        axes[0].set(title="IDF1", ylabel="Percent")
        axes[1].bar(labels, 100 * summary["mota"], color="#4b9a62")
        axes[1].set(title="MOTA", ylabel="Percent")
        axes[2].bar(labels, summary["num_switches"], color="#d37843")
        axes[2].set(title="Identity switches", ylabel="Count")
        for ax in axes:
            ax.tick_params(axis="x", rotation=32)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Tracker comparison on the approved 300-frame sequence", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output_dir / "tracker_mot_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(rendered)
    print(f"Best tracker by IDF1: {summary.index[0]}")


if __name__ == "__main__":
    main()
