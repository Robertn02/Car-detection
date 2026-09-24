"""Aggregate repeated triplet-loss runs into a chart and concise report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    frame = pd.DataFrame(records).sort_values("seed")
    frame.to_csv(args.output_dir / "reid_runs.csv", index=False)
    baseline = float(frame["baseline_validation_accuracy"].mean())
    trained_mean = float(frame["trained_validation_accuracy"].mean())
    trained_std = float(frame["trained_validation_accuracy"].std(ddof=1)) if len(frame) > 1 else 0.0
    best = frame.loc[frame["trained_validation_accuracy"].idxmax()]
    summary = {
        "runs": len(frame),
        "validation_triplets": int(frame.iloc[0]["validation_triplets"]),
        "baseline_accuracy": round(baseline, 4),
        "mean_trained_accuracy": round(trained_mean, 4),
        "trained_accuracy_std": round(trained_std, 4),
        "improvement_percentage_points": round(100 * (trained_mean - baseline), 2),
        "best_accuracy": round(float(best["trained_validation_accuracy"]), 4),
        "best_seed": int(best["seed"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = ["Frozen\nfeatures"] + [f"Triplet\nseed {int(seed)}" for seed in frame["seed"]]
    values = [baseline] + frame["trained_validation_accuracy"].tolist()
    colours = ["#8b9aa5"] + ["#2878a8"] * len(frame)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colours)
    ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3)
    ax.axhline(trained_mean, color="#d05b3e", linestyle="--", linewidth=1.3, label=f"Triplet mean {100 * trained_mean:.1f}%")
    ax.set(title="Reviewed triplet validation accuracy", ylabel="Correct same/different ordering", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "triplet_validation_accuracy.png", dpi=180)
    plt.close(fig)

    runs = "\n".join(
        f"| {int(row.seed)} | {100 * row.trained_validation_accuracy:.1f}% | {row.best_epoch:.0f} | {row.epochs_completed:.0f} |"
        for _, row in frame.iterrows()
    )
    report = f"""# Triplet-loss ReID pilot results

The experiment used 285 approved training triplets and {summary['validation_triplets']}
approved validation triplets from 420 vehicle crops. ImageNet-pretrained ResNet-18 features
were frozen; triplet loss trained a 128-dimensional projection. Early stopping selected the
checkpoint with the lowest validation loss.

| Seed | Validation accuracy | Best epoch | Epochs completed |
|---:|---:|---:|---:|
{runs}

The frozen-feature baseline scored {100 * baseline:.1f}%. Triplet training averaged
{100 * trained_mean:.1f}% (sample standard deviation {100 * trained_std:.1f} percentage
points), an average improvement of {summary['improvement_percentage_points']:.1f}
percentage points. The best run was seed {summary['best_seed']} at
{100 * summary['best_accuracy']:.1f}%.

This is a pilot on one reviewed clip and one fixed validation split. It supports the value
of learned appearance similarity, but deployment should wait for an independent annotated
sequence and a direct comparison of ID switches against standard BoT-SORT.
"""
    (args.output_dir / "TRIPLET_LOSS_RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
