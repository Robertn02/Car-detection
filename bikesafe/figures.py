"""Report figures: classifier reliability against distance, and exposure by riding context.

    python -m bikesafe.figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bikesafe.common import RELATIONS, ROOT, SCENES

RELATION_LABELS = {"ego_lane": "in my lane", "adjacent_same": "other lane,\nsame way", "oncoming": "oncoming",
                   "parked": "parked", "cross_side": "side road /\nother roadway"}
SCENE_LABELS = {"separated_path": "separated path", "bike_lane": "bike lane", "shared_road": "shared road"}


def accuracy_figure(results: Path, out: Path) -> None:
    table = pd.read_csv(results / "relation_accuracy_by_distance.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    positions = np.arange(len(table))
    ax.bar(positions, 100 * table["mean"], color="#2678a5")
    for i, (n, acc) in enumerate(zip(table["size"], table["mean"])):
        ax.text(i, 100 * acc + 1.5, f"{100 * acc:.0f}%\nn={n}", ha="center", fontsize=9)
    ax.set(xticks=positions, xticklabels=[str(i).replace("(", "").replace("]", "").replace(", ", "-") + " m" for i in table.index],
           ylabel="Leave-one-video-out accuracy (%)", ylim=(0, 100),
           title="Relation accuracy by how close the vehicle came to the rider")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "relation_accuracy_by_distance.png", dpi=170)
    plt.close(fig)


def exposure_figure(corpus: Path, out: Path) -> None:
    scene = pd.read_csv(corpus / "exposure_by_scene.csv")
    weighted = scene.copy()
    rows = []
    for name, group in weighted.groupby("scene"):
        minutes = group.minutes.sum()
        row = {"scene": name, "minutes": minutes}
        for rel in RELATIONS:
            row[rel] = float((group[f"per_min_{rel}"] * group.minutes).sum() / minutes)
        row["door_zone_share"] = float((group.door_zone_share * group.minutes).sum() / minutes)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("scene").reindex(SCENES)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})
    width = 0.26
    positions = np.arange(len(RELATIONS))
    colours = ["#2678a5", "#4b9a62", "#d37843"]
    for i, scene_name in enumerate(SCENES):
        if scene_name not in table.index:
            continue
        values = [table.loc[scene_name, rel] for rel in RELATIONS]
        axes[0].bar(positions + (i - 1) * width, values, width,
                    label=f"{SCENE_LABELS[scene_name]} ({table.loc[scene_name, 'minutes']:.0f} min)", color=colours[i])
    axes[0].set(xticks=positions, xticklabels=[RELATION_LABELS[r] for r in RELATIONS], yscale="log",
                ylabel="vehicles per minute (log scale)", title="What the rider meets, by riding context")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([SCENE_LABELS[s] for s in table.index], 100 * table.door_zone_share, color="#8a5fb0")
    axes[1].set(ylabel="% of seconds", title="Riding within 1.5 m of a parked car")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "exposure_by_scene.png", dpi=170)
    plt.close(fig)
    table.round(2).to_csv(out / "exposure_by_scene_overall.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "typology")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results" / "corpus")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    accuracy_figure(args.results, args.results)
    exposure_figure(args.corpus, args.results)
    print(f"wrote figures to {args.results}")


if __name__ == "__main__":
    main()
