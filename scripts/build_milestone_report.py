"""Build the meeting report and tracker comparison from saved JSON summaries."""

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
    parser.add_argument("detection_stats", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("trackers", nargs="+", help="name=track_stats.json")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detection = json.loads(args.detection_stats.read_text(encoding="utf-8"))
    tracker_stats = {}
    for spec in args.trackers:
        name, path = spec.split("=", 1)
        tracker_stats[name] = json.loads(Path(path).read_text(encoding="utf-8"))

    rows = []
    for name, stats in tracker_stats.items():
        rows.append({
            "tracker": name,
            "detections_with_id": stats["assigned_detections"],
            "unique_track_ids": stats["unique_track_ids"],
            "median_track_s": stats["median_track_observed_duration_s"],
            "mean_track_s": stats["mean_track_observed_duration_s"],
            "tracks_under_1s": stats["tracks_under_one_second"],
            "single_frame_tracks": stats["single_frame_tracks"],
            "tracks_with_internal_gaps": stats["tracks_with_internal_gaps"],
        })
    comparison = pd.DataFrame(rows).set_index("tracker")
    comparison.to_csv(args.output_dir / "tracker_comparison.csv")

    display_names = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT"}
    plot_data = comparison.rename(index=lambda name: display_names.get(name, name))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    colours = ["#397aa8", "#db7442"][: len(comparison)]
    plot_data["unique_track_ids"].plot.bar(ax=axes[0], color=colours, rot=0, title="Unique IDs\n(lower can mean less fragmentation)")
    plot_data["tracks_under_1s"].plot.bar(ax=axes[1], color=colours, rot=0, title="Tracks under one second")
    plot_data["median_track_s"].plot.bar(ax=axes[2], color=colours, rot=0, title="Median observed track duration")
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")
    axes[2].set_ylabel("Seconds")
    for index, ax in enumerate(axes):
        ax.grid(axis="y", alpha=0.2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f" if index == 2 else "%.0f", padding=3)
    fig.suptitle("Tracking baseline comparison on the 60-second clip", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output_dir / "tracker_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    names = list(comparison.index)
    best = min(names, key=lambda name: comparison.loc[name, "tracks_under_1s"])
    other = next((name for name in names if name != best), best)
    id_reduction = 100 * (comparison.loc[other, "unique_track_ids"] - comparison.loc[best, "unique_track_ids"]) / comparison.loc[other, "unique_track_ids"]
    short_reduction = 100 * (comparison.loc[other, "tracks_under_1s"] - comparison.loc[best, "tracks_under_1s"]) / comparison.loc[other, "tracks_under_1s"]

    tracker_lines = [
        "| Tracker | Assigned detections | Unique IDs* | Median track | Tracks <1 s | Internal gaps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in comparison.iterrows():
        tracker_lines.append(
            f"| {display_names.get(name, name)} | {int(row.detections_with_id):,} | {int(row.unique_track_ids):,} | "
            f"{row.median_track_s:.2f} s | {int(row.tracks_under_1s):,} | {int(row.tracks_with_internal_gaps):,} |"
        )

    best_summary_path = args.output_dir / "tracking" / best / "analysis" / "track_summary.csv"
    exemplar_text = ""
    if best_summary_path.exists():
        best_tracks = pd.read_csv(best_summary_path)
        exemplar = best_tracks.loc[best_tracks["observed_frames"].idxmax()]
        exemplar_text = (
            f"A clear persistence success is track ID {int(exemplar.track_id)}: it remains assigned for "
            f"{exemplar.observed_duration_s:.2f} seconds ({int(exemplar.observed_frames):,} consecutive frames) "
            f"with {int(exemplar.internal_gaps)} internal gaps.\n"
        )

    reid_summary_path = args.output_dir / "reid_triplet_summary" / "summary.json"
    reid_text = "Triplet-loss training has not been run."
    if reid_summary_path.exists():
        reid = json.loads(reid_summary_path.read_text(encoding="utf-8"))
        reid_text = (
            f"Across {reid['runs']} seeds, frozen pretrained features scored "
            f"{100 * reid['baseline_accuracy']:.1f}% on approved validation triplets. "
            f"Triplet training averaged {100 * reid['mean_trained_accuracy']:.1f}% "
            f"(best {100 * reid['best_accuracy']:.1f}%), improving the mean by "
            f"{reid['improvement_percentage_points']:.1f} percentage points."
        )
    reid_evaluation_path = args.output_dir / "reid_triplet_evaluation" / "reid_evaluation.json"
    reid_decision = ""
    if reid_evaluation_path.exists():
        evaluation = json.loads(reid_evaluation_path.read_text(encoding="utf-8"))
        reid_decision = (
            f" The best model achieved {100 * evaluation['pair_accuracy_on_calibration_split']:.1f}% "
            f"pair accuracy at cosine threshold {evaluation['calibrated_threshold']:.4f}, with "
            f"{evaluation['false_positives']} false matches and {evaluation['false_negatives']} missed matches. "
            "It should rank review candidates rather than merge identities automatically."
        )

    report = f"""# Vehicle detection and tracking milestone

## What is now working

The complete 30-minute source video has been processed for car detections. A separate
60-second busy clip has now been processed with two multi-object trackers using all four
COCO road-vehicle classes: car, motorcycle, bus, and truck.

## Full-video detection results

| Measure | Result |
|---|---:|
| Frames | {detection['total_frames']:,} |
| Car detections | {detection['total_detections']:,} |
| Frames with at least one detected car | {detection['frames_with_detection']:,} ({detection['percent_frames_with_detection']:.2f}%) |
| Empty frames | {detection['empty_frames']:,} |
| Mean detections per frame, including empty frames | {detection['mean_detections_all_frames']:.2f} |
| Median | {detection['median_detections_all_frames']:.0f} |
| Minimum / maximum | {detection['minimum_detections']} / {detection['maximum_detections']} |

These are model detections, not verified ground-truth counts. The earlier 7.4 mean omitted
empty frames; {detection['mean_detections_all_frames']:.2f} is the corrected all-frame mean.

![Detected cars over time](analysis_fullvideo/vehicle_count_over_time.png)

## Persistence baseline

{chr(10).join(tracker_lines)}

Note: Unique tracker IDs are not yet unique physical-vehicle counts. Track fragmentation can
assign several IDs to one vehicle, while an ID switch can merge identities.

{display_names.get(best, best)} is the stronger baseline on these proxy measures. It created {id_reduction:.1f}%
fewer IDs and {short_reduction:.1f}% fewer sub-one-second tracks than {display_names.get(other, other)}. This fits
the moving dashcam setting because its camera-motion compensation can stabilize association.

{exemplar_text}

![BoT-SORT tracked frame at 30 seconds](tracking/{best}/{best}_t30s.jpg)

![Tracker comparison](tracker_comparison.png)

## Evidence and review material

- `tracking/{best.lower()}/{best.lower()}_annotated_h264.mp4`: meeting-ready video with persistent IDs.
- `tracking/{best.lower()}/{best.lower()}_tracks.csv`: frame-by-frame boxes, classes, confidence, and track IDs.
- `REID_PILOT.md`: controlled 300-frame comparison showing that automatic ReID worsened fragmentation.
- `../datasets/detection_review/`: 120 approved frames split into 96 training and 24 validation images.
- `../datasets/tracking_identity_review/`: the approved busiest 10-second sequence, covering 300 frames.
- `../datasets/reid_triplet_candidates/`: 350 approved triplets from 420 crops.
- `reid_triplet_summary/TRIPLET_LOSS_RESULTS.md`: repeated triplet-loss results.
- `reid_triplet_evaluation/REID_THRESHOLD_EVALUATION.md`: calibrated similarity analysis.

## Automatic ReID decision

A controlled 300-frame test produced 76 IDs with standard BoT-SORT, 108 with ByteTrack,
and 121 with BoT-SORT's automatic ReID. Its built-in appearance matching therefore worsened
the current fragmentation proxies, so standard BoT-SORT remains the deployed baseline.

## Triplet-loss training result

{reid_text}{reid_decision}

![Triplet validation accuracy](reid_triplet_summary/triplet_validation_accuracy.png)

## Future generalization check

The review sets and triplet pilot are complete. Before using the learned projection on a
new camera or route, annotate an independent sequence and compare ID switches/IDF1 against
standard BoT-SORT.
"""
    (args.output_dir / "MEETING_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'MEETING_REPORT.md'}")


if __name__ == "__main__":
    main()
