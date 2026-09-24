"""Create zero-aware per-frame detection statistics and presentation charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detections_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--rolling-seconds", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections = pd.read_csv(args.detections_csv, usecols=["frame", "class_name", "confidence"])
    frames = pd.Index(range(1, args.total_frames + 1), name="frame")
    counts = detections.groupby("frame").size().reindex(frames, fill_value=0).astype(int)
    window = max(1, round(args.rolling_seconds * args.fps))

    per_frame = pd.DataFrame({
        "frame": frames,
        "time_s": (frames.to_numpy() - 1) / args.fps,
        "detected_vehicles": counts.to_numpy(),
    })
    per_frame["contains_vehicle"] = per_frame["detected_vehicles"] > 0
    per_frame[f"rolling_mean_{args.rolling_seconds:g}s"] = (
        per_frame["detected_vehicles"].rolling(window, center=True, min_periods=1).mean()
    )
    per_frame.to_csv(args.output_dir / "per_frame_counts.csv", index=False)

    per_frame["minute"] = np.floor(per_frame["time_s"] / 60).astype(int)
    by_minute = per_frame.groupby("minute").agg(
        mean_detected_vehicles=("detected_vehicles", "mean"),
        median_detected_vehicles=("detected_vehicles", "median"),
        max_detected_vehicles=("detected_vehicles", "max"),
        frames_with_vehicle=("contains_vehicle", "sum"),
        frames=("frame", "count"),
    )
    by_minute["percent_frames_with_vehicle"] = 100 * by_minute["frames_with_vehicle"] / by_minute["frames"]
    by_minute.to_csv(args.output_dir / "counts_by_minute.csv")

    values = per_frame["detected_vehicles"]
    stats = {
        "source": str(args.detections_csv),
        "classes_present": sorted(detections["class_name"].dropna().unique().tolist()),
        "total_frames": int(args.total_frames),
        "total_detections": int(len(detections)),
        "frames_with_detection": int((values > 0).sum()),
        "empty_frames": int((values == 0).sum()),
        "percent_frames_with_detection": round(float(100 * (values > 0).mean()), 3),
        "mean_detections_all_frames": round(float(values.mean()), 4),
        "median_detections_all_frames": float(values.median()),
        "minimum_detections": int(values.min()),
        "maximum_detections": int(values.max()),
        "mean_confidence": round(float(detections["confidence"].mean()), 4),
        "fps": args.fps,
        "duration_minutes": round(args.total_frames / args.fps / 60, 3),
    }
    (args.output_dir / "detection_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    minutes = per_frame["time_s"] / 60
    ax.plot(minutes, values, color="#9ab8d0", alpha=0.35, linewidth=0.45, label="Per-frame detections")
    ax.plot(
        minutes,
        per_frame[f"rolling_mean_{args.rolling_seconds:g}s"],
        color="#125a8a",
        linewidth=1.5,
        label=f"{args.rolling_seconds:g}-second rolling mean",
    )
    ax.set(title="Detected cars throughout the 30-minute video", xlabel="Video time (minutes)", ylabel="Detected cars")
    ax.set_xlim(0, minutes.max())
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "vehicle_count_over_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(values.max() + 2) - 0.5
    ax.hist(values, bins=bins, color="#2475a9", edgecolor="white", linewidth=0.4)
    ax.axvline(values.mean(), color="#c9473d", linestyle="--", label=f"Mean = {values.mean():.2f}")
    ax.axvline(values.median(), color="#2b7a3d", linestyle=":", label=f"Median = {values.median():.0f}")
    ax.set(title="Distribution of detected cars per frame", xlabel="Detected cars", ylabel="Number of frames")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "vehicle_count_distribution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
