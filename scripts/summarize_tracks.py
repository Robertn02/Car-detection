"""Summarize tracker output without pretending proxy measures are ground truth."""

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
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fps", type=float, default=29.97)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.tracks_csv)
    tracked = raw[raw["track_id"] >= 0].copy()
    if tracked.empty:
        raise SystemExit("No assigned track IDs found")

    def mode_name(series: pd.Series) -> str:
        modes = series.mode()
        return str(modes.iloc[0]) if not modes.empty else "unknown"

    rows = []
    for track_id, group in tracked.groupby("track_id"):
        frames = np.sort(group["frame"].unique())
        gaps = np.diff(frames)
        span = int(frames[-1] - frames[0] + 1)
        observed = int(len(frames))
        rows.append({
            "track_id": int(track_id), "class_name": mode_name(group["class_name"]),
            "start_frame": int(frames[0]), "end_frame": int(frames[-1]),
            "start_s": (frames[0] - 1) / args.fps, "end_s": (frames[-1] - 1) / args.fps,
            "observed_frames": observed, "span_frames": span,
            "observed_duration_s": observed / args.fps, "span_duration_s": span / args.fps,
            "coverage": observed / span, "internal_gaps": int((gaps > 1).sum()),
            "maximum_gap_frames": int(gaps.max() - 1) if len(gaps) and gaps.max() > 1 else 0,
            "mean_confidence": float(group["confidence"].mean()),
        })
    tracks = pd.DataFrame(rows).sort_values(["start_frame", "track_id"])
    tracks.to_csv(args.output_dir / "track_summary.csv", index=False)

    total_frames = int(raw["frame"].max())
    per_frame = tracked.groupby("frame")["track_id"].nunique().reindex(range(1, total_frames + 1), fill_value=0)
    cumulative = np.zeros(total_frames, dtype=int)
    seen: set[int] = set()
    for frame, group in tracked.groupby("frame"):
        seen.update(int(value) for value in group["track_id"].unique())
        cumulative[int(frame) - 1 :] = len(seen)
    frame_table = pd.DataFrame({
        "frame": range(1, total_frames + 1), "time_s": np.arange(total_frames) / args.fps,
        "active_track_ids": per_frame.to_numpy(), "cumulative_track_ids": cumulative,
    })
    frame_table.to_csv(args.output_dir / "tracking_counts_by_frame.csv", index=False)

    stats = {
        "source": str(args.tracks_csv), "frames": total_frames,
        "assigned_detections": int(len(tracked)), "unassigned_detections": int((raw["track_id"] < 0).sum()),
        "unique_track_ids": int(tracks["track_id"].nunique()),
        "mean_active_tracks_per_frame": round(float(per_frame.mean()), 3),
        "median_track_observed_duration_s": round(float(tracks["observed_duration_s"].median()), 3),
        "mean_track_observed_duration_s": round(float(tracks["observed_duration_s"].mean()), 3),
        "maximum_track_span_s": round(float(tracks["span_duration_s"].max()), 3),
        "single_frame_tracks": int((tracks["observed_frames"] == 1).sum()),
        "tracks_under_one_second": int((tracks["observed_duration_s"] < 1).sum()),
        "tracks_with_internal_gaps": int((tracks["internal_gaps"] > 0).sum()),
        "class_by_track": {str(k): int(v) for k, v in tracks["class_name"].value_counts().items()},
        "warning": "Track counts and fragmentation are diagnostic proxies. ID switches require human identity labels.",
    }
    (args.output_dir / "track_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(frame_table["time_s"], frame_table["active_track_ids"], color="#17689a", linewidth=1.0)
    ax1.set(xlabel="Clip time (seconds)", ylabel="Active track IDs", title=f"Tracking activity: {args.tracks_csv.stem}")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(frame_table["time_s"], frame_table["cumulative_track_ids"], color="#d46835", linewidth=1.3)
    ax2.set_ylabel("Cumulative IDs (may include fragmented duplicates)", color="#9b431d")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tracking_activity.png", dpi=180)
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
