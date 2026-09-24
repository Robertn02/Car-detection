"""Render a presentation-quality tracking video from a completed tracker CSV."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def colour_for(track_id: int) -> tuple[int, int, int]:
    """Return a bright, repeatable BGR colour for one identity."""
    hue = (track_id * 47) % 180
    pixel = np.uint8([[[hue, 175, 245]]])
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def put_label(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - tw - 8))
    y = max(th + 8, y)
    cv2.rectangle(frame, (x, y - th - 7), (x + tw + 7, y + 2), colour, -1)
    cv2.putText(frame, text, (x + 4, y - 3), font, scale, (18, 18, 18), thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("--title", default="Tuned BoT-SORT vehicle tracking")
    parser.add_argument("--min-hits", type=int, default=15, help="Minimum observed frames for a displayed track")
    parser.add_argument("--min-median-conf", type=float, default=0.15)
    parser.add_argument("--min-box-area", type=float, default=240.0)
    parser.add_argument("--trail", type=int, default=24)
    parser.add_argument("--snapshots", default="10,30,50")
    args = parser.parse_args()

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(args.tracks_csv)
    tracks = tracks[tracks["track_id"] >= 0].copy()
    tracks["box_area"] = (tracks["x2"] - tracks["x1"]) * (tracks["y2"] - tracks["y1"])
    grouped = tracks.groupby("track_id")
    quality = grouped.agg(
        hits=("frame", "nunique"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        median_confidence=("confidence", "median"),
        median_area=("box_area", "median"),
    )
    quality["span_frames"] = quality["last_frame"] - quality["first_frame"] + 1
    display_ids = set(
        quality[
            (quality["hits"] >= args.min_hits)
            & (quality["median_confidence"] >= args.min_median_conf)
            & (quality["median_area"] >= args.min_box_area)
        ].index.astype(int)
    )
    display = tracks[tracks["track_id"].isin(display_ids)].copy()
    per_frame = {int(frame): data for frame, data in display.groupby("frame", sort=False)}

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create {args.output_video}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
    ids_seen: set[int] = set()
    active_counts: list[int] = []
    snapshot_times = {float(value) for value in args.snapshots.split(",") if value.strip()}
    snapshot_dir = args.output_video.parent / f"{args.output_video.stem}_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        rows = per_frame.get(frame_no)
        active_ids: set[int] = set()
        if rows is not None:
            for row in rows.itertuples(index=False):
                tid = int(row.track_id)
                active_ids.add(tid)
                ids_seen.add(tid)
                x1, y1, x2, y2 = map(round, (row.x1, row.y1, row.x2, row.y2))
                colour = colour_for(tid)
                centre = ((x1 + x2) // 2, (y1 + y2) // 2)
                trails[tid].append(centre)
                points = list(trails[tid])
                for index in range(1, len(points)):
                    alpha = index / max(1, len(points) - 1)
                    trail_colour = tuple(round(channel * (0.35 + 0.65 * alpha)) for channel in colour)
                    cv2.line(frame, points[index - 1], points[index], trail_colour, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
                label = f"ID {tid}  {row.class_name}"
                put_label(frame, label, (x1, max(18, y1 - 3)), colour)

        active_counts.append(len(active_ids))
        seconds = (frame_no - 1) / fps
        timestamp = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
        panel_w = min(width - 32, 630)
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (panel_w, 92), (12, 18, 25), -1)
        cv2.addWeighted(overlay, 0.83, frame, 0.17, 0, frame)
        cv2.putText(frame, args.title, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        status = f"time {timestamp}   active {len(active_ids)}   stable IDs seen {len(ids_seen)}"
        cv2.putText(frame, status, (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 220, 245), 1, cv2.LINE_AA)
        note = "Stable tracks only (>=15 observations); IDs remain tracker estimates"
        (tw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(frame, note, (max(18, width - tw - 22), height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        writer.write(frame)

        for target in list(snapshot_times):
            if seconds >= target:
                cv2.imwrite(str(snapshot_dir / f"tracking_{int(target):02d}s.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
                snapshot_times.remove(target)
        if frame_no % 300 == 0:
            print(f"Rendered {frame_no}/{total_frames} frames", flush=True)

    cap.release()
    writer.release()
    stats = {
        "source": str(args.source),
        "tracks_csv": str(args.tracks_csv),
        "frames_rendered": frame_no,
        "raw_tracker_ids": int(tracks["track_id"].nunique()),
        "displayed_stable_ids": len(display_ids),
        "minimum_hits": args.min_hits,
        "minimum_median_confidence": args.min_median_conf,
        "minimum_median_box_area": args.min_box_area,
        "mean_active_displayed_tracks": float(np.mean(active_counts)) if active_counts else 0.0,
        "max_active_displayed_tracks": max(active_counts, default=0),
    }
    stats_path = args.output_video.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    quality.reset_index().to_csv(args.output_video.with_name(f"{args.output_video.stem}_track_quality.csv"), index=False)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
