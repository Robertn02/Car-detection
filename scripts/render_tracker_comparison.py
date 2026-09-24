"""Render a synchronized 2x2 visual comparison of tracker outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def colour_for(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    pixel = np.uint8([[[hue, 175, 245]]])
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def draw_panel(frame: np.ndarray, rows: pd.DataFrame | None, title: str, metrics: str) -> np.ndarray:
    canvas = frame.copy()
    active: set[int] = set()
    if rows is not None:
        for row in rows.itertuples(index=False):
            tid = int(row.track_id)
            if tid < 0:
                continue
            active.add(tid)
            x1, y1, x2, y2 = map(round, (row.x1, row.y1, row.x2, row.y2))
            colour = colour_for(tid)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"#{tid}", (x1 + 2, max(24, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, 2, cv2.LINE_AA)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 112), (10, 16, 22), -1)
    cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0, canvas)
    cv2.putText(canvas, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (250, 250, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, metrics, (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 220, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"active tracks {len(active)}", (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("trackers", nargs=4, help="Display name=tracks.csv, exactly four")
    args = parser.parse_args()
    args.output_video.parent.mkdir(parents=True, exist_ok=True)

    metric_table = pd.read_csv(args.metrics_csv, index_col=0)
    trackers: list[tuple[str, str, dict[int, pd.DataFrame]]] = []
    for spec in args.trackers:
        display_name, path = spec.split("=", 1)
        key = Path(path).parent.name
        data = pd.read_csv(path)
        frames = {int(frame): rows for frame, rows in data.groupby("frame", sort=False)}
        metrics = metric_table.loc[key]
        metric_text = (
            f"MOTA {100 * metrics['mota']:.1f}%  |  IDF1 {100 * metrics['idf1']:.1f}%"
            f"  |  switches {int(metrics['num_switches'])}"
        )
        trackers.append((display_name, metric_text, frames))

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tile_size = (width // 2, height // 2)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_size[0] * 2, tile_size[1] * 2))
    if not writer.isOpened():
        raise SystemExit(f"Could not create {args.output_video}")

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        panels = []
        for title, metric_text, frames in trackers:
            panel = draw_panel(frame, frames.get(frame_no), title, metric_text)
            panels.append(cv2.resize(panel, tile_size, interpolation=cv2.INTER_AREA))
        output = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        writer.write(output)
        if frame_no % 150 == 0:
            print(f"Rendered comparison frame {frame_no}", flush=True)

    cap.release()
    writer.release()
    print(f"Saved {args.output_video} ({frame_no} frames)")


if __name__ == "__main__":
    main()
