"""Export the busiest contiguous tracker window for manual identity review."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=29.97)
    args = parser.parse_args()

    data = pd.read_csv(args.tracks_csv)
    total_frames = int(data["frame"].max())
    window = max(1, round(args.seconds * args.fps))
    activity = data[data["track_id"] >= 0].groupby("frame")["track_id"].nunique().reindex(range(1, total_frames + 1), fill_value=0)
    rolling = activity.rolling(window, min_periods=window).mean()
    end_frame = int(rolling.idxmax())
    start_frame = max(1, end_frame - window + 1)

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    extracted = 0
    for frame_no in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(images_dir / f"frame_{frame_no:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted += 1
    cap.release()

    subset = data[(data["frame"] >= start_frame) & (data["frame"] <= end_frame)].copy()
    subset["review_status"] = "needs_human_review"
    subset["review_notes"] = ""
    subset.to_csv(args.output_dir / "pseudo_tracks_for_review.csv", index=False)
    (args.output_dir / "README.md").write_text(
        f"""# Tracking identity review

Frames {start_frame}–{end_frame} ({start_frame / args.fps:.1f}–{end_frame / args.fps:.1f}s)
were selected because they have the highest mean tracking activity in this clip.

`pseudo_tracks_for_review.csv` contains tracker output, not ground truth. Review every
frame in order and correct boxes, classes, and IDs. Record identity switches, fragments,
missed detections, and occlusion/reappearance events. Formal IDF1/HOTA evaluation is only
valid after this review.
""",
        encoding="utf-8",
    )
    print(f"Exported {extracted} frames ({start_frame}-{end_frame}) to {args.output_dir}")


if __name__ == "__main__":
    main()
