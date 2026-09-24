"""Build a conservative pseudo-labelled crop/manifest set from tracker output."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples-per-track", type=int, default=6)
    parser.add_argument("--min-track-frames", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=555)
    args = parser.parse_args()
    random.seed(args.seed)

    data = pd.read_csv(args.tracks_csv)
    data = data[(data["track_id"] >= 0) & (data["confidence"] >= args.min_confidence)].copy()
    eligible = [int(tid) for tid, group in data.groupby("track_id") if group["frame"].nunique() >= args.min_track_frames]
    data = data[data["track_id"].isin(eligible)]
    if len(eligible) < 2:
        raise SystemExit("Need at least two eligible tracks")

    selections = []
    for track_id, group in data.groupby("track_id"):
        group = group.sort_values("frame").drop_duplicates("frame")
        indexes = [round(i * (len(group) - 1) / (min(args.samples_per_track, len(group)) - 1)) for i in range(min(args.samples_per_track, len(group)))]
        selections.extend(group.iloc[indexes].to_dict("records"))

    by_frame = defaultdict(list)
    for row in selections:
        by_frame[int(row["frame"])].append(row)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.source))
    crop_records = []
    for frame_no in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        for row in by_frame[frame_no]:
            x1 = max(0, min(width - 1, round(row["x1"])))
            y1 = max(0, min(height - 1, round(row["y1"])))
            x2 = max(x1 + 1, min(width, round(row["x2"])))
            y2 = max(y1 + 1, min(height, round(row["y2"])))
            crop = frame[y1:y2, x1:x2]
            track_id = int(row["track_id"])
            path = crops_dir / f"track_{track_id:04d}_frame_{frame_no:06d}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            crop_records.append({"track_id": track_id, "frame": frame_no, "path": path.relative_to(args.output_dir)})
    cap.release()

    by_track = defaultdict(list)
    for record in crop_records:
        by_track[record["track_id"]].append(record)
    track_ids = sorted(by_track)
    triplets = []
    for track_id in track_ids:
        records = sorted(by_track[track_id], key=lambda item: item["frame"])
        if len(records) < 2:
            continue
        for index in range(len(records) - 1):
            anchor = records[index]
            positive = records[-1] if index == 0 else records[index + 1]
            negative_id = random.choice([candidate for candidate in track_ids if candidate != track_id])
            negative = random.choice(by_track[negative_id])
            split = "validation" if track_id % 5 == 0 else "train"
            triplets.append((split, anchor["path"], positive["path"], negative["path"], track_id, negative_id, "needs_human_review"))

    with (args.output_dir / "triplets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "anchor", "positive", "negative", "anchor_track_id", "negative_track_id", "status"])
        writer.writerows(triplets)
    (args.output_dir / "README.md").write_text(
        """# Pseudo-labelled ReID triplet candidates

These crops and triplets were generated from tracker IDs. They are candidates for human
review, not verified identity labels. Tracker fragmentation can put the same physical
vehicle under two IDs, and an ID switch can put different vehicles under one ID.

Review all anchor/positive and anchor/negative relationships before training. Keep every
physical vehicle in only one split. Benchmark a pretrained ReID embedding before training
a custom triplet-loss model.
""",
        encoding="utf-8",
    )
    print(f"Created {len(crop_records)} crops and {len(triplets)} triplet candidates from {len(track_ids)} tracks")


if __name__ == "__main__":
    main()
