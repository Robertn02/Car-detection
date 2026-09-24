"""
Turn YOLO's per-frame label .txt files into a single machine-readable CSV
plus a short summary of detection behaviour over the clip.

YOLO writes one .txt per frame that had at least one detection, named
<video_stem>_<frame_number>.txt. Each line is:

    class x_center y_center width height [confidence]

with box coordinates normalised to 0-1 and confidence present only when
the run used save_conf=True.

Usage:
    python scripts/summarize_detections.py <labels_dir> <output_csv> [--width 1920] [--height 1080] [--fps 29.97]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# COCO class ids that ultralytics' pretrained models use for vehicles.
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

FRAME_RE = re.compile(r"_(\d+)$")


def parse_labels(labels_dir: Path) -> pd.DataFrame:
    rows = []
    for txt in sorted(labels_dir.glob("*.txt")):
        match = FRAME_RE.search(txt.stem)
        if not match:
            continue
        frame = int(match.group(1))
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else None
            rows.append(
                {
                    "frame": frame,
                    "class_id": cls,
                    "class_name": COCO_NAMES.get(cls, str(cls)),
                    "x_center": xc,
                    "y_center": yc,
                    "width": w,
                    "height": h,
                    "confidence": conf,
                }
            )
    return pd.DataFrame(rows)


def add_pixel_columns(df: pd.DataFrame, width: int, height: int, fps: float) -> pd.DataFrame:
    df["time_s"] = (df["frame"] - 1) / fps
    df["x_center_px"] = df["x_center"] * width
    df["y_center_px"] = df["y_center"] * height
    df["width_px"] = df["width"] * width
    df["height_px"] = df["height"] * height
    df["box_area_px"] = df["width_px"] * df["height_px"]
    # Rough proximity proxy: bigger box == closer car.
    df["size_bucket"] = pd.cut(
        df["box_area_px"],
        bins=[0, 32 * 32, 96 * 96, float("inf")],
        labels=["small (<32px)", "medium", "large (>96px)"],
    )
    return df


def summarise(df: pd.DataFrame, total_frames: int) -> str:
    if df.empty:
        return "No detections found."

    frames_with_det = df["frame"].nunique()
    # Include every source frame. Omitting zero-detection frames biases the mean upward.
    per_frame = df.groupby("frame").size().reindex(range(1, total_frames + 1), fill_value=0)
    empty_frames = per_frame[per_frame == 0].index.tolist()

    lines = [
        f"Total detections        : {len(df):,}",
        f"Frames with >=1 detection: {frames_with_det:,} of {total_frames:,} "
        f"({frames_with_det / total_frames:.1%})",
        f"Detections per frame    : mean {per_frame.mean():.1f}, "
        f"median {per_frame.median():.0f}, max {per_frame.max()}",
        "",
        "Confidence distribution:",
    ]
    if df["confidence"].notna().any():
        q = df["confidence"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        for k in ["min", "10%", "25%", "50%", "75%", "90%", "max"]:
            lines.append(f"  {k:>5} : {q[k]:.3f}")
        lines.append(f"  mean  : {df['confidence'].mean():.3f}")
        low = (df["confidence"] < 0.5).mean()
        lines.append(f"  share below 0.50 : {low:.1%}")
    else:
        lines.append("  (no confidence values - run with save_conf=True)")

    lines += ["", "Detections by box size (proximity proxy):"]
    for bucket, count in df["size_bucket"].value_counts().sort_index().items():
        mean_conf = df.loc[df["size_bucket"] == bucket, "confidence"].mean()
        conf_txt = f", mean conf {mean_conf:.3f}" if pd.notna(mean_conf) else ""
        lines.append(f"  {bucket:>14} : {count:>6,} ({count / len(df):.1%}){conf_txt}")

    lines += ["", "Detections by class:"]
    for name, count in df["class_name"].value_counts().items():
        lines.append(f"  {name:>12} : {count:,}")

    if empty_frames:
        lines += [
            "",
            f"Frames with zero detections in the source range: {len(empty_frames)}",
            f"  e.g. {empty_frames[:15]}",
        ]

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--total-frames", type=int, default=1802)
    args = ap.parse_args()

    df = parse_labels(args.labels_dir)
    if not df.empty:
        df = add_pixel_columns(df, args.width, args.height, args.fps)
        df = df.sort_values(["frame", "confidence"], ascending=[True, False])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(df):,} rows to {args.output_csv}")
    print()
    print(summarise(df, args.total_frames))


if __name__ == "__main__":
    main()
