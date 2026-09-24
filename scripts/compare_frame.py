"""
Draw the same frame once per run, stacked, so the runs can be compared by eye.

Detection counts alone cannot tell you whether a setting is better - a model
that emits twice as many boxes might just be emitting twice as much noise.
This crops the horizon band (where the distant cars live) and stacks one strip
per run so true positives and false positives are distinguishable visually.

Usage:
    python scripts/compare_frame.py <source_video> <frame_no> <output_jpg> \
        <name>=<labels_dir> [<name>=<labels_dir> ...] [--crop y1,y2]
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

FRAME_RE = re.compile(r"_(\d+)$")


def load_frame_dets(labels_dir: Path, stem: str, frame_no: int):
    path = labels_dir / f"{stem}_{frame_no}.txt"
    if not path.exists():
        return []
    dets = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = (float(p) for p in parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        dets.append((xc, yc, w, h, conf))
    return dets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("frame_no", type=int)
    ap.add_argument("output", type=Path)
    ap.add_argument("runs", nargs="+", help="name=labels_dir")
    ap.add_argument("--crop", default=None, help="y1,y2 band to zoom into")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_no - 1)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame_no}")

    height, width = base.shape[:2]
    stem = args.source.stem

    y1, y2 = (0, height)
    if args.crop:
        y1, y2 = (int(v) for v in args.crop.split(","))

    panels = []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        dets = load_frame_dets(Path(path), stem, args.frame_no)
        img = base.copy()
        for xc, yc, w, h, conf in dets:
            bx1 = int((xc - w / 2) * width)
            by1 = int((yc - h / 2) * height)
            bx2 = int((xc + w / 2) * width)
            by2 = int((yc + h / 2) * height)
            colour = (80, 220, 100) if conf >= 0.6 else (60, 200, 250) if conf >= 0.4 else (70, 70, 240)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), colour, 2)

        strip = img[y1:y2, :]
        if args.scale != 1.0:
            strip = cv2.resize(strip, None, fx=args.scale, fy=args.scale)

        bar = np.full((34, strip.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(
            bar, f"{name}  -  {len(dets)} cars", (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA,
        )
        panels.append(np.vstack([bar, strip]))

    out = np.vstack(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Wrote {args.output}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
