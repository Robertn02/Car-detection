"""
Redraw YOLO detections onto the source clip with readable annotations.

Ultralytics scales its label font with the frame size, so on 1080p footage the
text boxes swamp the small distant detections we most want to look at. This
reads the saved label .txt files instead and redraws them with thin boxes and
small text, colour-coded by confidence so the near/far quality gap is obvious
at a glance:

    green  = conf >= 0.60      confident, usually near vehicles
    amber  = 0.40 <= conf < 0.60
    red    = conf < 0.40       usually small/distant vehicles

No inference happens here, so it runs in seconds rather than minutes.

Usage:
    python scripts/render_clean_overlay.py <source_video> <labels_dir> <output_mp4>
"""

import argparse
import re
from pathlib import Path

import cv2

FRAME_RE = re.compile(r"_(\d+)$")

HIGH = (80, 220, 100)    # BGR green
MID = (60, 200, 250)     # BGR amber
LOW = (70, 70, 240)      # BGR red


def colour_for(conf: float) -> tuple:
    if conf >= 0.60:
        return HIGH
    if conf >= 0.40:
        return MID
    return LOW


def load_labels(labels_dir: Path) -> dict:
    """frame_number -> list of (cls, xc, yc, w, h, conf) in normalised coords."""
    out = {}
    for txt in labels_dir.glob("*.txt"):
        match = FRAME_RE.search(txt.stem)
        if not match:
            continue
        frame = int(match.group(1))
        dets = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            dets.append((cls, xc, yc, w, h, conf))
        out[frame] = dets
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("labels_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--label", default="car")
    args = ap.parse_args()

    labels = load_labels(args.labels_dir)
    print(f"Loaded detections for {len(labels):,} frames")

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_no = 0
    total_boxes = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        dets = labels.get(frame_no, [])
        total_boxes += len(dets)

        for _cls, xc, yc, w, h, conf in dets:
            x1 = int((xc - w / 2) * width)
            y1 = int((yc - h / 2) * height)
            x2 = int((xc + w / 2) * width)
            y2 = int((yc + h / 2) * height)
            colour = colour_for(conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            text = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), colour, -1)
            cv2.putText(
                frame, text, (x1 + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
            )

        banner = f"frame {frame_no}  |  {len(dets)} {args.label}(s)"
        cv2.rectangle(frame, (12, 12), (12 + 9 * len(banner), 52), (30, 30, 30), -1)
        cv2.putText(
            frame, banner, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {frame_no:,} frames ({total_boxes:,} boxes) to {args.output}")


if __name__ == "__main__":
    main()
