"""Track road vehicles and save persistent IDs, an annotated video, and snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def colour_for(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (160, 160, 160)
    hue = (track_id * 47) % 180
    pixel = np.uint8([[[hue, 190, 235]]])
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--classes", default="2,3,5,7")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--snapshots", default="10,30,50", help="Seconds at which to save annotated stills")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional bounded pilot length")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated video rendering for benchmark runs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker_name = Path(args.tracker).stem
    classes = [int(item) for item in args.classes.split(",")]
    snapshots_s = {float(item) for item in args.snapshots.split(",") if item.strip()}

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    csv_path = args.output_dir / f"{tracker_name}_tracks.csv"
    video_path = args.output_dir / f"{tracker_name}_annotated.mp4"
    writer = None if args.no_video else cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer is not None and not writer.isOpened():
        raise SystemExit(f"Could not create {video_path}")

    model = YOLO(str(args.model))
    started = time.perf_counter()
    unique_ids: set[int] = set()
    detections_written = 0
    frames_processed = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow([
            "frame", "time_s", "track_id", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "x_center", "y_center", "width", "height",
        ])
        results = model.track(
            source=str(args.source), tracker=args.tracker, stream=True, persist=True,
            imgsz=args.imgsz, classes=classes, conf=args.conf, verbose=False,
        )
        for frame_no, result in enumerate(results, start=1):
            frames_processed = frame_no
            frame = result.orig_img.copy()
            current_ids: set[int] = set()
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.int().cpu().numpy()
                if boxes.id is None:
                    track_ids = np.full(len(boxes), -1, dtype=int)
                else:
                    track_ids = boxes.id.int().cpu().numpy()

                for coords, norm, confidence, class_id, track_id in zip(xyxy, xywhn, confs, class_ids, track_ids):
                    x1, y1, x2, y2 = (float(v) for v in coords)
                    xc, yc, bw, bh = (float(v) for v in norm)
                    tid = int(track_id)
                    cid = int(class_id)
                    if tid >= 0:
                        unique_ids.add(tid)
                        current_ids.add(tid)
                    out.writerow([
                        frame_no, (frame_no - 1) / fps, tid, cid, COCO_NAMES.get(cid, str(cid)),
                        float(confidence), x1, y1, x2, y2, xc, yc, bw, bh,
                    ])
                    detections_written += 1
                    colour = colour_for(tid)
                    p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
                    cv2.rectangle(frame, p1, p2, colour, 2)
                    label = f"ID {tid if tid >= 0 else '?'} {COCO_NAMES.get(cid, cid)} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    top = max(0, p1[1] - th - 8)
                    cv2.rectangle(frame, (p1[0], top), (p1[0] + tw + 6, top + th + 7), colour, -1)
                    cv2.putText(frame, label, (p1[0] + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

            banner = f"{tracker_name} | frame {frame_no}/{total_frames} | active IDs {len(current_ids)} | IDs seen {len(unique_ids)}"
            cv2.rectangle(frame, (12, 12), (min(width - 12, 20 + 10 * len(banner)), 52), (25, 25, 25), -1)
            cv2.putText(frame, banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(frame)

            second = (frame_no - 1) / fps
            for target in list(snapshots_s):
                if second >= target:
                    cv2.imwrite(str(args.output_dir / f"{tracker_name}_t{int(target):02d}s.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    snapshots_s.remove(target)
            if frame_no % 150 == 0:
                elapsed = time.perf_counter() - started
                print(f"{tracker_name}: {frame_no}/{total_frames} frames, {len(unique_ids)} IDs, {elapsed:.1f}s", flush=True)
            if args.max_frames and frame_no >= args.max_frames:
                break

    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - started
    meta = {
        "source": str(args.source), "model": str(args.model), "tracker": args.tracker,
        "imgsz": args.imgsz, "classes": classes, "confidence_threshold": args.conf,
        "fps": fps, "frames": frames_processed, "source_frames": total_frames,
        "max_frames": args.max_frames, "detections": detections_written, "video_rendered": not args.no_video,
        "unique_track_ids": len(unique_ids), "elapsed_seconds": round(elapsed, 2),
    }
    (args.output_dir / f"{tracker_name}_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
