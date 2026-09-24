"""Create a stratified, machine-prelabelled set for human detection review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

COCO_TO_REVIEW = {2: (0, "car"), 3: (1, "motorcycle"), 5: (2, "bus"), 7: (3, "truck")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    images_dir = args.output_dir / "images" / "review"
    pseudo_dir = args.output_dir / "pseudo_labels" / "review"
    images_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_numbers = sorted({round(1 + index * (total_frames - 1) / (args.samples - 1)) for index in range(args.samples)})
    image_records = []
    for position, frame_no in enumerate(frame_numbers, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        image_path = images_dir / f"video1_frame_{frame_no:06d}.jpg"
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        image_records.append((frame_no, image_path))
        if position % 20 == 0:
            print(f"Extracted {position}/{len(frame_numbers)} review frames", flush=True)
    cap.release()

    model = YOLO(str(args.model))
    manifest_path = args.output_dir / "review_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(["frame", "time_s", "image", "pseudo_label", "prediction_count", "status", "review_notes"])
        for index, (frame_no, image_path) in enumerate(image_records, start=1):
            result = model.predict(str(image_path), imgsz=args.imgsz, classes=list(COCO_TO_REVIEW), conf=args.conf, verbose=False)[0]
            pseudo_path = pseudo_dir / f"{image_path.stem}.txt"
            lines = []
            if result.boxes is not None:
                for class_id, norm, confidence in zip(
                    result.boxes.cls.int().cpu().tolist(),
                    result.boxes.xywhn.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    review_id, _ = COCO_TO_REVIEW[int(class_id)]
                    xc, yc, width, height = norm
                    lines.append(f"{review_id} {xc:.7f} {yc:.7f} {width:.7f} {height:.7f} {confidence:.6f}")
            pseudo_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            writer.writerow([
                frame_no, (frame_no - 1) / fps, image_path.relative_to(args.output_dir),
                pseudo_path.relative_to(args.output_dir), len(lines), "needs_human_review", "",
            ])
            if index % 20 == 0:
                print(f"Predicted {index}/{len(image_records)} review frames", flush=True)

    (args.output_dir / "README.md").write_text(
        """# Detection review set

This is a stratified sample across the source video. `pseudo_labels/` contains model
predictions, not ground truth. A human reviewer must add missed vehicles, remove false
positives, correct boxes/classes, and change each manifest row to `reviewed`.

Classes are `0 car`, `1 motorcycle`, `2 bus`, and `3 truck`. The sixth value in each
pseudo-label row is confidence; remove it when copying corrected labels into a standard
YOLO `labels/` directory. Do not train or report accuracy from this set until review is
complete.
""",
        encoding="utf-8",
    )
    (args.output_dir / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: car\n  1: motorcycle\n  2: bus\n  3: truck\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(image_records)} review images in {args.output_dir}")


if __name__ == "__main__":
    main()
