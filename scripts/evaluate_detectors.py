"""Compare detectors on the same approved YOLO-format validation frames."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

COCO_TO_REVIEW = {2: 0, 3: 1, 5: 2, 7: 3}


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area1 + area2 - intersection, 1e-9)


def load_ground_truth(images_dir: Path, labels_dir: Path):
    ground_truth = defaultdict(lambda: defaultdict(list))
    images = sorted(images_dir.glob("*.jpg"))
    for image_path in images:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, xc, yc, bw, bh = map(float, line.split()[:5])
            ground_truth[int(class_id)][image_path.name].append(np.array([
                (xc - bw / 2) * width, (yc - bh / 2) * height,
                (xc + bw / 2) * width, (yc + bh / 2) * height,
            ]))
    return images, ground_truth


def score_predictions(predictions, ground_truth, threshold=0.5):
    total_gt = sum(len(boxes) for per_image in ground_truth.values() for boxes in per_image.values())
    matched = defaultdict(set)
    ranked = sorted(predictions, key=lambda row: row["confidence"], reverse=True)
    tp, fp = [], []
    for prediction in ranked:
        key = (prediction["class_id"], prediction["image"])
        candidates = np.array(ground_truth[prediction["class_id"]][prediction["image"]])
        overlaps = iou(prediction["box"], candidates)
        if len(overlaps):
            best = int(overlaps.argmax())
            if overlaps[best] >= threshold and best not in matched[key]:
                matched[key].add(best)
                tp.append(1)
                fp.append(0)
                continue
        tp.append(0)
        fp.append(1)
    tp_cumulative = np.cumsum(tp)
    fp_cumulative = np.cumsum(fp)
    recall = tp_cumulative / max(1, total_gt)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1)
    interpolated = [precision[recall >= target].max() if np.any(recall >= target) else 0 for target in np.linspace(0, 1, 101)]
    ap50 = float(np.mean(interpolated))
    selected = [index for index, row in enumerate(ranked) if row["confidence"] >= 0.25]
    if selected:
        index = selected[-1]
        selected_tp = int(tp_cumulative[index])
        selected_fp = int(fp_cumulative[index])
    else:
        selected_tp = selected_fp = 0
    selected_fn = total_gt - selected_tp
    p25 = selected_tp / max(1, selected_tp + selected_fp)
    r25 = selected_tp / max(1, total_gt)
    f1 = 2 * p25 * r25 / max(1e-9, p25 + r25)
    return {
        "ground_truth_boxes": total_gt, "predictions": len(predictions), "ap50": ap50,
        "precision_at_0.25": p25, "recall_at_0.25": r25, "f1_at_0.25": f1,
        "tp_at_0.25": selected_tp, "fp_at_0.25": selected_fp, "fn_at_0.25": selected_fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("models", nargs="+", help="name=model.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images, ground_truth = load_ground_truth(args.images_dir, args.labels_dir)
    results = []
    for spec in args.models:
        name, model_path = spec.split("=", 1)
        print(f"Evaluating {name} on {len(images)} images", flush=True)
        model = YOLO(model_path)
        predictions = []
        inference_ms = []
        for image_path, result in zip(images, model.predict([str(path) for path in images], imgsz=args.imgsz, conf=0.001, iou=0.7, classes=list(COCO_TO_REVIEW), stream=True, verbose=False)):
            inference_ms.append(float(result.speed.get("inference", 0)))
            if result.boxes is None:
                continue
            for class_id, confidence, box in zip(
                result.boxes.cls.int().cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().numpy(),
            ):
                predictions.append({
                    "image": image_path.name, "class_id": COCO_TO_REVIEW[int(class_id)],
                    "confidence": float(confidence), "box": box.astype(float),
                })
        metrics = score_predictions(predictions, ground_truth)
        metrics.update({"model": name, "weights": model_path, "mean_inference_ms": float(np.mean(inference_ms))})
        results.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    frame = pd.DataFrame(results).sort_values(["ap50", "f1_at_0.25"], ascending=False)
    frame.to_csv(args.output_dir / "detector_comparison.csv", index=False)
    best = frame.iloc[0].to_dict()
    (args.output_dir / "best_detector.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    labels = frame["model"]
    axes[0].bar(labels, 100 * frame["ap50"], color="#2678a5")
    axes[0].set(title="AP50", ylabel="Percent")
    axes[1].bar(labels, 100 * frame["f1_at_0.25"], color="#4b9a62")
    axes[1].set(title="F1 at confidence 0.25", ylabel="Percent")
    axes[2].bar(labels, frame["mean_inference_ms"], color="#d37843")
    axes[2].set(title="Mean CPU inference", ylabel="Milliseconds/image")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Detector comparison on approved validation frames", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output_dir / "detector_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Best detector: {best['model']}")


if __name__ == "__main__":
    main()
