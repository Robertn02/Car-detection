"""Check that the faster perception settings keep identity quality on the approved 300-frame sequence.

Compares BoT-SORT's own sparse-flow camera-motion compensation with the dense-flow RANSAC replacement used by
bikesafe.perceive, at several frame strides. Metrics are computed only on processed frames.

    python -m bikesafe.validate_tracking
"""

from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "asfarray"):
    np.asfarray = lambda value: np.asarray(value, dtype=float)  # type: ignore[attr-defined]

import motmetrics as mm
import torch
from ultralytics import YOLO

from bikesafe.common import ROOT, VEHICLE_CLASSES, probe_video
from bikesafe.perceive import frame_reader, scaled_tracker_config


def run_variant(video: Path, model_path: Path, tracker: Path, stride: int, dense_gmc: bool, work: Path, device: str) -> pd.DataFrame:
    info = probe_video(video)
    work.mkdir(parents=True, exist_ok=True)
    cfg = scaled_tracker_config(tracker, info.fps / stride, 2.0, work)
    model = YOLO(str(model_path))
    frames_q: queue.Queue = queue.Queue(maxsize=16)
    threading.Thread(target=frame_reader, args=(video, stride, 0, info.frames, frames_q), daemon=True).start()
    warp_state = {"warp": np.eye(2, 3)}
    rows = []
    while (item := frames_q.get()) is not None:
        idx, frame, _, warp_state["warp"] = item
        result = model.track(frame, persist=True, tracker=str(cfg), imgsz=1280, conf=0.10,
                             classes=list(VEHICLE_CLASSES), device=device, verbose=False)[0]
        assert model.predictor is not None
        tracker_obj = model.predictor.trackers[0]
        if dense_gmc and not getattr(tracker_obj.gmc, "uses_dense_flow", False):
            tracker_obj.gmc.apply = lambda raw_frame, detections=None: warp_state["warp"]
            tracker_obj.gmc.uses_dense_flow = True
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        for b, c, tid in zip(boxes.xyxy.cpu().numpy(), boxes.cls.int().cpu().numpy(), boxes.id.int().cpu().numpy()):
            rows.append((idx + 1, int(tid), int(c), *map(float, b)))
    return pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "x1", "y1", "x2", "y2"])


def accumulate(reference: pd.DataFrame, prediction: pd.DataFrame, frames: list[int]) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=False)
    for frame_no in frames:
        gt = reference[reference["frame"] == frame_no]
        pred = prediction[prediction["frame"] == frame_no]
        gt_boxes = np.column_stack([gt.x1, gt.y1, gt.x2 - gt.x1, gt.y2 - gt.y1])
        pred_boxes = np.column_stack([pred.x1, pred.y1, pred.x2 - pred.x1, pred.y2 - pred.y1])
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        if dist.size:
            dist[gt["class_id"].to_numpy()[:, None] != pred["class_id"].to_numpy()[None, :]] = np.nan
        acc.update(gt["track_id"].astype(int).tolist(), pred["track_id"].astype(int).tolist(), dist, frameid=frame_no)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "approved_sequence.mp4")
    parser.add_argument("--reference", type=Path, default=ROOT / "data" / "approved_tracks.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "yolo11n.pt")
    parser.add_argument("--tracker", type=Path, default=ROOT / "configs" / "botsort_vehicle_tuned.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "typology")
    parser.add_argument("--strides", default="1,2,3")
    args = parser.parse_args()
    device = "0" if torch.cuda.is_available() else "cpu"

    reference = pd.read_csv(args.reference)
    reference = reference[reference.track_id >= 0].copy()
    reference["frame"] = reference["frame"] - reference["frame"].min() + 1
    accs, names = [], []
    for stride in (int(s) for s in args.strides.split(",")):
        frames = list(range(1, int(reference.frame.max()) + 1, stride))
        for dense in (False, True):
            name = f"stride{stride}_{'dense' if dense else 'sparse'}_gmc"
            pred = run_variant(args.video, args.model, args.tracker, stride, dense, ROOT / "work" / "tracking_validation" / name, device)
            accs.append(accumulate(reference, pred, frames))
            names.append(name)
            print(f"{name}: {len(pred)} boxes", flush=True)
    metrics = ["num_frames", "mota", "idf1", "precision", "recall", "num_switches", "num_fragmentations"]
    summary = mm.metrics.create().compute_many(accs, names=names, metrics=metrics, generate_overall=False)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "tracking_rate_validation.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
