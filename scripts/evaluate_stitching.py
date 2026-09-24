"""Evaluate track stitching (bikesafe.stitch) against the approved 300-frame reference sequence.

Two settings, each scored with and without stitching on the same tracker output:
  * the tracker benchmark setting: every frame at 30 fps (a scripts/track_vehicles.py CSV);
  * the corpus pipeline setting: bikesafe.perceive at its processed frame rate (a perception folder), scored on the
    processed frames only.

    python scripts/evaluate_stitching.py data/approved_tracks.csv results/stitching \
        benchmark_30fps=outputs/approved/botsort_vehicle_tuned_tracks.csv pipeline=work/perception_approved/approved_sequence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.x.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda value: np.asarray(value, dtype=float)  # type: ignore[attr-defined]

import motmetrics as mm

from bikesafe.stitch import duplicates, stitch, vehicle_ids

METRICS = ["num_frames", "num_objects", "mota", "idf1", "idp", "idr", "precision", "recall", "num_switches",
           "num_fragmentations", "mostly_tracked", "num_unique_objects"]


def load_predictions(path: Path) -> tuple[pd.DataFrame, int, float]:
    """Tracks in the evaluator's frame numbering (1-based), frame width and processed frame rate."""
    if path.is_dir():
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        dets = pd.read_parquet(path / "detections.parquet")
        dets["frame"] = dets.frame - meta["start_frame"] + 1
        return dets, meta["width"], meta["processed_fps"]
    dets = pd.read_csv(path)
    fps = 1.0 / float(np.median(np.diff(np.unique(dets.time_s))))
    return dets, int(np.ceil(dets.x2.max())), fps


def xywh(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame.x1, frame.y1, frame.x2 - frame.x1, frame.y2 - frame.y1])


def accumulate(reference: pd.DataFrame, prediction: pd.DataFrame, frames: list[int], iou: float) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=False)
    for frame_no in frames:
        gt = reference[reference.frame == frame_no]
        pred = prediction[prediction.frame == frame_no]
        distances = mm.distances.iou_matrix(xywh(gt), xywh(pred), max_iou=iou)
        if distances.size:
            distances[gt.class_id.to_numpy()[:, None] != pred.class_id.to_numpy()[None, :]] = np.nan
        acc.update(gt.track_id.astype(int).tolist(), pred.track_id.astype(int).tolist(), distances, frameid=frame_no)
    return acc


def without_duplicates(tracks: pd.DataFrame) -> pd.DataFrame:
    """Drop the shorter track of every duplicate pair (a second box on one vehicle at the same time)."""
    return tracks[~tracks.track_id.isin(set(duplicates(tracks).B))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("predictions", nargs="+", help="name=tracks.csv or name=perception_folder")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(args.reference_csv)
    reference = reference[reference.track_id >= 0].copy()
    reference["frame"] = reference.frame - reference.frame.min() + 1
    # The reference boxes a doubly detected vehicle twice too (car + truck), as separate identities. Scoring also with
    # the second boxes removed from both sides isolates what joining broken tracks does.
    clean_reference = without_duplicates(reference)
    accumulators, names, link_rows = [], [], []
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        dets, width, fps = load_predictions(Path(path))
        dets = dets[dets.track_id >= 0]
        frames = sorted(int(f) for f in dets.frame.unique()) if Path(path).is_dir() else list(range(1, int(reference.frame.max()) + 1))
        links = stitch(dets, width, fps)
        mapping = vehicle_ids(dets.track_id.unique(), links)
        stitched = dets.assign(track_id=dets.track_id.map(mapping))
        clean = without_duplicates(dets)
        clean_links = stitch(clean, width, fps)
        clean_stitched = clean.assign(track_id=clean.track_id.map(vehicle_ids(clean.track_id.unique(), clean_links)))
        for label, ref, prediction in ((name, reference, dets), (f"{name} + vehicle ids", reference, stitched),
                                       (f"{name}, duplicates removed", clean_reference, clean),
                                       (f"{name}, duplicates removed + break joins", clean_reference, clean_stitched)):
            accumulators.append(accumulate(ref, prediction, frames, args.iou))
            names.append(label)
        link_rows.append({"setting": name, "processed_fps": round(fps, 2), "tracks": int(dets.track_id.nunique()),
                          "duplicate_links": int((links.kind == "duplicate").sum()),
                          "break_links": int((links.kind == "gap").sum()), "vehicles": len(set(mapping.values())),
                          "break_links_without_duplicates": int((clean_links.kind == "gap").sum()),
                          "reference_duplicate_tracks": int(reference.track_id.nunique() - clean_reference.track_id.nunique())})

    summary = mm.metrics.create().compute_many(accumulators, names=names, metrics=METRICS, generate_overall=False)
    summary.to_csv(args.output_dir / "stitching_mot_metrics.csv")
    pd.DataFrame(link_rows).to_csv(args.output_dir / "stitching_links_summary.csv", index=False)
    print(mm.io.render_summary(summary, formatters=mm.metrics.create().formatters))
    print(pd.DataFrame(link_rows).to_string(index=False))


if __name__ == "__main__":
    main()
