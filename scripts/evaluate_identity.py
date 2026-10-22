"""Identity errors of the pipeline's tracks on the approved 300-frame reference sequence, by kind.

* transfers: a predicted track that moves from one reference vehicle to another (one id on two vehicles) - what
  bikesafe.identity.split_switches removes;
* breaks: one reference vehicle covered by several predicted tracks in turn - what bikesafe.stitch joins.

Scores the tracks as the tracker wrote them and after the identity split (with and without the vehicle ids of
bikesafe.stitch), and writes the transfer steps with the split detector's score at each track step, so the detector
can be checked step by step.

    python scripts/evaluate_identity.py data/approved_tracks.csv work/perception_approved/approved_sequence \
        data/approved_sequence.mp4 results/stitching
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "asfarray"):  # motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.x
    np.asfarray = lambda value: np.asarray(value, dtype=float)  # type: ignore[attr-defined]
import motmetrics as mm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikesafe.identity import appearance_descriptors, split_switches, step_scores  # noqa: E402
from bikesafe.stitch import stitch, vehicle_ids  # noqa: E402

METRICS = ["idf1", "idp", "idr", "mota", "num_switches", "num_fragmentations", "num_unique_objects"]


def xywh(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame.x1, frame.y1, frame.x2 - frame.x1, frame.y2 - frame.y1])


def matches(reference: pd.DataFrame, prediction: pd.DataFrame, frames: list[int], iou: float = 0.5):
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        gt, pred = reference[reference.frame == f], prediction[prediction.frame == f]
        d = mm.distances.iou_matrix(xywh(gt), xywh(pred), max_iou=iou)
        if d.size:
            d[gt.class_id.to_numpy()[:, None] != pred.class_id.to_numpy()[None, :]] = np.nan
        acc.update(gt.track_id.astype(int).tolist(), pred.track_id.astype(int).tolist(), d, frameid=f)
    return acc


def identity_errors(acc) -> tuple[int, int, pd.DataFrame]:
    """(transfers, breaks, matched pairs) from an accumulator: per predicted track, changes of the reference vehicle
    it covers; per reference vehicle, changes of the predicted track covering it."""
    ev = acc.mot_events
    m = ev[ev.Type.isin(["MATCH", "SWITCH"])].reset_index()
    m = m.sort_values("FrameId")
    transfers = int(sum((g.OId.to_numpy()[1:] != g.OId.to_numpy()[:-1]).sum() for _, g in m.groupby("HId")))
    breaks = int(sum((g.HId.to_numpy()[1:] != g.HId.to_numpy()[:-1]).sum() for _, g in m.groupby("OId")))
    return transfers, breaks, m[["FrameId", "OId", "HId"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("perception_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(args.reference_csv)
    reference = reference[reference.track_id >= 0].copy()
    reference["frame"] = reference.frame - reference.frame.min() + 1
    meta = json.loads((args.perception_dir / "meta.json").read_text(encoding="utf-8"))
    dets = pd.read_parquet(args.perception_dir / "detections.parquet")
    dets = dets[dets.track_id >= 0].reset_index(drop=True)
    desc = appearance_descriptors(args.video, dets)
    split = split_switches(dets, desc)
    for d in (dets, split):
        d["frame"] = d.frame - meta["start_frame"] + 1
    frames = sorted(int(f) for f in dets.frame.unique())

    rows, accs, names = [], [], []
    for name, pred in (("tracker", dets), ("identity split", split)):
        links = stitch(pred.assign(frame=pred.frame + meta["start_frame"] - 1), meta["width"], meta["processed_fps"])
        joined = pred.assign(track_id=pred.track_id.map(vehicle_ids(pred.track_id.unique(), links)))
        for label, p in ((name, pred), (f"{name} + vehicle ids", joined)):
            acc = matches(reference, p, frames)
            transfers, breaks, _ = identity_errors(acc)
            accs.append(acc)
            names.append(label)
            rows.append({"setting": label, "tracks": int(p.track_id.nunique()), "transfers": transfers,
                         "breaks": breaks})
    summary = mm.metrics.create().compute_many(accs, names=names, metrics=METRICS, generate_overall=False)
    table = summary.join(pd.DataFrame(rows).set_index("setting"))
    table.to_csv(args.out / "identity_split_metrics.csv")
    print(table.round(3).to_string())

    # step-level check of the detector: which steps of the tracker's tracks are real transfers
    _, _, matched = identity_errors(accs[0])
    owner = {(int(r.FrameId), int(r.HId)): int(r.OId) for r in matched.itertuples()}
    steps = step_scores(dets.assign(frame=dets.frame + meta["start_frame"] - 1), desc)
    steps["frame_eval"] = steps.frame - meta["start_frame"] + 1
    steps["prev_eval"] = steps.prev_frame - meta["start_frame"] + 1
    steps["owner"] = [owner.get((int(f), int(t))) for f, t in zip(steps.frame_eval, steps.track_id)]
    steps["prev_owner"] = [owner.get((int(f), int(t))) for f, t in zip(steps.prev_eval, steps.track_id)]
    known = steps.owner.notna() & steps.prev_owner.notna()
    steps["transfer"] = known & (steps.owner != steps.prev_owner)
    steps[known].to_csv(args.out / "identity_split_steps.csv", index=False)
    t = steps[known]
    print(f"steps with both ends matched: {len(t)}, transfers among them: {int(t.transfer.sum())}")
    for col in ("appearance", "motion", "score"):
        if col in t:
            print(col, "transfer median", round(float(t[t.transfer][col].median()), 3),
                  "| other median", round(float(t[~t.transfer][col].median()), 3),
                  "| other p99", round(float(t[~t.transfer][col].quantile(0.99)), 3))


if __name__ == "__main__":
    main()
