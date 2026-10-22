"""Keep track labels attached to the right vehicle when the detection stage is re-run.

A track label names a tracker id (video, track_id) from one particular perception run. The tracker hands out ids in
order of appearance, so re-running perception - on another GPU, in half precision, at another frame rate or with a
newer ultralytics - numbers the same vehicles differently, and the labels would silently describe other vehicles.

Every track card therefore also records where its vehicle was: the boxes shown on the card (start, middle, end and
zoom frames) go to data/typology/track_anchors.csv, keyed by card id. Before labels are used, each card's boxes are
looked up in the current detections and the label follows the track that covers them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bikesafe.common import ROOT

ANCHORS = ROOT / "data" / "typology" / "track_anchors.csv"
ANCHOR_COLUMNS = ["card_id", "video", "track_id", "frame", "x1", "y1", "x2", "y2"]
MATCH_IOU = 0.5
FRAME_TOLERANCE = 2  # source frames: a run at another frame rate may not have processed the anchor frame itself


def anchor_rows(card_id: str, video: str, track: pd.DataFrame, frames) -> list[dict]:
    """Anchor rows for one card: the boxes of `track` (the detections of one track id) at the given frames."""
    by_frame = track.drop_duplicates("frame").set_index("frame")
    rows = []
    for frame in dict.fromkeys(int(f) for f in frames):
        if frame in by_frame.index:
            box = by_frame.loc[frame]
            rows.append({"card_id": card_id, "video": video, "track_id": int(box.track_id), "frame": frame,
                         **{k: round(float(box[k]), 1) for k in ("x1", "y1", "x2", "y2")}})
    return rows


def load_anchors(path: Path = ANCHORS) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ANCHOR_COLUMNS)


def append_anchors(rows: list[dict], path: Path = ANCHORS) -> None:
    """Add (or replace, by card id) anchor rows in the anchors file."""
    if not rows:
        return
    new = pd.DataFrame(rows, columns=ANCHOR_COLUMNS)
    old = load_anchors(path)
    table = pd.concat([old[~old.card_id.isin(new.card_id)], new], ignore_index=True) if len(old) else new
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_card(anchors: pd.DataFrame, dets_by_frame: dict[int, pd.DataFrame]) -> int | None:
    """The current track id that covers a card's anchor boxes, or None.

    Each anchor votes for the tracks whose box overlaps it (IoU >= MATCH_IOU) on the nearest processed frame within
    FRAME_TOLERANCE. The track covering the most anchors wins (summed IoU breaks ties) and must cover at least half of
    them; otherwise the vehicle is not in the current detections any more.
    """
    votes: dict[int, list[float]] = {}
    for a in anchors.itertuples(index=False):
        box = np.array([a.x1, a.y1, a.x2, a.y2], float)
        for offset in sorted(range(-FRAME_TOLERANCE, FRAME_TOLERANCE + 1), key=abs):
            frame_dets = dets_by_frame.get(int(a.frame) + offset)
            if frame_dets is None:
                continue
            ious = pd.Series(pd.Series(_iou(box, frame_dets[["x1", "y1", "x2", "y2"]].to_numpy(float)),
                                       index=frame_dets.track_id.to_numpy()).groupby(level=0).max())
            for track_id, iou in ious[ious >= MATCH_IOU].items():
                votes.setdefault(int(track_id), []).append(float(iou))
            break  # only the nearest processed frame counts
    if not votes:
        return None
    track_id, ious = max(votes.items(), key=lambda item: (len(item[1]), sum(item[1])))
    return track_id if 2 * len(ious) >= len(anchors) else None


def align_track_labels(labels: pd.DataFrame, perception: Path, anchors: pd.DataFrame | None = None,
                       analysis: Path | None = None) -> pd.DataFrame:
    """The labels with `track_id` pointing into the current detections of each video - the track pieces of
    bikesafe.tracks (kinematics.parquet in `analysis`) when given, which differ from the tracker's ids where the
    identity split cut a track.

    Adds `labelled_track_id` (as written in the label file) and `alignment`: "same" (the labelled track still covers
    its card), "remapped" (the vehicle now has another track id), "lost" (no current track covers the card; callers
    drop these rows), "no_anchor" (a card rendered before anchors existed; used as written) or "no_perception" (the
    video has not been processed on this machine).
    """
    anchors = load_anchors() if anchors is None else anchors
    by_card = {card: group for card, group in anchors.groupby("card_id")} if len(anchors) else {}
    out = labels.copy()
    out["labelled_track_id"] = out.track_id
    out["alignment"] = "no_anchor"
    for video, rows in out.groupby("video"):
        path = perception / str(video) / "detections.parquet"
        pieces = analysis / str(video) / "kinematics.parquet" if analysis is not None else None
        if pieces is not None and pieces.exists():
            path = pieces
        if not path.exists():
            out.loc[rows.index, "alignment"] = "no_perception"
            continue
        anchored = [i for i in rows.index if out.at[i, "card_id"] in by_card]
        if not anchored:
            continue
        frames = anchors.loc[anchors.video == video, "frame"].astype(int)
        near = {f + o for f in frames for o in range(-FRAME_TOLERANCE, FRAME_TOLERANCE + 1)}
        dets = pd.read_parquet(path, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        dets = dets[(dets.track_id >= 0) & dets.frame.isin(near)]
        dets_by_frame = {int(f): g for f, g in dets.groupby("frame")}
        for i in anchored:
            track_id = match_card(by_card[out.at[i, "card_id"]], dets_by_frame)
            if track_id is None:
                out.at[i, "alignment"] = "lost"
            else:
                out.at[i, "alignment"] = "same" if track_id == out.at[i, "labelled_track_id"] else "remapped"
                out.at[i, "track_id"] = track_id
    return out


def alignment_summary(aligned: pd.DataFrame) -> str:
    counts = aligned.alignment.value_counts()
    return ", ".join(f"{counts.get(k, 0)} {k}" for k in ("same", "remapped", "lost", "no_anchor", "no_perception")
                     if counts.get(k, 0))
