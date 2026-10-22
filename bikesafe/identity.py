"""Split tracks where the tracker handed one id to another vehicle ("identity transfers").

BoT-SORT at the pipeline's ~12 fps associates boxes by overlap with a predicted box. At junctions, where boxes are
small, close together and cross one another, and after a vehicle is hidden for a moment (the tracker keeps a lost
track for 2 s), it sometimes continues a track on a different vehicle. A track that covers two vehicles mixes their
motion, so at least one of them gets the wrong relation, and the overlay shows one box hopping between cars. On the
approved reference sequence most of the pipeline's identity errors at its frame rate were of this kind.

Each tracked box gets an appearance descriptor - joint hue / saturation / value histograms of the upper and lower
half of its inner 70% (the windows and the body of a vehicle) - from one pass over the video (appearance_descriptors,
cached as appearance.npz in the perception folder). Every step of a track is scored on two independent signs of a
transfer (step_scores):

* appearance: Bhattacharyya distance between the box and the track's recent boxes (the mean descriptor of up to
  REFERENCE_BOXES boxes before it); a transfer changes the colours, a vehicle that turns or is lit differently
  changes them only gradually;
* motion: how far the box is from where the track's recent image motion puts it, in box heights, plus the change of
  its size against the recent trend (|log size ratio|); a transfer jumps, a vehicle moves smoothly.

A track is split where both are high, or where one is extreme (split_switches). The pieces get new track ids and keep
the tracker's id in `raw_track_id`. Thresholds were set on the reference sequence (scripts/evaluate_identity.py).

Evaluation, and why it is off by default (`bikesafe.tracks --identity-split` turns it on). On the 300-frame reference
sequence it cut 6 times, 5 of them real transfers, and raised IDF1 from 69.6% to 71.9% (transfers 68 -> 55). On the
rides it cut far more often (1,868 times in 3,717 tracks of ride 006) and a check of 24 random cuts found 14 real, 7
wrong and 3 undecidable - small, distant and night-time boxes, whose colours and positions jump without a change of
vehicle. Over 5 seeds of the relation model (leave-one-video-out, near labels) it lowered accuracy from 0.735 to
0.718 and macro-F1 from 0.626 to 0.594: the shorter pieces carry less motion evidence than whole tracks, even when
some of those are mixed. A better appearance model (a vehicle re-identification network) would be needed before it
helps.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import open_video

H_BINS, S_BINS, V_BINS = 8, 3, 3
DESCRIPTOR_DIM = 2 * H_BINS * S_BINS * V_BINS
# hue bins are centred on 0 so red, which wraps round 0/180 in OpenCV's hue, falls in one bin and not two
HUE_SHIFT = int(round(180 / H_BINS / 2))
DESCRIPTOR_VERSION = 2  # stored with the cache; bump when the descriptor changes
INNER = 0.15  # each side of the box left out (background, neighbouring vehicles)
MIN_DESCRIPTOR_H = 14  # px; smaller boxes have too few pixels for a stable histogram
REFERENCE_BOXES = 5
# Between consecutive processed frames: split where appearance >= APPEARANCE_SPLIT and motion >= MOTION_SPLIT, or
# either alone is extreme. Across a gap (the tracker lost the vehicle for a frame or more and picked a box up again,
# where most real transfers happen) the bar is lower: GAP_APPEARANCE and GAP_MOTION together. On the reference
# sequence the gap rule caught 5 of 6 transfers across gaps with one false split; the consecutive rule, which must
# not cut vehicles that turn or pass under shade, made no false split.
APPEARANCE_SPLIT, MOTION_SPLIT = 0.25, 0.8
APPEARANCE_ALONE, MOTION_ALONE = 0.40, 3.0
GAP_APPEARANCE, GAP_MOTION = 0.13, 0.30
CACHE = "appearance.npz"


def _smooth(hist: np.ndarray) -> np.ndarray:
    """Spread each bin a quarter into its neighbours (hue circularly, saturation and value with the edges held), so a
    colour on a bin boundary does not flip between bins from one frame to the next."""
    out = 0.25 * np.roll(hist, 1, axis=0) + 0.5 * hist + 0.25 * np.roll(hist, -1, axis=0)
    for axis in (1, 2):
        lo = np.concatenate([np.take(out, [0], axis=axis), np.take(out, range(out.shape[axis] - 1), axis=axis)], axis=axis)
        hi = np.concatenate([np.take(out, range(1, out.shape[axis]), axis=axis), np.take(out, [-1], axis=axis)], axis=axis)
        out = 0.25 * lo + 0.5 * out + 0.25 * hi
    return out


def descriptor(crop: np.ndarray) -> np.ndarray:
    """Normalised, smoothed joint HSV histograms of the upper and lower half of a BGR crop (DESCRIPTOR_DIM values)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + HUE_SHIFT) % 180).astype(np.uint8)
    half = hsv.shape[0] // 2
    out = []
    for part in (hsv[:half], hsv[half:]):
        hist = cv2.calcHist([part], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
        hist = _smooth(hist.reshape(H_BINS, S_BINS, V_BINS)).ravel()
        out.append(hist / max(hist.sum(), 1e-9))
    return np.concatenate(out).astype(np.float32)


def appearance_descriptors(video: Path, dets: pd.DataFrame, cache: Path | None = None) -> np.ndarray:
    """(len(dets), DESCRIPTOR_DIM) descriptors in the row order of `dets` (NaN for boxes too small), from one
    sequential pass over the frames that hold tracked boxes. With `cache`, read from / written to that .npz, keyed by
    (frame, track_id, box) so it stays valid for the same detections."""
    keys = np.column_stack([dets.frame.to_numpy(np.int64), dets.track_id.to_numpy(np.int64),
                            np.round(dets[["x1", "y1", "x2", "y2"]].to_numpy(float)).astype(np.int64)])
    if cache is not None and cache.exists():
        stored = np.load(cache)
        version = int(stored["version"]) if "version" in stored else 1
        if version == DESCRIPTOR_VERSION and stored["keys"].shape == keys.shape and np.array_equal(stored["keys"], keys):
            return stored["descriptors"].astype(np.float32)
    out = np.full((len(dets), DESCRIPTOR_DIM), np.nan, np.float32)
    by_frame = {int(f): idx for f, idx in dets.groupby("frame").indices.items()}
    frames = sorted(by_frame)
    if frames:
        cap = open_video(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        position = frames[0]
        boxes = dets[["x1", "y1", "x2", "y2"]].to_numpy(float)
        for frame in frames:
            while position < frame:
                cap.grab()
                position += 1
            ok, image = cap.read()
            position += 1
            if not ok:
                break
            for i in by_frame[frame]:
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                if h < MIN_DESCRIPTOR_H or w < 8:
                    continue
                crop = image[int(max(0, y1 + INNER * h)): int(y2 - INNER * h), int(max(0, x1 + INNER * w)): int(x2 - INNER * w)]
                if crop.size:
                    out[i] = descriptor(crop)
        cap.release()
    if cache is not None:
        np.savez_compressed(cache, keys=keys, descriptors=out.astype(np.float16), version=DESCRIPTOR_VERSION)
    return out


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """Distance in [0, 1] between two descriptors (each half a normalised histogram); NaN if either is missing."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    half = DESCRIPTOR_DIM // 2
    d = [np.sqrt(max(0.0, 1.0 - float(np.sqrt(a[s] * b[s]).sum()))) for s in (slice(0, half), slice(half, None))]
    return float(np.mean(d))


def step_scores(dets: pd.DataFrame, desc: np.ndarray) -> pd.DataFrame:
    """One row per step between consecutive boxes of a track: appearance and motion discontinuity (see module doc)."""
    rows = []
    boxes = dets[["x1", "y1", "x2", "y2"]].to_numpy(float)
    frames = dets.frame.to_numpy(np.int64)
    # the processed-frame step (perception stride): the most common gap between a track's boxes
    gaps = np.concatenate([np.diff(np.sort(frames[i])) for i in dets.groupby("track_id").indices.values()] or [[1]])
    gaps = gaps[gaps > 0]
    stride = int(np.bincount(gaps).argmax()) if len(gaps) else 1
    for track_id, idx in dets.groupby("track_id").indices.items():
        idx = idx[np.argsort(frames[idx])]
        centres = np.column_stack([(boxes[idx, 0] + boxes[idx, 2]) / 2, (boxes[idx, 1] + boxes[idx, 3]) / 2])
        heights = np.maximum(boxes[idx, 3] - boxes[idx, 1], 1.0)
        f = frames[idx]
        start = 0  # first box of the current piece: after a cut, the boxes before it are another vehicle
        for k in range(1, len(idx)):
            gap = max(1, int(f[k] - f[k - 1]))
            lo = max(start, k - REFERENCE_BOXES)
            ref = desc[idx[lo:k]]
            ref = ref[np.isfinite(ref).all(axis=1)]
            appearance = bhattacharyya(desc[idx[k]], ref.mean(axis=0)) if len(ref) else float("nan")
            # image velocity and size trend over the last few steps, per frame
            j = max(start, k - 3)
            span = max(1, int(f[k - 1] - f[j]))
            velocity = (centres[k - 1] - centres[j]) / span if k - 1 > j else np.zeros(2)
            growth = np.log(heights[k - 1] / heights[j]) / span if k - 1 > j else 0.0
            predicted = centres[k - 1] + velocity * gap
            scale = max(heights[k], heights[k - 1])
            motion = float(np.hypot(*(centres[k] - predicted)) / scale
                           + abs(np.log(heights[k] / heights[k - 1]) - growth * gap))
            cut = bool(is_split(appearance, motion, gap > stride))
            if cut:
                start = k
            rows.append((int(track_id), int(f[k]), int(f[k - 1]), gap, gap > stride, appearance, motion, cut, int(idx[k])))
    return pd.DataFrame(rows, columns=["track_id", "frame", "prev_frame", "gap", "across_gap", "appearance", "motion",
                                       "split", "row"])


def is_split(appearance: float, motion: float, across_gap: bool) -> bool:
    a = 0.0 if not np.isfinite(appearance) else appearance
    if (a >= APPEARANCE_SPLIT and motion >= MOTION_SPLIT) or a >= APPEARANCE_ALONE or motion >= MOTION_ALONE:
        return True
    return bool(across_gap and a >= GAP_APPEARANCE and motion >= GAP_MOTION)


def remap_ids(table: pd.DataFrame, id_map: pd.DataFrame, frame: str = "frame", track: str = "track_id") -> pd.DataFrame:
    """A table keyed by the tracker's ids (scene3d, vehicle context, CLIP crops: `frame`, `track`) with the ids of the
    track pieces instead. `id_map` has frame, raw_track_id, track_id; rows without a match keep their id."""
    if table.empty or id_map.empty:
        return table
    m = id_map.drop_duplicates(["frame", "raw_track_id"]).set_index(["frame", "raw_track_id"]).track_id
    keys = pd.MultiIndex.from_arrays([table[frame].to_numpy(np.int64), table[track].to_numpy(np.int64)])
    mapped = m.reindex(keys).to_numpy()
    return table.assign(**{track: np.where(pd.isna(mapped), table[track].to_numpy(), mapped).astype(np.int64)})


def split_switches(dets: pd.DataFrame, desc: np.ndarray) -> pd.DataFrame:
    """`dets` with each track cut at its transfers: pieces after a cut get new track ids, the tracker's id is kept as
    `raw_track_id`. `desc` is aligned with the rows of `dets`. Untracked boxes (track_id < 0) are left unchanged."""
    out = dets.copy()
    out["raw_track_id"] = out.track_id
    tracked = np.flatnonzero((out.track_id >= 0).to_numpy())
    if not len(tracked):
        return out
    sub = out.iloc[tracked].reset_index(drop=True)
    cuts = step_scores(sub, desc[tracked])
    cuts = cuts[cuts.split]
    if cuts.empty:
        return out
    track = out.track_id.to_numpy().copy()
    frame = out.frame.to_numpy()
    sub_track = sub.track_id.to_numpy()
    next_id = int(track.max()) + 1
    for track_id, group in cuts.groupby("track_id"):
        members = tracked[sub_track == track_id]
        for cut in sorted(group.frame):  # every box from the cut on goes to a new piece
            track[members[frame[members] >= cut]] = next_id
            next_id += 1
    out["track_id"] = track
    return out
