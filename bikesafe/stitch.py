"""One id per physical vehicle: join the tracks the tracker split, and merge duplicate boxes of one vehicle.

Two tracker artefacts make one vehicle look like several:

1. Duplicates. The detector suppresses overlapping boxes class by class, so a pickup, van or SUV is often boxed twice
   at the same time, once as "car" and once as "truck" (or "bus"), and each box gets its own track. Two tracks that
   share frames with a mean overlap (IoU) of at least DUPLICATE_IOU, over at least half of the shorter track's frames,
   are one vehicle.
2. Breaks. At 12 frames per second a vehicle near the rider moves tens of pixels between frames; a missed detection,
   a short occlusion or a sudden change of box makes the tracker start a new id. The tracker only looks forward;
   afterwards both sides of a break are known and can be matched with more care:
   * the last detection of a track is extrapolated forward and the first detection of a later track backward to the
     middle of the gap (image velocity from a fit over the last / first detections, or from the optical flow the
     perception pass stored when a track has a single detection);
   * e  = distance between the two predictions and dv = difference of the two image velocities, both relative to the
     vehicle's box height; a pair is joined when  e + 0.5 dv <= MAX_SCORE  and also
   * the gap is at most MAX_GAP_S (the tracks never coexist), the class agrees (car and truck count as one: pickups
     and SUVs flip between them), the box size does not jump, and neither end lies on the left / right image border
     (a vehicle that leaves the frame there is gone, and the next one to appear there is usually a different vehicle);
   * no other vehicle that coexists with the chosen partner is nearly as good a match (the join would be a guess);
   * the one-to-one assignment over all ends and starts picks it.

The thresholds were set on visually checked pairs from the bike-ride demo clips so that a wrong join is rare, since it
merges two vehicles; a missed join leaves a vehicle split, as before.

vehicle_id is the smallest track id of a group of joined tracks; a track that is not joined keeps its own id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MAX_GAP_S = 2.0
MAX_SCORE = 0.45
VELOCITY_WEIGHT = 0.5
UNKNOWN_DV = 0.4  # velocity mismatch assumed when an end's image velocity is unknown (so only e <= 0.25 can join)
MAX_SIZE_RATIO = 2.0
AMBIGUITY_MARGIN = 0.15  # another vehicle scoring within this of the best makes the join a guess
EDGE_PX = 6
FIT_WINDOW = 6  # detections used for an end's image velocity
MIN_FIT_SPAN_S = 0.15
MIN_FLOW_SAMPLES = 9
CAR_LIKE = {2, 7}  # COCO car, truck
DUPLICATE_IOU = 0.7
DUPLICATE_SHARE = 0.5  # of the shorter track's frames
INFEASIBLE = 1e6
LINK_COLUMNS = ["A", "B", "kind", "gap_s", "e", "dv", "score", "iou"]


class _Groups:
    """Union-find over track ids; the root is the smallest id."""

    def __init__(self, ids) -> None:
        self.parent = {int(i): int(i) for i in ids}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def duplicates(dets: pd.DataFrame) -> pd.DataFrame:
    """Pairs of coexisting tracks that box the same vehicle: (A, B, shared frames, mean IoU), A the longer track."""
    d = dets[["frame", "track_id", "x1", "y1", "x2", "y2"]]
    pairs = d.merge(d, on="frame", suffixes=("_a", "_b"))
    pairs = pairs[pairs.track_id_a < pairs.track_id_b]
    if pairs.empty:
        return pd.DataFrame(columns=["A", "B", "shared", "iou"])
    ix1 = np.maximum(pairs.x1_a, pairs.x1_b)
    iy1 = np.maximum(pairs.y1_a, pairs.y1_b)
    ix2 = np.minimum(pairs.x2_a, pairs.x2_b)
    iy2 = np.minimum(pairs.y2_a, pairs.y2_b)
    inter = (ix2 - ix1).clip(lower=0) * (iy2 - iy1).clip(lower=0)
    area_a = (pairs.x2_a - pairs.x1_a) * (pairs.y2_a - pairs.y1_a)
    area_b = (pairs.x2_b - pairs.x1_b) * (pairs.y2_b - pairs.y1_b)
    pairs = pairs.assign(iou=inter / (area_a + area_b - inter))
    stats = pairs.groupby(["track_id_a", "track_id_b"]).agg(shared=("frame", "size"), iou=("iou", "mean")).reset_index()
    length = d.groupby("track_id").size()
    n_a, n_b = length.reindex(stats.track_id_a).to_numpy(), length.reindex(stats.track_id_b).to_numpy()
    keep = (stats.shared >= 2) & (stats.iou >= DUPLICATE_IOU) & (stats.shared >= DUPLICATE_SHARE * np.minimum(n_a, n_b))
    stats, n_a, n_b = stats[keep], n_a[keep.to_numpy()], n_b[keep.to_numpy()]
    longer_first = n_a >= n_b
    return pd.DataFrame({"A": np.where(longer_first, stats.track_id_a, stats.track_id_b),
                         "B": np.where(longer_first, stats.track_id_b, stats.track_id_a),
                         "shared": stats.shared.to_numpy(), "iou": stats.iou.to_numpy()})


def _fit_velocity(seg: pd.DataFrame) -> tuple[float, float]:
    times = seg.time_s.to_numpy(float)
    if len(seg) < 3 or np.ptp(times) < MIN_FIT_SPAN_S:
        return float("nan"), float("nan")
    t = times - times.mean()
    denom = float((t * t).sum())
    cu = ((seg.x1 + seg.x2) / 2).to_numpy(float)
    cv = ((seg.y1 + seg.y2) / 2).to_numpy(float)
    return float((t * (cu - cu.mean())).sum() / denom), float((t * (cv - cv.mean())).sum() / denom)


def endpoints(dets: pd.DataFrame, width: int, fps: float) -> pd.DataFrame:
    """First and last observation of every track: time, box centre and height, image velocity (px/s), class, and
    whether the box touches the left or right image border. `fps` is the rate of the processed frames, used to turn
    the stored per-step optical flow into a velocity."""
    has_flow = {"obj_dx", "obj_dy", "obj_n"} <= set(dets.columns)
    rows = []
    for tid, t in dets.sort_values(["track_id", "frame"]).groupby("track_id", sort=False):
        cls = int(t.class_id.mode().iloc[0])
        for end in (False, True):
            seg = t.iloc[-FIT_WINDOW:] if end else t.iloc[:FIT_WINDOW]
            row = seg.iloc[-1] if end else seg.iloc[0]
            vu, vv = _fit_velocity(seg)
            if not np.isfinite(vu) and has_flow and row.obj_n >= MIN_FLOW_SAMPLES and np.isfinite(row.obj_dx):
                vu, vv = float(row.obj_dx) * fps, float(row.obj_dy) * fps
            rows.append({"track_id": int(tid), "is_end": end, "t": float(row.time_s), "u": float((row.x1 + row.x2) / 2),
                         "v": float((row.y1 + row.y2) / 2), "h": float(max(row.y2 - row.y1, 1.0)), "vu": vu, "vv": vv,
                         "cls": cls, "side_edge": bool(row.x1 <= EDGE_PX or row.x2 >= width - EDGE_PX),
                         "t_first": float(t.time_s.iloc[0]), "t_last": float(t.time_s.iloc[-1])})
    return pd.DataFrame(rows)


def candidates(points: pd.DataFrame) -> pd.DataFrame:
    """Every (end of A, start of B) pair that passes the gates, with its score."""
    ends = points[points.is_end & ~points.side_edge].reset_index(drop=True)
    starts = points[~points.is_end & ~points.side_edge].sort_values("t").reset_index(drop=True)
    start_t = starts.t.to_numpy()
    rows = []
    for a in ends.itertuples(index=False):
        lo, hi = np.searchsorted(start_t, a.t, side="right"), np.searchsorted(start_t, a.t + MAX_GAP_S, side="right")
        for b in starts.iloc[lo:hi].itertuples(index=False):
            if b.track_id == a.track_id or not (a.cls == b.cls or {a.cls, b.cls} <= CAR_LIKE):
                continue
            if max(a.h, b.h) / min(a.h, b.h) > MAX_SIZE_RATIO:
                continue
            t_mid = (a.t + b.t) / 2
            known_a, known_b = np.isfinite(a.vu), np.isfinite(b.vu)
            pa = np.array([a.u, a.v]) + (np.array([a.vu, a.vv]) * (t_mid - a.t) if known_a else 0.0)
            pb = np.array([b.u, b.v]) - (np.array([b.vu, b.vv]) * (b.t - t_mid) if known_b else 0.0)
            h_ref = float(np.sqrt(a.h * b.h))
            e = float(np.hypot(*(pa - pb)) / h_ref)
            dv = float(np.hypot(a.vu - b.vu, a.vv - b.vv) / h_ref) if known_a and known_b else UNKNOWN_DV
            score = e + VELOCITY_WEIGHT * dv
            if score <= MAX_SCORE + AMBIGUITY_MARGIN:
                rows.append({"A": a.track_id, "B": b.track_id, "gap_s": b.t - a.t, "e": e, "dv": dv, "score": score,
                             "A_first": a.t_first, "A_last": a.t_last, "B_first": b.t_first, "B_last": b.t_last})
    columns = ["A", "B", "gap_s", "e", "dv", "score", "A_first", "A_last", "B_first", "B_last"]
    return pd.DataFrame(rows, columns=columns)


def _coexist(first1: float, last1: float, first2: float, last2: float) -> bool:
    return first1 <= last2 and first2 <= last1


def _unambiguous(cand: pd.DataFrame, same_vehicle) -> pd.Series:
    """False for a pair when another vehicle that coexists with its partner scores within AMBIGUITY_MARGIN of it,
    from either side: two different vehicles fit the break about equally well. Duplicate boxes of the partner are
    the same vehicle and do not count."""
    ok = pd.Series(True, index=cand.index)
    for key, other, first, last in (("A", "B", "B_first", "B_last"), ("B", "A", "A_first", "A_last")):
        for _, group in cand.groupby(key):
            if len(group) < 2:
                continue
            for i, row in group.iterrows():
                rivals = group[(group.index != i) & (group.score <= row.score + AMBIGUITY_MARGIN)]
                for _, rival in rivals.iterrows():
                    if (not same_vehicle(row[other], rival[other])
                            and _coexist(row[first], row[last], rival[first], rival[last])):
                        ok[i] = False
                        break
    return ok


def stitch(dets: pd.DataFrame, width: int, fps: float) -> pd.DataFrame:
    """Links between tracks of one video: kind "duplicate" (two boxes on one vehicle at the same time) and kind
    "gap" (A ended, B continues it), the latter one-to-one, best score first. `dets` needs frame, time_s, track_id,
    class_id, x1, y1, x2, y2 and, when the perception pass stored them, obj_dx, obj_dy, obj_n."""
    dets = dets[dets.track_id >= 0]
    if dets.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    dup = duplicates(dets)
    groups = _Groups(dets.track_id.unique())
    for a, b in zip(dup.A, dup.B):
        groups.union(a, b)
    links = [dup.assign(kind="duplicate", gap_s=0.0, e=np.nan, dv=np.nan, score=np.nan)[LINK_COLUMNS]]

    points = endpoints(dets, width, fps)
    cand = candidates(points)
    if not cand.empty:
        # a break joins two vehicles only if the first is gone before the second appears (duplicates included)
        root = points.track_id.map(lambda t: groups.find(int(t)))
        first, last = points.t_first.groupby(root).min(), points.t_last.groupby(root).max()
        ra, rb = cand.A.map(lambda t: groups.find(int(t))), cand.B.map(lambda t: groups.find(int(t)))
        cand = cand[(ra != rb).to_numpy() & (last.reindex(ra).to_numpy() < first.reindex(rb).to_numpy())]
    if not cand.empty:
        cand = cand[_unambiguous(cand, lambda x, y: groups.find(int(x)) == groups.find(int(y)))
                    & (cand.score <= MAX_SCORE)]
    if not cand.empty:
        a_ids, b_ids = sorted(cand.A.unique()), sorted(cand.B.unique())
        a_index, b_index = {a: i for i, a in enumerate(a_ids)}, {b: j for j, b in enumerate(b_ids)}
        cost = np.full((len(a_ids), len(b_ids)), INFEASIBLE)
        for row in cand.itertuples(index=False):
            cost[a_index[row.A], b_index[row.B]] = row.score
        rows, cols = linear_sum_assignment(cost)
        chosen = {(a_ids[i], b_ids[j]) for i, j in zip(rows, cols) if cost[i, j] < INFEASIBLE}
        gap = cand[[(a, b) in chosen for a, b in zip(cand.A, cand.B)]]
        links.append(gap.assign(kind="gap", iou=np.nan)[LINK_COLUMNS])
    out = pd.concat([frame for frame in links if len(frame)], ignore_index=True) if any(len(f) for f in links) else None
    if out is None:
        return pd.DataFrame(columns=LINK_COLUMNS)
    return out.astype({"A": int, "B": int}).sort_values(["kind", "A"]).reset_index(drop=True)


def vehicle_ids(track_ids, links: pd.DataFrame) -> dict[int, int]:
    """track_id -> vehicle_id: the smallest track id among the tracks joined into one vehicle."""
    groups = _Groups(track_ids)
    for a, b in zip(links.A.astype(int), links.B.astype(int)):
        groups.union(a, b)
    return {t: groups.find(t) for t in list(groups.parent)}
