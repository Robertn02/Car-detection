"""Review and correct the pipeline's annotations in the browser: the human-in-the-loop part of the pipeline.

    python -m bikesafe.review [--reviewer NAME] [--corpus results/corpus] [--port 8765]

then open http://127.0.0.1:8765. Three queues, each pre-filled with what the models said:

* **Vehicles** - the relation of a vehicle to the rider, shown as a looping clip of its track (magenta box) and a
  zoom. Keys 1-5 pick the relation, Enter accepts the model's answer, U marks it unclear, N not a vehicle (a delivery
  robot, a scooter, a bin), S skips, Z undoes.
* **Signs** - each physical sign's largest view with its type and text (bikesafe.signs); pick the type, correct the
  text, or mark it "not a traffic sign".
* **Plates** - each plate crop with the text read (bikesafe.plates); correct it or mark it unreadable.

The queues put first what is most likely wrong, so a reviewer's time goes where the models fail: low model confidence,
and conflicts between the model and the physics - "parked" while sitting in the rider's path or moving with the
rider, "oncoming" while its rear is seen, model and rule baseline disagreeing. In "verify" mode the vehicle queue
shows the existing labels instead (most of them came from AI-assisted review), those the model disagrees with first.

Every decision goes to data/typology (or --labels) - track_labels.csv with track_anchors.csv (so labels survive a
perception re-run), sign_labels.csv, plate_labels.csv - and to review_log.csv together with what the model had said.
The Progress tab turns that log into how often each model agreed with the reviewer, per class; "Retrain" runs
bikesafe.train and bikesafe.exposure so the next queue comes from the corrected models. This is the loop the pipeline
learns from: model proposes, reviewer confirms or corrects, models retrain on the corrections.

    python -m bikesafe.review --export site/review      # a copy that needs no server (GitHub Pages)
    python -m bikesafe.review --import decisions.json   # the decisions made on it

The exported copy has the first --per-queue items of each queue with their clips and crops; its decisions stay in the
viewer's browser until downloaded from its Progress tab.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import RELATIONS, ROOT, VIDEOS, VideoOut, read_json
from bikesafe.labels import anchor_rows, append_anchors, load_anchors
from bikesafe.render import fill_box_gaps

LABELS_DIR = ROOT / "data" / "typology"
REVIEW_LOG_COLUMNS = ["time", "reviewer", "kind", "video", "item", "model", "model_conf", "human", "text", "model_text"]
SIGN_LABEL_COLUMNS = ["video", "object_id", "frame", "x1", "y1", "x2", "y2", "label", "text", "reviewer", "time"]
PLATE_LABEL_COLUMNS = ["video", "track_id", "frame", "x1", "y1", "x2", "y2", "text", "reviewer", "time"]
QUEUE_SIZE = 40
MIN_PLATE_W, MAX_PLATE_W = 40, 300  # px; narrower plates are unreadable to a reviewer too, wider boxes are not plates
MAX_Z_M = 15.0  # vehicles reviewed: those that came this close (where relations are validated and matter)
MIN_BOX_H = 40
SPREAD_S = 4.0  # at most one queued vehicle per ride and this many seconds, so a queue is not one traffic jam
CLIP_WIDTH = 960  # px; vehicle clips
CLIP_CRF = 25
CLIP_MAX_S = 8.0  # a longer track plays faster, so every clip loops in about this long
CLIP_PAD_S = 0.5  # shown before the track starts and after it ends
SIGN_TYPES_UI = ["stop", "yield", "speed_limit", "do_not_enter", "one_way", "no_turn", "parking", "bike", "pedestrian",
                 "warning", "street_name", "guide", "lane_use", "bus_stop", "roadwork", "other_sign", "not_a_sign"]


# --------------------------------------------------------------------------------------------------------------------
# queues


def vehicle_priority(tracks: pd.DataFrame) -> pd.DataFrame:
    """Score how likely each track's predicted relation is wrong, with the reasons, highest first.

    Uncertainty is 1 minus the model's probability for its answer. Physics conflicts add to it: a vehicle called
    parked that spent time in the rider's path or whose flow says it moves with or against the rider, a vehicle called
    oncoming whose crop looks like its rear, and disagreement with the transparent rule baseline.
    """
    t = tracks.copy()
    proba = [f"proba_{r}" for r in RELATIONS if f"proba_{r}" in t]
    if proba:
        conf = t[proba].max(axis=1)
    elif "relation_confidence" in t:
        conf = t["relation_confidence"]
    else:
        conf = pd.Series(0.5, index=t.index)
    t["model_conf"] = conf.fillna(0.5)
    score = (1 - t.model_conf).to_numpy(float)
    reasons = [[] for _ in range(len(t))]

    def flag(mask: pd.Series, weight: float, text: str) -> None:
        nonlocal score
        hits = mask.fillna(False).astype(bool).to_numpy()
        score = score + weight * hits
        for i in np.flatnonzero(hits):
            reasons[i].append(text)

    get = lambda col: t[col] if col in t else pd.Series(np.nan, index=t.index)  # noqa: E731
    parked = t.vehicle_relation == "parked"
    flag(parked & (get("frac_in_path") >= 0.3), 0.6, "parked but in the rider's path")
    flag(parked & (get("frac_closing_with") >= 0.5), 0.4, "parked but moving with the rider")
    flag(parked & (get("frac_closing_against") >= 0.6), 0.4, "parked but closing fast")
    flag((t.vehicle_relation == "oncoming") & (get("view_rear") > get("view_front") + 0.1), 0.5,
         "oncoming but its rear is seen")
    flag(get("rule_relation") != t.vehicle_relation, 0.2, "rules disagree")
    flag(get("z_min") < 8, 0.2, "close to the rider")
    flag(get("not_vehicle") >= 0.5, 0.5, "may not be a vehicle")
    t["score"] = score
    t["reasons"] = ["; ".join(r) for r in reasons]
    return t.sort_values("score", ascending=False)


def spread(items: pd.DataFrame, n: int, gap_s: float = SPREAD_S) -> pd.DataFrame:
    """The first n rows (in order) such that no two picked rows of one video start within gap_s of each other."""
    picked, taken = [], {}
    for idx, row in items.iterrows():
        times = taken.setdefault(row.video, [])
        if any(abs(row.t_start - t) < gap_s for t in times):
            continue
        times.append(row.t_start)
        picked.append(idx)
        if len(picked) >= n:
            break
    return items.loc[picked]


def next_card_id(video: str, labels: pd.DataFrame, anchors: pd.DataFrame) -> str:
    """A card id (T<ride>_<n>) not used by any label or anchor of the video."""
    cards = pd.concat([labels.loc[labels.video == video, "card_id"], anchors.loc[anchors.video == video, "card_id"]])
    numbers = cards.str.split("_").str[1].astype(int) if len(cards) else pd.Series([0])
    return f"T{video.rsplit('_', 1)[-1]}_{int(numbers.max()) + 1:03d}"


def agreement(log: pd.DataFrame) -> dict:
    """Per kind: decisions, share where the reviewer kept the model's answer, and for vehicles and signs the counts of
    (model answer, reviewer answer) - the confusion of the model against the reviewer."""
    out = {}
    for kind, rows in log[log.human.notna() & (log.human != "skip")].groupby("kind"):
        decided = rows[rows.human != "unclear"]
        agree = (decided.human == decided.model)
        if kind == "plate":
            agree = decided.text.fillna("") == decided.model_text.fillna("")
        out[str(kind)] = {"decisions": int(len(rows)), "unclear": int((rows.human == "unclear").sum()),
                          "model_kept": round(float(agree.mean()), 3) if len(decided) else None,
                          "by_reviewer": rows.reviewer.value_counts().to_dict(),
                          "confusion": decided.groupby(["model", "human"]).size().rename("n").reset_index()
                          .to_dict("records") if kind != "plate" else []}
    return out


# --------------------------------------------------------------------------------------------------------------------
# images


def _read(cap: cv2.VideoCapture, frame: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    return image if ok else None


def track_clip(video: Path, dets: pd.DataFrame, track_id: int, out: Path) -> bytes:
    """The track as a short H.264 clip at the video's frame rate, its box in magenta and the other vehicles' in grey,
    interpolated between the frames the detector ran on; a track longer than CLIP_MAX_S plays faster (the speed-up is
    shown). Written to `out` and returned."""
    t = dets[dets.track_id == track_id]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    pad = int(round(CLIP_PAD_S * fps))
    start, end = max(0, int(t.frame.min()) - pad), int(t.frame.max()) + pad
    speed = max(1, int(np.ceil((end - start + 1) / fps / CLIP_MAX_S)))
    window = dets[dets.frame.between(start - 2 * pad, end + 2 * pad)]
    detected = np.unique(window.frame)
    stride = int(np.median(np.diff(detected))) if len(detected) > 1 else 1
    boxes = fill_box_gaps(window, stride, step=1)
    boxes = boxes.assign(own=boxes.track_id == track_id).sort_values("own").groupby("frame")  # its box drawn last
    present = set(boxes.groups)
    scale = CLIP_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    size = (CLIP_WIDTH, int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)) // 2 * 2)
    part = out.with_name(f"{out.stem}.{threading.get_ident()}.part.mp4")
    writer = VideoOut(part, fps, size, crf=CLIP_CRF)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame in range(start, end + 1):
        if not cap.grab():
            break
        if (frame - start) % speed:
            continue
        image = cv2.resize(cap.retrieve()[1], size, interpolation=cv2.INTER_AREA)
        for o in boxes.get_group(frame).itertuples() if frame in present else []:
            x1, y1, x2, y2 = (np.array([o.x1, o.y1, o.x2, o.y2]) * scale).astype(int)
            if o.own:
                cv2.rectangle(image, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 0, 255), 4)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), 1)
        label = f"t={frame / fps:.1f}s" + (f"  {speed}x speed" if speed > 1 else "")
        cv2.putText(image, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    cap.release()
    os.replace(part, out)
    return out.read_bytes()


def track_zoom(video: Path, dets: pd.DataFrame, track_id: int) -> bytes:
    """The track's largest view, cropped with some margin."""
    t = dets[dets.track_id == track_id]
    box = t.loc[(t.y2 - t.y1).idxmax()]
    cap = cv2.VideoCapture(str(video))
    image = _read(cap, int(box.frame))
    cap.release()
    if image is None:
        return b""
    w, h = box.x2 - box.x1, box.y2 - box.y1
    x1, y1 = int(max(0, box.x1 - 0.4 * w)), int(max(0, box.y1 - 0.4 * h))
    x2, y2 = int(min(image.shape[1], box.x2 + 0.4 * w)), int(min(image.shape[0], box.y2 + 0.4 * h))
    crop = image[y1:y2, x1:x2]
    scale = 300 / max(crop.shape[0], 1)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 300))
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def context_crop(video: Path, frame: int, box: tuple[float, float, float, float], margin: float, height: int) -> bytes:
    """A crop around a box in one frame, with the box drawn, `height` px tall."""
    cap = cv2.VideoCapture(str(video))
    image = _read(cap, frame)
    cap.release()
    if image is None:
        return b""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - margin * w)), int(max(0, y1 - margin * h))
    cx2, cy2 = int(min(image.shape[1], x2 + margin * w)), int(min(image.shape[0], y2 + margin * h))
    crop = image[cy1:cy2, cx1:cx2]
    scale = height / max(crop.shape[0], 1)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)
    # the box is drawn after enlarging, just outside the object, so it never covers what is to be read
    bx1, by1 = int((x1 - cx1) * scale) - 3, int((y1 - cy1) * scale) - 3
    bx2, by2 = int((x2 - cx1) * scale) + 3, int((y2 - cy1) * scale) + 3
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


# --------------------------------------------------------------------------------------------------------------------
# state


class Store:
    """The tables the queues come from and the label files the decisions go to."""

    def __init__(self, analysis: Path, perception: Path, corpus: Path, videos: Path, labels_dir: Path = LABELS_DIR,
                 cache: Path = ROOT / "work" / "review_cache") -> None:
        self.analysis, self.perception, self.corpus, self.videos = analysis, perception, corpus, videos
        self.labels_dir, self.cache = labels_dir, cache
        self.lock = threading.Lock()
        self.clip_lock = threading.Lock()
        self.retrain: dict = {"state": "idle", "log": ""}
        self.reload()

    def reload(self) -> None:
        frames = []
        for path in sorted(self.corpus.glob("*/tracks_typed.csv")):
            frames.append(pd.read_csv(path))
        self.tracks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._dets: dict[str, pd.DataFrame] = {}

    # files
    def _csv(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        path = self.labels_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)

    def log(self, entry: dict) -> None:
        log = self._csv("review_log.csv", REVIEW_LOG_COLUMNS)
        row = {"time": datetime.now().isoformat(timespec="seconds"), **entry}
        pd.concat([log, pd.DataFrame([row], columns=REVIEW_LOG_COLUMNS)], ignore_index=True).to_csv(
            self.labels_dir / "review_log.csv", index=False)

    def dets(self, video: str) -> pd.DataFrame:
        """Boxes by track id as bikesafe.tracks made them (identity transfers cut), the tracker's as a fallback."""
        if video not in self._dets:
            pieces = self.analysis / video / "kinematics.parquet"
            path = pieces if pieces.exists() else self.perception / video / "detections.parquet"
            self._dets[video] = pd.read_parquet(path, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        return self._dets[video]

    def video_path(self, video: str) -> Path:
        meta = read_json(self.perception / video / "meta.json")
        path = Path(meta["video"])
        return path if path.exists() else self.videos / f"{video}.mp4"

    def known_video(self, video: str) -> bool:
        return (self.perception / video / "meta.json").exists()

    # queues
    def vehicle_queue(self, mode: str) -> list[dict]:
        if self.tracks.empty:
            return []
        labels = self._csv("track_labels.csv", ["card_id", "video", "track_id", "label", "labeler", "notes"])
        t = self.tracks
        # one row per vehicle: its longest track carries the label
        t = t.sort_values("n_det", ascending=False).drop_duplicates(["video", "vehicle_id"])
        t = t[(t.z_min <= MAX_Z_M) & (t.box_h_max >= MIN_BOX_H) & (t.duration_s >= 1.0)]
        keys = set(zip(labels.video, labels.track_id))
        labelled = np.array([(v, k) in keys for v, k in zip(t.video, t.track_id)], bool)
        if mode == "verify":
            t = t[labelled].merge(labels[["video", "track_id", "label", "labeler"]], on=["video", "track_id"])
            t = t[t.label.isin(RELATIONS)].copy()  # one block: the many proba columns fragment it
            p = np.array([row.get(f"proba_{row.label}", np.nan) for _, row in t.iterrows()], float)
            t = t.assign(score=1 - np.nan_to_num(p, nan=0.5), reasons=[f"labelled {lab} by {who}" for lab, who in
                                                                         zip(t.label, t.labeler)])
            t = t.sort_values("score", ascending=False)
        else:
            t = vehicle_priority(t[~labelled])
        rows = spread(t, QUEUE_SIZE)
        items = []
        for row in rows.itertuples():
            proba = {r: round(float(getattr(row, f"proba_{r}", np.nan)), 3) for r in RELATIONS
                     if hasattr(row, f"proba_{r}")}
            items.append({"kind": "vehicle", "video": row.video, "item": int(row.track_id),
                          "model": row.vehicle_relation, "proba": proba, "rule": getattr(row, "rule_relation", None),
                          "current": getattr(row, "label", None), "reasons": row.reasons,
                          "t": round(float(row.t_start), 1), "z_min": round(float(row.z_min), 1),
                          "x": round(float(row.x_at_min_z), 1), "duration": round(float(row.duration_s), 1)})
        return items

    def sign_queue(self) -> list[dict]:
        done = self._csv("sign_labels.csv", SIGN_LABEL_COLUMNS)
        keys = set(zip(done.video, done.object_id))
        items = []
        for path in sorted(self.analysis.glob("*/infrastructure.parquet")):
            video = path.parent.name
            if not (path.parent / "sign_crops.parquet").exists():
                continue
            obj = pd.read_parquet(path)
            signs = obj[obj.kind.isin(["sign", "rejected_sign"])].assign(h=lambda o: o.y2 - o.y1)
            for object_id, g in signs.groupby("object_id"):
                if (video, int(object_id)) in keys:
                    continue
                best = g.sort_values("h").iloc[-1]
                if best.h < 20:
                    continue
                conf = float(best.type_conf) if pd.notna(best.type_conf) else 0.0
                category = best.category if best.kind == "sign" else "not_a_sign"
                items.append({"kind": "sign", "video": video, "item": int(object_id), "row": int(best.name),
                              "model": category, "model_conf": round(conf, 3), "text": str(best.sign_text or ""),
                              "h": int(best.h), "t": round(float(best.time_s), 1),
                              "score": (1 - conf) + 0.3 * (category in ("other_sign", "not_a_sign")) + best.h / 1000})
        return sorted(items, key=lambda i: -i["score"])[:QUEUE_SIZE]

    def plate_queue(self) -> list[dict]:
        done = self._csv("plate_labels.csv", PLATE_LABEL_COLUMNS)
        keys = set(zip(done.video, done.track_id))
        items = []
        for path in sorted(self.analysis.glob("*/plates.parquet")):
            video = path.parent.name
            plates = pd.read_parquet(path)
            if "plate_frame" not in plates:
                continue  # read before plate boxes were kept; re-run bikesafe.plates --overwrite
            # plates a person can read (at least MIN_PLATE_W px wide) and that are shaped like plates: the detector
            # also boxes lettered panels on trucks, which are far wider and squarer
            w, h = plates.plate_x2 - plates.plate_x1, plates.plate_y2 - plates.plate_y1
            plates = plates[plates.plate_frame.notna() & w.between(MIN_PLATE_W, MAX_PLATE_W) & (w / h).between(1.3, 4.5)]
            for row in plates.itertuples():
                if (video, int(row.track_id)) in keys:
                    continue
                read = row.plate_text if isinstance(row.plate_text, str) else ""
                conf = float(row.plate_conf) if pd.notna(row.plate_conf) else 0.0
                # most useful to check: readings that are not certain (one read, or reads that disagree), then
                # readable-sized plates the reader gave up on
                uncertain = read != "" and (row.agreeing_reads < 2 or conf < 0.95)
                items.append({"kind": "plate", "video": video, "item": int(row.track_id), "model": read,
                              "model_conf": round(conf, 3), "reads": str(row.all_reads or ""),
                              "score": 1.0 * uncertain + 0.5 * (read == "") + min(row.plate_x2 - row.plate_x1, 150) / 1000})
        return sorted(items, key=lambda i: -i["score"])[:QUEUE_SIZE]

    # images and clips (cached on disk)
    def image(self, kind: str, video: str, item: int, row: int | None = None) -> bytes:
        path = self.cache / video / f"{kind}_{item}.{'mp4' if kind == 'clip' else 'jpg'}"
        if path.exists():
            return path.read_bytes()
        if kind == "clip":
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.clip_lock:  # the page asks for a clip while prefetching it: encode it once
                return path.read_bytes() if path.exists() else track_clip(self.video_path(video), self.dets(video),
                                                                          item, path)
        if kind == "zoom":
            data = track_zoom(self.video_path(video), self.dets(video), item)
        elif kind == "sign":
            crops = pd.read_parquet(self.analysis / video / "sign_crops.parquet")
            hit = crops[crops.row == row]
            data = b""
            if len(hit):
                im = cv2.imdecode(np.frombuffer(bytes(hit.jpeg.iloc[0]), np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    return b""
                scale = 260 / max(im.shape[:2])
                im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))),
                                interpolation=cv2.INTER_CUBIC)
                data = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        elif kind == "signctx":
            obj = pd.read_parquet(self.analysis / video / "infrastructure.parquet")
            o = obj.loc[row]
            data = context_crop(self.video_path(video), int(o.frame), (o.x1, o.y1, o.x2, o.y2), 6.0, 320)
        elif kind == "plate":
            p = pd.read_parquet(self.analysis / video / "plates.parquet").set_index("track_id").loc[item]
            data = context_crop(self.video_path(video), int(p.plate_frame),
                                (p.plate_x1, p.plate_y1, p.plate_x2, p.plate_y2), 1.2, 160)
        else:
            raise KeyError(kind)
        if data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return data

    # decisions
    def save(self, d: dict) -> dict:
        kind, video, reviewer = d["kind"], d["video"], str(d.get("reviewer") or "reviewer")[:40]
        human = d.get("label")
        with self.lock:
            if kind == "vehicle":  # the label it replaced goes to the log, so undo can put it back
                d = {**d, "text": json.dumps(self._save_vehicle(video, int(d["item"]), human, reviewer, d))}
            elif kind == "sign" and human != "skip":
                self._save_sign(video, int(d["item"]), int(d["row"]), human, str(d.get("text", "")), reviewer)
            elif kind == "plate" and human != "skip":
                self._save_plate(video, int(d["item"]), str(d.get("text", "")).upper().strip(), reviewer)
            self.log({"reviewer": reviewer, "kind": kind, "video": video, "item": d["item"], "model": d.get("model"),
                      "model_conf": d.get("model_conf"), "human": human, "text": d.get("text"),
                      "model_text": d.get("model_text")})
        return {"ok": True}

    def _save_vehicle(self, video: str, track_id: int, label: str, reviewer: str, d: dict) -> dict | None:
        """Add or update the track's label; returns the label row it replaced (None if it added one)."""
        if label == "skip":
            return None
        if label not in RELATIONS + ["unclear", "not_vehicle"]:
            raise ValueError(label)
        path = self.labels_dir / "track_labels.csv"
        labels = self._csv("track_labels.csv", ["card_id", "video", "track_id", "label", "labeler", "notes"])
        labels = labels.astype({c: object for c in ("label", "labeler", "notes")})  # empty notes read back as floats
        existing = (labels.video == video) & (labels.track_id == track_id)
        note = f"review: model said {d.get('model')}"
        replaced = None
        if existing.any():  # verify mode: the reviewer confirms or corrects an earlier label
            i = labels.index[existing][0]
            replaced = {k: (None if pd.isna(v) else v) for k, v in labels.loc[i].to_dict().items()}
            old = labels.at[i, "label"]
            labels.at[i, "notes"] = (f"{note}; was {old} ({labels.at[i, 'labeler']})" if old != label else
                                     f"{note}; confirmed {old} ({labels.at[i, 'labeler']})")
            labels.at[i, "label"], labels.at[i, "labeler"] = label, reviewer
        else:
            anchors = load_anchors(self.labels_dir / "track_anchors.csv")
            card = next_card_id(video, labels, anchors)
            t = self.dets(video)
            t = t[t.track_id == track_id].sort_values("frame")
            frames = [int(t.frame.iloc[int(k * (len(t) - 1) / 3)]) for k in range(4)]
            append_anchors(anchor_rows(card, video, t, frames), self.labels_dir / "track_anchors.csv")
            labels = pd.concat([labels, pd.DataFrame([{"card_id": card, "video": video, "track_id": track_id,
                                                       "label": label, "labeler": reviewer, "notes": note}])],
                               ignore_index=True)
        labels.to_csv(path, index=False)
        return replaced

    def _save_sign(self, video: str, object_id: int, row: int, label: str, text: str, reviewer: str) -> None:
        if label not in SIGN_TYPES_UI:
            raise ValueError(label)
        o = pd.read_parquet(self.analysis / video / "infrastructure.parquet").loc[row]
        labels = self._csv("sign_labels.csv", SIGN_LABEL_COLUMNS)
        labels = labels[~((labels.video == video) & (labels.object_id == object_id))]
        new = {"video": video, "object_id": object_id, "frame": int(o.frame), "x1": round(float(o.x1), 1),
               "y1": round(float(o.y1), 1), "x2": round(float(o.x2), 1), "y2": round(float(o.y2), 1), "label": label,
               "text": text, "reviewer": reviewer, "time": datetime.now().isoformat(timespec="seconds")}
        pd.concat([labels, pd.DataFrame([new], columns=SIGN_LABEL_COLUMNS)], ignore_index=True).to_csv(
            self.labels_dir / "sign_labels.csv", index=False)

    def _save_plate(self, video: str, track_id: int, text: str, reviewer: str) -> None:
        p = pd.read_parquet(self.analysis / video / "plates.parquet").set_index("track_id").loc[track_id]
        labels = self._csv("plate_labels.csv", PLATE_LABEL_COLUMNS)
        labels = labels[~((labels.video == video) & (labels.track_id == track_id))]
        new = {"video": video, "track_id": track_id, "frame": int(p.plate_frame), "x1": p.plate_x1, "y1": p.plate_y1,
               "x2": p.plate_x2, "y2": p.plate_y2, "text": text, "reviewer": reviewer,
               "time": datetime.now().isoformat(timespec="seconds")}
        pd.concat([labels, pd.DataFrame([new], columns=PLATE_LABEL_COLUMNS)], ignore_index=True).to_csv(
            self.labels_dir / "plate_labels.csv", index=False)

    def undo(self, reviewer: str) -> dict:
        """Take back the reviewer's last decision: the log entry, and a vehicle label it added."""
        with self.lock:
            log = self._csv("review_log.csv", REVIEW_LOG_COLUMNS)
            mine = log.index[log.reviewer == reviewer]
            if not len(mine):
                return {"ok": False}
            last = log.loc[mine[-1]]
            if last.kind == "vehicle" and last.human != "skip":
                labels = self._csv("track_labels.csv")
                labels = labels.astype({c: object for c in ("label", "labeler", "notes") if c in labels})
                hit = (labels.video == last.video) & (labels.track_id == int(last["item"]))
                replaced = json.loads(last.text) if isinstance(last.text, str) and last.text else None
                if replaced:  # a verified label: restore what it was
                    for key, value in replaced.items():
                        labels.loc[hit, key] = value
                else:  # a new label: remove it and the anchors written for its card
                    cards = set(labels.loc[hit, "card_id"])
                    labels = labels[~hit]
                    anchors = load_anchors(self.labels_dir / "track_anchors.csv")
                    if len(anchors) and cards:
                        anchors[~anchors.card_id.isin(cards)].to_csv(self.labels_dir / "track_anchors.csv", index=False)
                labels.to_csv(self.labels_dir / "track_labels.csv", index=False)
            for name, key in (("sign_labels.csv", "object_id"), ("plate_labels.csv", "track_id")):
                if last.kind == name.split("_")[0]:
                    table = self._csv(name)
                    table[~((table.video == last.video) & (table[key] == int(last["item"])))].to_csv(
                        self.labels_dir / name, index=False)
            log.drop(index=mine[-1]).to_csv(self.labels_dir / "review_log.csv", index=False)
            return {"ok": True, "kind": str(last.kind), "item": int(last["item"])}

    def progress(self) -> dict:
        log = self._csv("review_log.csv", REVIEW_LOG_COLUMNS)
        labels = self._csv("track_labels.csv")
        return {"agreement": agreement(log) if len(log) else {},
                "track_labels": labels.groupby("labeler").label.count().to_dict() if len(labels) else {},
                "sign_labels": int(len(self._csv("sign_labels.csv"))),
                "plate_labels": int(len(self._csv("plate_labels.csv"))), "retrain": self.retrain}

    def start_retrain(self) -> dict:
        if self.retrain["state"] == "running":
            return self.retrain

        def work() -> None:
            steps = [["bikesafe.train", "--analysis", str(self.analysis), "--perception", str(self.perception),
                      "--videos", str(self.videos)]]
            # signs are typed again with the new probe, keeping the text already read
            steps += [["bikesafe.infrastructure", str(self.perception / d.name), "--analysis", str(self.analysis),
                       "--videos", str(self.videos), "--read-only", "--keep-text"]
                      for d in sorted(self.analysis.iterdir()) if (d / "sign_crops.parquet").exists()]
            steps.append(["bikesafe.exposure", "--analysis", str(self.analysis), "--perception", str(self.perception),
                          "--out", str(self.corpus)])
            started = time.perf_counter()
            for step in steps:
                self.retrain.update(state="running", log=f"running {step[0]}")
                done = subprocess.run([sys.executable, "-m", *step], cwd=ROOT, capture_output=True, text=True)
                if done.returncode:
                    self.retrain.update(state="failed", log=done.stderr[-2000:])
                    return
            self.reload()
            self.retrain.update(state="done", log=f"retrained in {(time.perf_counter() - started) / 60:.1f} min")

        threading.Thread(target=work, daemon=True).start()
        self.retrain.update(state="running", log="starting")
        return self.retrain


# --------------------------------------------------------------------------------------------------------------------
# server


def clean(value):
    """NaN and infinity as null: they are not JSON, and a browser rejects the whole response over one of them."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def make_handler(store: Store, html: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet
            pass

        def send(self, body: bytes, content_type: str, status: int = 200) -> None:
            first, last = 0, len(body) - 1
            # byte ranges for clips: Safari plays a video only from a server that serves them
            ranged = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "")) if content_type == "video/mp4" else None
            if ranged and body and (ranged[1] or ranged[2]):
                first = int(ranged[1]) if ranged[1] else max(0, len(body) - int(ranged[2]))
                last = min(last, int(ranged[2])) if ranged[1] and ranged[2] else last
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(max(0, last - first + 1)))
            if content_type == "video/mp4":
                self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "max-age=3600")
            self.end_headers()
            self.wfile.write(body[first:last + 1])

        def json(self, payload, status: int = 200) -> None:
            self.send(json.dumps(clean(payload), default=str).encode(), "application/json", status)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self.send(html.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/queue":
                    kind = q.get("kind", "vehicle")
                    items = (store.vehicle_queue(q.get("mode", "new")) if kind == "vehicle" else
                             store.sign_queue() if kind == "sign" else store.plate_queue())
                    self.json(items)
                elif url.path == "/api/img":
                    video = q["video"]
                    if not store.known_video(video):
                        raise KeyError(video)
                    row = int(q["row"]) if "row" in q else None
                    self.send(store.image(q["kind"], video, int(q["item"]), row),
                              "video/mp4" if q["kind"] == "clip" else "image/jpeg")
                elif url.path == "/api/progress":
                    self.json(store.progress())
                else:
                    self.json({"error": "not found"}, 404)
            except Exception as error:  # report to the page instead of dropping the connection
                self.json({"error": repr(error)}, 500)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/api/label":
                    if not store.known_video(body["video"]):
                        raise KeyError(body["video"])
                    self.json(store.save(body))
                elif self.path == "/api/undo":
                    self.json(store.undo(str(body.get("reviewer") or "reviewer")))
                elif self.path == "/api/retrain":
                    self.json(store.start_retrain())
                else:
                    self.json({"error": "not found"}, 404)
            except Exception as error:
                self.json({"error": repr(error)}, 500)

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotation review</title>
<style>
:root { --bg:#f6f6f4; --panel:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#deded9; --accent:#2b6cb0; --warn:#b7791f;
  --ok:#2f855a; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --panel:#1f1f1d; --ink:#ececea; --muted:#9a9a94;
  --line:#33332f; --accent:#63a4e8; --warn:#e0a54a; --ok:#5fc08c; } }
* { box-sizing:border-box; } body { margin:0; font:15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
header { display:flex; gap:12px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line);
  background:var(--panel); flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 12px 0 0; } .tab { padding:6px 12px; border-radius:6px; cursor:pointer; border:1px solid transparent; }
.tab.on { border-color:var(--line); background:var(--bg); font-weight:600; }
input, select, button { font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px 9px; }
button { cursor:pointer; } button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1180px; margin:0 auto; padding:16px; } .grid { display:grid; grid-template-columns:minmax(0,640px) 1fr; gap:16px; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
#anim { width:100%; aspect-ratio:16/9; background:#000; border-radius:6px; display:block; }
.zoom img, .crop img { max-width:100%; border-radius:6px; display:block; } .muted { color:var(--muted); }
.reason { color:var(--warn); } .choices { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.choice { padding:6px 10px; border-radius:6px; border:1px solid var(--line); cursor:pointer; user-select:none; }
.choice.model { outline:2px solid var(--accent); } .choice kbd { font-size:12px; color:var(--muted); margin-right:4px; }
.bar { height:8px; background:var(--accent); border-radius:4px; } table { border-collapse:collapse; } td, th { padding:3px 8px; border-bottom:1px solid var(--line); text-align:left; }
.status { margin-left:auto; } .done { color:var(--ok); }
</style></head><body>
<header><h1>Annotation review</h1>
<span class="tab on" data-kind="vehicle">Vehicles</span><span class="tab" data-kind="sign">Signs</span>
<span class="tab" data-kind="plate">Plates</span><span class="tab" data-kind="progress">Progress</span>
<select id="mode"><option value="new">new: likely errors first</option><option value="verify">verify existing labels</option></select>
<label>Reviewer <input id="reviewer" size="12" placeholder="your name"></label>
<span class="status muted" id="status"></span></header>
<main id="main"></main>
<script>
const RELS = [["ego_lane","in my lane"],["adjacent_same","other lane, same way"],["oncoming","oncoming"],["parked","parked"],["cross_side","side road / other roadway"]];
const SIGNS = __SIGNS__;
// STATIC: the exported copy (python -m bikesafe.review --export), read from files with no server behind it; decisions
// are kept in this browser until downloaded from the Progress tab
const STATIC = __STATIC__;
let kind = "vehicle", queue = [], pos = 0, saved = [];
const $ = s => document.querySelector(s);
const reviewer = () => $("#reviewer").value.trim() || "reviewer";
try { $("#reviewer").value = localStorage.getItem("reviewer") || ""; saved = JSON.parse(localStorage.getItem("decisions") || "[]"); } catch (e) {}
const keep = () => { try { localStorage.setItem("decisions", JSON.stringify(saved)); } catch (e) {} };
$("#reviewer").addEventListener("change", () => { try { localStorage.setItem("reviewer", $("#reviewer").value); } catch (e) {} });
document.querySelectorAll(".tab").forEach(t => t.onclick = () => { document.querySelectorAll(".tab").forEach(x => x.classList.remove("on")); t.classList.add("on"); kind = t.dataset.kind; load(); });
$("#mode").onchange = load;
const img = (k, it, extra="") => STATIC ? `media/${it.video}/${k}_${it.item}.${k === "clip" ? "mp4" : "jpg"}`
  : `/api/img?kind=${k}&video=${encodeURIComponent(it.video)}&item=${it.item}${extra}`;
async function load() {
  if (kind === "progress") return progress();
  $("#main").innerHTML = '<p class="muted">Loading the queue...</p>';
  const r = await fetch(STATIC ? `queue_${kind === "vehicle" ? "vehicle_" + $("#mode").value : kind}.json` : `/api/queue?kind=${kind}&mode=${$("#mode").value}`);
  queue = await r.json(); pos = 0;
  if (STATIC && Array.isArray(queue)) queue = queue.filter(it => !saved.some(s => s.kind === it.kind && s.video === it.video && s.item === it.item));
  show();
}
function prefetch() { for (const it of queue.slice(pos + 1, pos + 3)) { if (kind === "vehicle") { fetch(img("clip", it)); new Image().src = img("zoom", it); } else if (kind === "sign") { new Image().src = img("sign", it, `&row=${it.row}`); } else { new Image().src = img("plate", it); } } }
function show() {
  if (queue.error) { $("#main").innerHTML = `<p>${queue.error}</p>`; return; }
  if (pos >= queue.length) { $("#main").innerHTML = `<div class="card"><p>Queue done. ${STATIC ? "Download your decisions under Progress." : "Reload for more, or Retrain under Progress."}</p></div>`; return; }
  const it = queue[pos]; $("#status").textContent = `${pos + 1} / ${queue.length}`;
  if (kind === "vehicle") showVehicle(it); else if (kind === "sign") showSign(it); else showPlate(it);
  prefetch();
}
function showVehicle(it) {
  const probs = Object.entries(it.proba).sort((a,b) => b[1]-a[1]).map(([k,v]) => `<tr><td>${k}</td><td style="width:140px"><div class="bar" style="width:${Math.round(v*100)}%"></div></td><td>${Math.round(v*100)}%</td></tr>`).join("");
  $("#main").innerHTML = `<div class="grid"><div class="card"><video id="anim" src="${img("clip", it)}" autoplay muted loop playsinline></video><p class="muted">${it.video} &middot; t=${it.t}s &middot; closest ${it.z_min} m, ${it.x} m to the side &middot; ${it.duration} s</p></div>
  <div class="card"><div class="zoom"><img src="${img("zoom", it)}" alt="vehicle"></div>
  <p>Model: <b>${it.model}</b>${it.current ? ` &middot; label now: <b>${it.current}</b>` : ""} &middot; rules: ${it.rule}</p>
  ${it.reasons ? `<p class="reason">${it.reasons}</p>` : ""}<table>${probs}</table>
  <div class="choices">${RELS.map(([k,n],i) => `<span class="choice ${k===it.model?"model":""}" data-v="${k}"><kbd>${i+1}</kbd>${n}</span>`).join("")}
  <span class="choice" data-v="unclear"><kbd>U</kbd>unclear</span><span class="choice" data-v="not_vehicle"><kbd>N</kbd>not a vehicle</span>
  <span class="choice" data-v="skip"><kbd>S</kbd>skip</span></div>
  <p class="muted">Enter accepts the model &middot; Z undoes the last decision</p></div></div>`;
  document.querySelectorAll(".choice").forEach(c => c.onclick = () => decide(c.dataset.v));
}
function showSign(it) {
  $("#main").innerHTML = `<div class="grid"><div class="card crop"><img src="${img("signctx", it, `&row=${it.row}`)}" alt="context"></div>
  <div class="card"><div class="crop"><img src="${img("sign", it, `&row=${it.row}`)}" alt="sign"></div>
  <p>Model: <b>${it.model}</b> (${Math.round(it.model_conf*100)}%) &middot; ${it.h} px tall &middot; ${it.video} t=${it.t}s</p>
  <label>Text <input id="text" size="34" value="${(it.text||"").replace(/"/g,"&quot;")}"></label>
  <div class="choices">${SIGNS.map(([k,key]) => `<span class="choice ${k===it.model?"model":""}" data-v="${k}"><kbd>${key}</kbd>${k.replace(/_/g," ")}</span>`).join("")}
  <span class="choice" data-v="skip"><kbd>S</kbd>skip</span></div><p class="muted">Enter accepts the model's type with the text as written &middot; T edits the text</p></div></div>`;
  document.querySelectorAll(".choice").forEach(c => c.onclick = () => decide(c.dataset.v));
}
function showPlate(it) {
  $("#main").innerHTML = `<div class="grid"><div class="card crop"><img src="${img("plate", it)}" alt="plate"></div>
  <div class="card"><p>Read: <b>${it.model || "(nothing confident)"}</b> &middot; reads: <span class="muted">${it.reads}</span></p>
  <label>Plate <input id="text" size="12" value="${it.model}"></label>
  <div class="choices"><span class="choice model" data-v="read"><kbd>Enter</kbd>save text</span><span class="choice" data-v="unreadable"><kbd>X</kbd>unreadable</span><span class="choice" data-v="skip"><kbd>S</kbd>skip</span></div></div></div>`;
  document.querySelectorAll(".choice").forEach(c => c.onclick = () => decide(c.dataset.v));
  $("#text").focus();
}
async function decide(v) {
  const it = queue[pos]; if (!it) return;
  const text = $("#text") ? $("#text").value : "";
  let label = v, out = {kind: it.kind, video: it.video, item: it.item, row: it.row, model: it.model, model_conf: it.model_conf || Math.max(...Object.values(it.proba || {0:0})), reviewer: reviewer()};
  if (it.kind === "plate") { label = v === "skip" ? "skip" : "plate"; out.text = v === "unreadable" ? "" : text; out.model_text = it.model; out.model = "plate"; }
  else if (it.kind === "sign") { out.text = text; out.model_text = it.text; }
  out.label = label;
  if (STATIC) { saved.push({...out, time: new Date().toISOString()}); keep(); pos += 1; show(); return; }
  const r = await (await fetch("/api/label", {method:"POST", body: JSON.stringify(out)})).json();
  if (r.error) { alert(r.error); return; } pos += 1; show();
}
document.addEventListener("keydown", e => {
  if (kind === "progress" || !queue[pos]) return;
  const typing = document.activeElement && document.activeElement.id === "text";
  if (kind === "plate") { if (e.key === "Enter") decide("read"); else if (!typing && e.key.toLowerCase() === "x") decide("unreadable"); else if (!typing && e.key.toLowerCase() === "s") decide("skip"); return; }
  if (typing) { if (e.key === "Enter") { decide(queue[pos].model); } if (e.key === "Escape") document.activeElement.blur(); return; }
  const k = e.key.toLowerCase();
  if (e.key === "Enter") decide(queue[pos].model);
  else if (k === "z") undo();
  else if (k === "s") decide("skip");
  else if (kind === "vehicle" && "12345".includes(k)) decide(RELS[+k - 1][0]);
  else if (kind === "vehicle" && k === "u") decide("unclear");
  else if (kind === "vehicle" && k === "n") decide("not_vehicle");
  else if (kind === "sign" && k === "t") { e.preventDefault(); $("#text").focus(); }
  else if (kind === "sign") { const hit = SIGNS.find(([, key]) => key === k); if (hit) decide(hit[0]); }
});
async function undo() {
  if (STATIC) { if (saved.length && pos > 0) { saved.pop(); keep(); pos -= 1; show(); } return; }
  const r = await (await fetch("/api/undo", {method:"POST", body: JSON.stringify({reviewer: reviewer()})})).json(); if (r.ok && pos > 0) { pos -= 1; show(); }
}
function staticProgress() {
  $("#main").innerHTML = `<div class="card"><p>A copy of the review page that needs no server: the queues are as they were when it was made, and
    your decisions stay in this browser - <b>${saved.length}</b> so far.</p>
    <p><button class="primary" id="dl">Download decisions</button> <button id="clear">Clear them</button></p>
    <p class="muted">To add them to the labels: <code>python -m bikesafe.review --import decisions.json</code>, then Retrain.</p></div>`;
  $("#dl").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(saved, null, 1)], {type: "application/json"})); a.download = "decisions.json"; a.click(); };
  $("#clear").onclick = () => { if (confirm("Delete the decisions kept in this browser?")) { saved = []; keep(); staticProgress(); } };
}
async function progress() {
  if (STATIC) return staticProgress();
  const p = await (await fetch("/api/progress")).json();
  const blocks = Object.entries(p.agreement).map(([k, a]) => `<div class="card"><h3>${k}</h3><p>${a.decisions} decisions, ${a.unclear} unclear &middot; model kept by the reviewer: <b>${a.model_kept === null ? "-" : Math.round(a.model_kept*100) + "%"}</b></p>
    ${a.confusion.length ? `<table><tr><th>model said</th><th>reviewer said</th><th>n</th></tr>${a.confusion.sort((x,y)=>y.n-x.n).map(c => `<tr><td>${c.model}</td><td>${c.human}</td><td>${c.n}</td></tr>`).join("")}</table>` : ""}</div>`).join("");
  $("#main").innerHTML = `<div class="card"><p>Vehicle labels by labeller: ${Object.entries(p.track_labels).map(([k,v]) => `${k} ${v}`).join(", ") || "none"} &middot; sign labels ${p.sign_labels} &middot; plate labels ${p.plate_labels}</p>
    <p><button class="primary" id="retrain">Retrain on the labels</button> <span id="rt" class="muted">${p.retrain.state}: ${p.retrain.log}</span></p></div>${blocks}`;
  $("#retrain").onclick = async () => { const r = await (await fetch("/api/retrain", {method:"POST", body:"{}"})).json(); $("#rt").textContent = `${r.state}: ${r.log}`; };
  if (p.retrain.state === "running") setTimeout(progress, 5000);
}
load();
</script></body></html>
"""


def page(static: bool = False) -> str:
    keys = "abcdefghijklmnopqrstuvwxyz"
    shortcuts = [(t, k) for t, k in zip(SIGN_TYPES_UI, (c for c in keys if c not in "stuz"))]
    return PAGE.replace("__SIGNS__", json.dumps(shortcuts)).replace("__STATIC__", json.dumps(static))


MEDIA = {"vehicle": ["clip", "zoom"], "sign": ["sign", "signctx"], "plate": ["plate"]}


def export(store: Store, out: Path, per_queue: int) -> None:
    """The page as files that need no server (GitHub Pages, any web host): the first `per_queue` items of each queue
    with their clips and crops. Decisions made there stay in the viewer's browser until downloaded (Progress tab);
    --import adds them to the labels."""
    shutil.rmtree(out / "media", ignore_errors=True)
    queues = {"vehicle_new": store.vehicle_queue("new"), "vehicle_verify": store.vehicle_queue("verify"),
              "sign": store.sign_queue(), "plate": store.plate_queue()}
    for name, items in queues.items():
        kept = []
        for it in items[:per_queue]:
            files = {kind: store.image(kind, it["video"], it["item"], it.get("row")) for kind in MEDIA[it["kind"]]}
            if not all(files.values()):
                continue
            for kind, data in files.items():
                path = out / "media" / it["video"] / f"{kind}_{it['item']}.{'mp4' if kind == 'clip' else 'jpg'}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            kept.append(it)
        (out / f"queue_{name}.json").write_text(json.dumps(clean(kept), default=str), encoding="utf-8")
        print(f"{name}: {len(kept)} items", flush=True)
    (out / "index.html").write_text(page(static=True), encoding="utf-8")


def import_decisions(store: Store, path: Path) -> int:
    """Decisions downloaded from an exported page, saved as if they had been made here (reviewer names kept)."""
    decisions = json.loads(path.read_text(encoding="utf-8"))
    unknown = {d["video"] for d in decisions if not store.known_video(d["video"])}
    if unknown:
        raise SystemExit(f"rides not in {store.perception}: {', '.join(sorted(unknown))}")
    for d in decisions:
        store.save(d)
    return len(decisions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results" / "corpus", help="Where tracks_typed.csv are")
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labels", type=Path, default=LABELS_DIR,
                        help="Where decisions are saved (track_labels.csv, sign_labels.csv, ...; default data/typology)")
    parser.add_argument("--export", type=Path, default=None,
                        help="Write the page and its queues as static files to this folder instead of serving it")
    parser.add_argument("--per-queue", type=int, default=20, help="Items per queue in the --export copy")
    parser.add_argument("--import", dest="import_", type=Path, default=None,
                        help="Add the decisions downloaded from an exported page (decisions.json) to the labels")
    args = parser.parse_args()
    args.labels.mkdir(parents=True, exist_ok=True)
    store = Store(args.analysis, args.perception, args.corpus, args.videos, labels_dir=args.labels)
    if args.export is not None:
        export(store, args.export, args.per_queue)
        print(f"wrote {args.export / 'index.html'}")
        return
    if args.import_ is not None:
        print(f"added {import_decisions(store, args.import_)} decisions to {args.labels}")
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, page()))
    print(f"review at http://127.0.0.1:{args.port}  ({len(store.tracks)} tracks from {args.corpus})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
