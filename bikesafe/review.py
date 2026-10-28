"""Review and correct the pipeline's annotations in the browser: the human-in-the-loop part of the pipeline.

    python -m bikesafe.review [--corpus results/corpus] [--port 8765]

then open http://127.0.0.1:8765 and enter your name. Three queues, each pre-filled with what the models
said:

* **Vehicles** - the relation of a vehicle to the rider, shown as a looping clip of its track (magenta box) and a
  zoom. Keys 1-5 pick the relation, Enter accepts the model's answer (in verify mode: keeps the label), U marks it
  unclear, N not a vehicle (a delivery robot, a scooter, a bin), S skips, Z undoes; every key has a button too.
* **Signs** - each physical sign's largest whole view with its type and text (bikesafe.signs); pick the type, correct the
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
    python -m bikesafe.review --import "G:/My Drive/bikesafe_runs/<video>"   # a Colab run and its decisions

The exported copy has the first --per-queue items of each queue with their clips and crops - none filmed in a
ride's privacy zones (bikesafe.gps; on a ride without GPS, its first and last NO_GPS_PRIVATE_S), as the copy is
public; its decisions stay in the viewer's browser until downloaded (header or Progress tab). Importing them twice
adds them once. Retrain is offered only where the decisions go to
data/typology, beside the labelled rides: a Colab run (--labels in its Drive folder) collects decisions, and the
models learn from them once the run is imported on that machine.
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
MAX_PLATE_CHARS = 8  # US plates carry at most 8 characters (California 7)
MIN_PLATE_W, MAX_PLATE_W = 40, 300  # px; narrower plates are unreadable to a reviewer too, wider boxes are not plates
MAX_Z_M = 25.0  # vehicles reviewed: those that came this close (where relations are validated and matter)
MIN_BOX_H = 40
SPREAD_S = 4.0  # at most one queued vehicle per ride and this many seconds, so a queue is not one traffic jam
CLIP_WIDTH = 960  # px; vehicle clips
CLIP_CRF = 25
CLIP_MAX_S = 8.0  # a longer track plays faster, so every clip loops in about this long
CLIP_PAD_S = 1.0  # shown before the track starts and after it ends
MAX_PER_SIGN_TYPE = QUEUE_SIZE // 4  # a sign queue is not twenty of one type
NO_GPS_PRIVATE_S = 180  # a ride without GPS: this much of its start and end is kept out of the public copy
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
    flag(get("z_min") < 13, 0.2, "close to the rider")
    flag(get("not_vehicle") >= 0.5, 0.5, "may not be a vehicle")
    t["score"] = score
    t["reasons"] = ["; ".join(r) for r in reasons]
    return t.sort_values("score", ascending=False)


def labeller_text(who) -> str | None:
    """Who made a label, as the page shows it: the first pass was AI-assisted (labeler claude-visual-review)."""
    if not isinstance(who, str) or not who:
        return None
    return "AI-assisted review" if who.startswith("claude") else "an earlier reviewer" if who == "reviewer" else who


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
    """A card id (T<ride>_<n>) not used by any label or anchor, of any video: card ids key the anchors, and videos can
    share the name ending the id is made of (every 3-minute cut of the Colab notebook ends in "_3min")."""
    prefix = f"T{video.rsplit('_', 1)[-1]}_"
    cards = pd.concat([labels.card_id, anchors.card_id]).dropna().astype(str)
    numbers = pd.to_numeric(cards[cards.str.startswith(prefix)].str[len(prefix):], errors="coerce").dropna()
    return f"{prefix}{int(numbers.max()) + 1 if len(numbers) else 1:03d}"


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
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(image, (0, 0), (tw + 16, th + 16), (20, 20, 20), -1)
        cv2.putText(image, label, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
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
        # Retraining learns from data/typology over the rides in `analysis`. Decisions saved elsewhere (a Colab run's
        # labels/) only join them through --import on the machine that has those rides.
        self.retrainable = labels_dir.resolve() == LABELS_DIR.resolve()
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
        self._fresh: set[str] = set()

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
                          "current": getattr(row, "label", None), "current_by": labeller_text(getattr(row, "labeler", None)),
                          "reasons": row.reasons,
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
            calib = read_json(path.parent / "calibration.json")
            signs = obj[obj.kind.isin(["sign", "rejected_sign"])].assign(h=lambda o: o.y2 - o.y1)
            # views cut off by the picture's edge are the largest but show part of the sign
            whole = (signs.x1 > 2) & (signs.y1 > 2) & (signs.x2 < calib["width"] - 2) & (signs.y2 < calib["height"] - 2)
            for object_id, g in signs.groupby("object_id"):
                if (video, int(object_id)) in keys:
                    continue
                views = g[whole.loc[g.index]]
                best = (views if len(views) else g).sort_values("h").iloc[-1]
                if best.h < 20:
                    continue
                conf = float(best.type_conf) if pd.notna(best.type_conf) else 0.0
                category = best.category if best.kind == "sign" else "not_a_sign"
                items.append({"kind": "sign", "video": video, "item": int(object_id), "row": int(best.name),
                              "model": category, "model_conf": round(conf, 3), "text": str(best.sign_text or ""),
                              "h": int(best.h), "t": round(float(best.time_s), 1),
                              "score": (1 - conf) + 0.3 * (category in ("other_sign", "not_a_sign")) + best.h / 1000})
        queue, per_type = [], {}
        for item in sorted(items, key=lambda i: -i["score"]):
            if per_type.get(item["model"], 0) < MAX_PER_SIGN_TYPE:
                per_type[item["model"]] = per_type.get(item["model"], 0) + 1
                queue.append(item)
        return queue[:QUEUE_SIZE]

    def plate_queue(self) -> list[dict]:
        done = self._csv("plate_labels.csv", PLATE_LABEL_COLUMNS)
        keys = set(zip(done.video, done.track_id))
        items = []
        for path in sorted(self.analysis.glob("*/plates.parquet")):
            video = path.parent.name
            plates = pd.read_parquet(path)
            fps = float(read_json(path.parent / "calibration.json").get("source_fps", 24.0))
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
                if len(read) > MAX_PLATE_CHARS:
                    continue  # lettering on a truck or a shop sign, not a plate
                conf = float(row.plate_conf) if pd.notna(row.plate_conf) else 0.0
                # most useful to check: readings that are not certain (one read, or reads that disagree), then
                # readable-sized plates the reader gave up on
                uncertain = read != "" and (row.agreeing_reads < 2 or conf < 0.95)
                items.append({"kind": "plate", "video": video, "item": int(row.track_id), "model": read,
                              "model_conf": round(conf, 3), "reads": str(row.all_reads or ""),
                              "t": round(float(row.plate_frame) / fps, 1), "w": int(row.plate_x2 - row.plate_x1),
                              "score": 1.0 * uncertain + 0.5 * (read == "") + min(row.plate_x2 - row.plate_x1, 150) / 1000})
        queue, seen = [], set()
        for item in sorted(items, key=lambda i: -i["score"]):  # one plate read once: the same car, other tracks
            if item["model"] and (item["video"], item["model"]) in seen:
                continue
            seen.add((item["video"], item["model"]))
            queue.append(item)
        return queue[:QUEUE_SIZE]

    # images and clips (cached on disk)
    def fresh_cache(self, video: str) -> None:
        """Empty the ride's cached clips and crops when its tables changed since they were made: after a re-run the
        same track or object id can be another vehicle or sign."""
        if video in self._fresh:
            return
        sources = [self.analysis / video / name for name in ("kinematics.parquet", "infrastructure.parquet",
                                                             "plates.parquet", "sign_crops.parquet")]
        stamp = str(max((int(f.stat().st_mtime) for f in sources if f.exists()), default=0)) + f"/pad{CLIP_PAD_S}"
        marker = self.cache / video / "made_from.txt"
        if not marker.exists() or marker.read_text() != stamp:  # unknown or older tables
            shutil.rmtree(self.cache / video, ignore_errors=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(stamp)
        self._fresh.add(video)

    def image(self, kind: str, video: str, item: int, row: int | None = None) -> bytes:
        self.fresh_cache(video)
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
            entry = {"reviewer": reviewer, "kind": kind, "video": video, "item": d["item"], "model": d.get("model"),
                     "model_conf": d.get("model_conf"), "human": human, "text": d.get("text"),
                     "model_text": d.get("model_text")}
            self.log({**entry, "time": d["time"]} if d.get("time") else entry)
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
        if int(o.object_id) != object_id:
            raise ValueError(f"row {row} is not a view of sign {object_id}")
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

    def undo(self, reviewer: str, expect: dict | None = None) -> dict:
        """Take back the reviewer's last decision: the log entry, and a vehicle label it added. With `expect` (the
        kind, video and item the page means to take back) nothing is undone unless that is the last decision."""
        with self.lock:
            log = self._csv("review_log.csv", REVIEW_LOG_COLUMNS)
            mine = log.index[log.reviewer == reviewer]
            if not len(mine):
                return {"ok": False, "reason": "nothing to undo"}
            last = log.loc[mine[-1]]
            if expect and (str(last.kind), str(last.video), str(last["item"])) != (
                    str(expect.get("kind")), str(expect.get("video")), str(expect.get("item"))):
                return {"ok": False, "reason": "the last decision saved is not the one on the page"}
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
                "plate_labels": int(len(self._csv("plate_labels.csv"))), "retrain": self.retrain,
                "retrainable": self.retrainable, "labels_dir": str(self.labels_dir)}

    def start_retrain(self) -> dict:
        if not self.retrainable:
            return {"state": "unavailable", "log": f"decisions are saved in {self.labels_dir}; retrain after --import"}
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
                    self.json(store.undo(str(body.get("reviewer") or "reviewer"), body.get("expect")))
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
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Overpass:wght@600;700;800&display=swap" rel="stylesheet">
<style>
:root { --bg:#e6eaeb; --panel:#f4f6f6; --ink:#161a1d; --muted:#4f5a60; --line:#cdd4d6; --accent:#1f7445; --on-accent:#fff;
  --warn:#8a5a00; --ok:#2f855a; --bad:#c53030; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --panel:#1f1f1d; --ink:#ececea; --muted:#a3a39d;
  --line:#323a3f; --accent:#35a565; --on-accent:#0b1a11; --warn:#e0a54a; --ok:#5fc08c; --bad:#fc8181; } }
* { box-sizing:border-box; } body { margin:0; font:15px/1.45 "Atkinson Hyperlegible", system-ui, sans-serif; background:var(--bg); color:var(--ink); }
header { display:flex; gap:10px 14px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line);
  background:var(--panel); flex-wrap:wrap; position:sticky; top:0; z-index:5; }
header h1 { font:800 17px "Overpass", sans-serif; margin:0 6px 0 0; white-space:nowrap; }
.plate { background:#1f6e42; color:#fff; font:700 13px "Overpass", sans-serif; padding:4px 9px 2px; border-radius:6px;
  box-shadow:inset 0 0 0 2px #1f6e42, inset 0 0 0 4px #fff; text-decoration:none; white-space:nowrap; }
.tab, button, .choice { font-family:"Overpass", "Atkinson Hyperlegible", sans-serif; }
.tabs { display:flex; gap:4px; overflow-x:auto; scrollbar-width:none; }
.tab { padding:6px 12px; border-radius:6px; border:1px solid transparent; background:none; white-space:nowrap; }
.tab[aria-selected=true] { border-color:var(--line); background:var(--bg); font-weight:600; }
input, select, button { font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px 9px; }
button { cursor:pointer; } button.primary { background:var(--accent); color:var(--on-accent); border-color:var(--accent); }
button:focus-visible, .tab:focus-visible, input:focus-visible, select:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
#reviewer.missing { border-color:var(--bad); }
.status { margin-left:auto; display:flex; gap:10px; align-items:center; }
main { max-width:1180px; margin:0 auto; padding:16px; }
.grid { display:grid; grid-template-columns:minmax(0,640px) minmax(0,1fr); gap:16px; align-items:start; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
#anim { width:100%; aspect-ratio:16/9; background:#000; border-radius:6px; display:block; }
.zoom img, .crop img { max-width:100%; border-radius:6px; display:block; margin:0 auto; }
.crop img.big { width:100%; max-height:60vh; object-fit:contain; image-rendering:auto; }
.muted { color:var(--muted); } .small { font-size:13px; } .reason { color:var(--warn); } p { margin:8px 0; }
.choices { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.choice { padding:6px 10px; border-radius:6px; border:1px solid var(--line); background:var(--panel); text-align:left; }
.choice.default { outline:2px solid var(--accent); } .choice kbd { font:12px ui-monospace, monospace; color:var(--muted); margin-right:5px; }
.choice:hover { border-color:var(--accent); }
.bar { height:8px; background:var(--accent); border-radius:4px; min-width:1px; }
table { border-collapse:collapse; width:100%; } td, th { padding:3px 8px; border-bottom:1px solid var(--line); text-align:left; }
td.num { text-align:right; width:48px; font-variant-numeric:tabular-nums; }
.actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.notice { padding:8px 12px; border-radius:8px; margin:0 0 12px; border:1px solid var(--warn); color:var(--warn); }
.notice.bad { border-color:var(--bad); color:var(--bad); }
#toast { position:fixed; left:50%; bottom:18px; transform:translateX(-50%); background:var(--ink); color:var(--bg);
  padding:7px 14px; border-radius:8px; opacity:0; transition:opacity .2s; pointer-events:none; z-index:10; }
#toast.on { opacity:.92; }
.hint { font-size:13px; color:var(--muted); }
@media (hover:none) { .hint, .choice kbd { display:none; } }
@media (max-width:600px) {
  header { padding:8px 10px; } main { padding:10px; }
  .decide { position:sticky; bottom:0; background:var(--panel); border-top:1px solid var(--line); margin:0 -14px -14px;
    padding:8px 14px; border-radius:0 0 10px 10px; }
  .choices { margin:6px 0; } .choice { flex:1 1 auto; }
}
</style></head><body>
<header>__HOME__<h1>Annotation review</h1>
<div class="tabs" role="tablist" aria-label="Queues">
<button class="tab" role="tab" aria-selected="true" data-kind="vehicle">Vehicles</button>
<button class="tab" role="tab" aria-selected="false" data-kind="sign">Signs</button>
<button class="tab" role="tab" aria-selected="false" data-kind="plate">Plates</button>
<button class="tab" role="tab" aria-selected="false" data-kind="progress">Progress</button></div>
<label id="modebox">Queue <select id="mode"><option value="new">new: likely errors first</option>
<option value="verify">verify existing labels</option></select></label>
<label>Reviewer <input id="reviewer" size="12" placeholder="your name" autocomplete="name"></label>
<div class="status"><span class="muted" id="status" aria-live="polite"></span><span id="savedbox"></span></div></header>
<main id="main"></main><div id="toast" role="status" aria-live="polite"></div>
<script>
const RELS = [["ego_lane","in my lane","In the lane the rider is travelling in: ahead of the rider (moving or queued at a light), or in the rider's bike lane."],
  ["adjacent_same","other lane, same way","Going the rider's way on the rider's road, in another lane - including vehicles overtaking the rider."],
  ["oncoming","oncoming","Coming the other way on the rider's road (no physical median between)."],
  ["parked","parked","Left at the kerb, in a parking lane, a driveway or a lot. Stopped in traffic is not parked."],
  ["cross_side","side road / other roadway","Not on the rider's road: side streets, traffic crossing or turning at a junction, driveways, beyond a median."]];
const NAME = Object.fromEntries(RELS.map(([k, n]) => [k, n]));
Object.assign(NAME, {unclear: "unclear", not_vehicle: "not a vehicle", skip: "skipped"});
const SIGNS = __SIGNS__;
// STATIC: the exported copy (python -m bikesafe.review --export), read from files with no server behind it; decisions
// are kept in this browser until downloaded
const STATIC = __STATIC__;
const KEY = "bikesafe-review:";
let kind = "vehicle", queue = [], pos = 0, saved = [], history = [], busy = false, stored = true;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const name = k => NAME[k] || String(k ?? "-").replace(/_/g, " ");
const readable = t => /[A-Za-z]{3}|\d\s?[AP]M/.test(t || "");
const clock = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const ride = v => `ride ${String(v).slice(-1)}`;
function get(k, fallback) { try { return localStorage.getItem(KEY + k) ?? localStorage.getItem(k) ?? fallback; } catch (e) { return fallback; } }
function put(k, v) { try { localStorage.setItem(KEY + k, v); return true; } catch (e) { return false; } }
$("#reviewer").value = get("reviewer", "");
try { saved = JSON.parse(get("decisions", "[]")) || []; } catch (e) { saved = []; }
const decided = () => saved.filter(d => d.label !== "skip").length;
function keep() { stored = put("decisions", JSON.stringify(saved)); savedBox(); }
function savedBox() {
  if (!STATIC) return;
  $("#savedbox").innerHTML = `<span class="muted">${decided()} saved here</span> <button id="dl" class="primary">Download</button>`;
  $("#dl").onclick = download;
}
function download() {
  const who = ($("#reviewer").value.trim() || "reviewer").replace(/[^\w-]+/g, "_");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(saved, null, 1)], {type: "application/json"}));
  a.download = `decisions_${who}_${new Date().toISOString().slice(0, 10)}.json`; a.click();
}
function toast(text) { const t = $("#toast"); t.textContent = text; t.classList.add("on"); clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove("on"), 1400); }
$("#reviewer").addEventListener("input", () => { put("reviewer", $("#reviewer").value); $("#reviewer").classList.remove("missing"); });
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.setAttribute("aria-selected", x === t ? "true" : "false"));
  kind = t.dataset.kind; history = []; load(); });
$("#mode").onchange = () => { $("#mode").blur(); history = []; load(); };
const img = (k, it, extra="") => STATIC ? `media/${it.video}/${k}_${it.item}.${k === "clip" ? "mp4" : "jpg"}`
  : `/api/img?kind=${k}&video=${encodeURIComponent(it.video)}&item=${it.item}${extra}`;
async function load() {
  $("#modebox").style.display = kind === "vehicle" ? "" : "none";
  if (kind === "progress") { $("#status").textContent = ""; return progress(); }
  $("#main").innerHTML = '<p class="muted">Loading the queue...</p>';
  const wanted = kind + $("#mode").value;
  let items;
  try {
    const r = await fetch(STATIC ? `queue_${kind === "vehicle" ? "vehicle_" + $("#mode").value : kind}.json` : `/api/queue?kind=${kind}&mode=${$("#mode").value}`);
    items = await r.json();
  } catch (e) {
    $("#main").innerHTML = `<div class="notice bad">Could not load the queue (${esc(e.message)}). ${STATIC ? "Open the page from a web server (the project page), not as a file." : "Is the review server still running?"}</div>`;
    return;
  }
  if (kind + $("#mode").value !== wanted) return;  // another tab or mode was picked while this queue loaded
  queue = items; pos = 0;
  if (STATIC && Array.isArray(queue))  // decided items stay done; skipped ones come back
    queue = queue.filter(it => !saved.some(s => s.label !== "skip" && s.kind === it.kind && s.video === it.video && s.item === it.item));
  show();
}
function prefetch() { for (const it of queue.slice(pos + 1, pos + 3)) { if (kind === "vehicle") { fetch(img("clip", it)); new Image().src = img("zoom", it); } else if (kind === "sign") { new Image().src = img("sign", it, `&row=${it.row}`); } else { new Image().src = img("plate", it); } } }
const notices = () => (!stored ? `<div class="notice bad">This browser is not keeping your decisions (a private window, or storage is blocked): download them before you close the page.</div>` : "");
const undoButton = () => history.length ? `<button id="undo">Undo <span class="hint">(Z)</span></button>` : "";
function show() {
  window.scrollTo(0, 0);
  if (!Array.isArray(queue)) { $("#main").innerHTML = `<div class="notice bad">${esc(queue.error || "No queue")}</div>`; return; }
  if (pos >= queue.length) {
    $("#status").textContent = queue.length ? `${queue.length} / ${queue.length}` : "";
    $("#main").innerHTML = `${notices()}<div class="card"><p>${queue.length ? `Queue done - ${history.length} decided in this visit.` : "Nothing to review in this queue."}
      ${STATIC ? "Your decisions stay in this browser until you download them." : "Reload for more, or Retrain under Progress."}</p>
      <div class="actions">${STATIC && decided() ? '<button class="primary" id="dl2">Download decisions</button>' : ""}${undoButton()}</div></div>`;
    if ($("#dl2")) $("#dl2").onclick = download;
    if ($("#undo")) $("#undo").onclick = undo;
    return;
  }
  const it = queue[pos]; $("#status").textContent = `${pos + 1} / ${queue.length}`;
  if (kind === "vehicle") showVehicle(it); else if (kind === "sign") showSign(it); else showPlate(it);
  document.querySelectorAll(".choice").forEach(c => c.onclick = () => decide(c.dataset.v));
  if ($("#undo")) $("#undo").onclick = undo;
  prefetch();
}
function choiceButtons(list, preset) {
  return list.map(([v, label, key, tip]) => `<button class="choice ${v === preset ? "default" : ""}" data-v="${v}" ${tip ? `title="${esc(tip)}"` : ""}><kbd>${key}</kbd>${esc(label)}</button>`).join("");
}
function showVehicle(it) {
  const verify = !!it.current, preset = verify ? it.current : it.model;
  const probs = RELS.filter(([k]) => it.proba && k in it.proba).map(([k, n]) => `<tr><td>${n}</td><td style="width:45%"><div class="bar" style="width:${Math.round(it.proba[k]*100)}%"></div></td><td class="num">${Math.round(it.proba[k]*100)}%</td></tr>`).join("");
  const choices = [...RELS.map(([k, n, tip], i) => [k, n, i + 1, tip]), ["unclear", "unclear", "U", "Too small, dark or occluded, or the box jumps between vehicles."],
    ["not_vehicle", "not a vehicle", "N", "A scooter, bicycle, delivery robot, bin..."], ["skip", "skip", "S", "Decide later."]];
  $("#main").innerHTML = `${notices()}<div class="grid"><div class="card"><video id="anim" src="${img("clip", it)}" autoplay muted loop playsinline controls></video>
    <p class="muted small"><b>Magenta box: the vehicle to judge</b> &middot; grey: other vehicles &middot; ${ride(it.video)} at ${clock(it.t)} &middot; closest ${it.z_min} m ahead, ${Math.abs(it.x)} m to the ${it.x < 0 ? "left" : "right"} &middot; seen ${it.duration} s</p>
    <p class="muted small">${esc(it.video)}</p></div>
  <div class="card"><div class="zoom"><img src="${img("zoom", it)}" alt="the vehicle, close up"></div>
  <p>Model: <b>${name(it.model)}</b>${it.rule && it.rule !== it.model ? ` &middot; <span class="muted">rule of thumb: ${name(it.rule)}</span>` : ""}</p>
  ${verify ? `<p>Labelled <b>${name(it.current)}</b> by ${esc(it.current_by || "an earlier reviewer")}: keep it or correct it.</p>` : ""}
  ${it.reasons && !verify ? `<p class="reason">Why it is here: ${esc(it.reasons)}</p>` : ""}<table>${probs}</table>
  <div class="decide"><div class="choices">${choiceButtons(choices, preset)}</div>
  <div class="actions"><button class="primary" id="accept">${verify ? "Keep" : "Accept"} ${esc(name(preset))} <span class="hint">(Enter)</span></button>${undoButton()}</div></div>
  <p class="hint">Hover a choice for what it means &middot; keys 1-5, U, N, S &middot; Z undoes</p></div></div>`;
  $("#accept").onclick = () => decide(preset);
}
function showSign(it) {
  const text = readable(it.text) ? it.text : "";
  const conf = it.model_conf > 0 ? ` (${Math.round(it.model_conf*100)}% sure)` : it.model === "other_sign" ? " (type not decided)" : "";
  $("#main").innerHTML = `${notices()}<div class="grid"><div class="card crop"><img src="${img("signctx", it, `&row=${it.row}`)}" alt="the sign in its surroundings">
    <p class="muted small">Yellow box: the sign &middot; ${ride(it.video)} at ${clock(it.t)} &middot; ${it.h} px tall</p></div>
  <div class="card"><div class="crop"><img class="big" src="${img("sign", it, `&row=${it.row}`)}" alt="the sign"></div>
  <p>Model: <b>${name(it.model)}</b>${conf}</p>
  <label>What it says <input id="text" size="30" value="${esc(text)}" placeholder="${esc(it.text && !text ? "read: " + it.text : "")}"></label>
  <div class="decide"><div class="choices">${choiceButtons([...SIGNS.map(([k, key]) => [k, k.replace(/_/g, " "), key.toUpperCase()]), ["skip", "skip", "S"]], it.model)}</div>
  <div class="actions"><button class="primary" id="accept">Accept ${esc(name(it.model))} <span class="hint">(Enter)</span></button>${undoButton()}</div></div>
  <p class="hint">T edits the text, Esc leaves it &middot; Z undoes</p></div></div>`;
  $("#accept").onclick = () => decide(it.model);
}
function reads(r) {
  const counts = {};
  for (const part of String(r || "").split("|")) { const t = part.split(":")[0]; if (t) counts[t] = (counts[t] || 0) + 1; }
  return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([t, n]) => `${esc(t)}${n > 1 ? ` &times;${n}` : ""}`).join(", ") || "none";
}
function showPlate(it) {
  $("#main").innerHTML = `${notices()}<div class="grid"><div class="card crop"><img class="big" src="${img("plate", it)}" alt="the plate">
    <p class="muted small">Yellow box: the plate &middot; ${ride(it.video)}${it.t !== undefined ? ` at ${clock(it.t)}` : ""}${it.w ? ` &middot; ${it.w} px wide` : ""}</p></div>
  <div class="card"><p>Read: <b>${esc(it.model) || "(nothing confident)"}</b> &middot; <span class="muted">reads: ${reads(it.reads)}</span></p>
  <label>Plate <input id="text" size="12" value="${esc(it.model)}" autocapitalize="characters" spellcheck="false"></label>
  <div class="decide"><div class="choices">${choiceButtons([["read", "save the text", "Enter"], ["unreadable", "unreadable", "X"], ["skip", "skip", "S"]], "read")}</div>
  <div class="actions">${undoButton()}</div></div>
  <p class="hint">T edits the plate, Enter saves it, Esc leaves the field &middot; Z undoes</p></div></div>`;
}
async function decide(v) {
  const it = queue[pos]; if (!it || busy) return;
  if (!$("#reviewer").value.trim()) { $("#reviewer").classList.add("missing"); $("#reviewer").focus(); toast("Enter your name first"); return; }
  const text = $("#text") ? $("#text").value.trim() : "";
  if (it.kind === "plate" && v === "read" && !text) { toast("Type the plate, or X for unreadable"); return; }
  let label = v, out = {kind: it.kind, video: it.video, item: it.item, row: it.row, model: it.model, model_conf: it.model_conf ?? Math.max(...Object.values(it.proba || {0:0})), reviewer: $("#reviewer").value.trim()};
  if (it.kind === "plate") { label = v === "skip" ? "skip" : "plate"; out.text = v === "unreadable" ? "" : text.toUpperCase(); out.model_text = it.model; out.model = "plate"; }
  else if (it.kind === "sign") { out.text = text; out.model_text = it.text; }
  out.label = label; out.time = new Date().toISOString();
  busy = true;
  try {
    if (STATIC) { saved.push(out); keep(); }
    else {
      const r = await (await fetch("/api/label", {method:"POST", body: JSON.stringify(out)})).json();
      if (r.error) throw new Error(r.error);
    }
    history.push({pos, out});
    toast(it.kind === "plate" ? (v === "unreadable" ? "Saved: unreadable" : v === "skip" ? "Skipped" : `Saved: ${out.text}`) : (v === "skip" ? "Skipped" : `Saved: ${name(v)}`));
    pos += 1; show();
  } catch (e) { toast(`Not saved: ${e.message}`); }
  finally { busy = false; }
}
document.addEventListener("keydown", e => {
  if (kind === "progress" || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const t = e.target, typing = t && t.id === "text";
  if (t && (t.tagName === "SELECT" || (t.tagName === "INPUT" && !typing) || t.tagName === "TEXTAREA")) return;
  const k = e.key.toLowerCase();
  if (typing) {
    if (e.key === "Enter") { e.preventDefault(); decide(kind === "plate" ? "read" : queue[pos] && queue[pos].model); }
    else if (e.key === "Escape") t.blur();
    return;
  }
  const it = queue[pos];
  let act = null;  // the key's action; the key itself goes nowhere else (not into a field focused meanwhile)
  if (k === "z") act = undo;
  else if (it && e.key === "Enter") act = () => decide(kind === "plate" ? "read" : (it.current || it.model));
  else if (it && kind !== "vehicle" && k === "t") act = () => $("#text").focus();
  else if (it && k === "s") act = () => decide("skip");
  else if (it && kind === "plate") { if (k === "x") act = () => decide("unreadable"); }
  else if (it && kind === "vehicle") {
    if (k.length === 1 && "12345".includes(k)) act = () => decide(RELS[+k - 1][0]);
    else if (k === "u") act = () => decide("unclear"); else if (k === "n") act = () => decide("not_vehicle");
  }
  else if (it) { const hit = SIGNS.find(([, key]) => key === k); if (hit) act = () => decide(hit[0]); }
  if (act) { e.preventDefault(); act(); }
});
async function undo() {
  const last = history[history.length - 1]; if (!last || busy) return;
  busy = true;
  try {
    if (STATIC) {
      const i = saved.lastIndexOf(saved.find(s => s.time === last.out.time && s.item === last.out.item && s.video === last.out.video));
      if (i >= 0) { saved.splice(i, 1); keep(); }
    } else {
      const r = await (await fetch("/api/undo", {method:"POST", body: JSON.stringify({reviewer: last.out.reviewer,
        expect: {kind: last.out.kind, video: last.out.video, item: last.out.item}})})).json();
      if (!r.ok) throw new Error(r.reason || r.error || "not undone");
    }
    history.pop(); pos = last.pos; toast("Undone"); show();
  } catch (e) { toast(`Could not undo: ${e.message}`); }
  finally { busy = false; }
}
function staticProgress() {
  const kinds = ["vehicle", "sign", "plate"].map(k => `${saved.filter(d => d.kind === k && d.label !== "skip").length} ${k}s`).join(", ");
  const recent = saved.slice(-8).reverse().map(d => `<tr><td>${esc(d.kind)}</td><td>${ride(d.video)}</td><td>${esc(d.kind === "plate" ? (d.text || "unreadable") : name(d.label))}</td><td class="muted">${esc((d.time || "").slice(0, 16).replace("T", " "))}</td></tr>`).join("");
  $("#main").innerHTML = `${notices()}<div class="card"><p>A copy of the review page that needs no server: the queues are as they were when it was made, and
    your decisions stay in this browser - <b>${decided()}</b> so far (${kinds}).</p>
    <p><button class="primary" id="dl3">Download decisions</button> <button id="clear">Clear them</button></p>
    <p class="muted">Send the file to the project; it is added to the labels with <code>python -m bikesafe.review --import decisions.json</code>, then Retrain.</p>
    ${recent ? `<table><tr><th>queue</th><th>ride</th><th>decision</th><th>when</th></tr>${recent}</table>` : ""}</div>`;
  $("#dl3").onclick = download;
  $("#clear").onclick = () => { if (confirm("Delete the decisions kept in this browser?")) { saved = []; keep(); staticProgress(); } };
}
async function progress() {
  if (STATIC) return staticProgress();
  let p;
  try { p = await (await fetch("/api/progress")).json(); } catch (e) { $("#main").innerHTML = `<div class="notice bad">Could not reach the review server.</div>`; return; }
  if (kind !== "progress") return;
  const label = (k, v) => k === "plate" ? (v || "-") : name(v);
  const blocks = Object.entries(p.agreement).map(([k, a]) => `<div class="card"><h3>${esc(k)}</h3><p>${a.decisions} decisions, ${a.unclear} unclear &middot; model kept by the reviewer: <b>${a.model_kept === null ? "-" : Math.round(a.model_kept*100) + "%"}</b></p>
    ${a.confusion.length ? `<table><tr><th>model said</th><th>reviewer said</th><th>n</th></tr>${a.confusion.sort((x,y)=>y.n-x.n).map(c => `<tr><td>${esc(label(k, c.model))}</td><td>${esc(label(k, c.human))}</td><td>${c.n}</td></tr>`).join("")}</table>` : ""}</div>`).join("");
  $("#main").innerHTML = `<div class="card"><p>Vehicle labels by labeller: ${Object.entries(p.track_labels).map(([k,v]) => `${esc(k)} ${v}`).join(", ") || "none"} &middot; sign labels ${p.sign_labels} &middot; plate labels ${p.plate_labels}</p>
    ${p.retrainable ? `<p><button class="primary" id="retrain">Retrain on the labels</button> <span id="rt" class="muted">${esc(p.retrain.state)}: ${esc(p.retrain.log)}</span></p>`
      : `<p>Decisions are saved in <code>${esc(p.labels_dir)}</code>. The models learn from them on the machine with the labelled rides:
      there, <code>python -m bikesafe.review --import &lt;this run's folder&gt;</code> adds this run and its decisions, then Retrain.</p>`}</div>${blocks}`;
  if (p.retrainable) $("#retrain").onclick = async () => { const r = await (await fetch("/api/retrain", {method:"POST", body:"{}"})).json(); $("#rt").textContent = `${r.state}: ${r.log}`; if (r.state === "running") setTimeout(progress, 5000); };
  if (p.retrain.state === "running") setTimeout(() => { if (kind === "progress") progress(); }, 5000);
}
savedBox();
load();
</script></body></html>
"""


def page(static: bool = False) -> str:
    keys = "abcdefghijklmnopqrstuvwxyz"
    shortcuts = [(t, k) for t, k in zip(SIGN_TYPES_UI, (c for c in keys if c not in "stuz"))]
    home = '<a class="plate" href="../">Bike route vehicle typology</a>' if static else ""  # the project page, when published
    return (PAGE.replace("__SIGNS__", json.dumps(shortcuts)).replace("__STATIC__", json.dumps(static))
            .replace("__HOME__", home))


MEDIA = {"vehicle": ["clip", "zoom"], "sign": ["sign", "signctx"], "plate": ["plate"]}


def private_seconds(corpus: Path, video: str) -> set[int]:
    """Seconds of the ride filmed in its privacy zones (gps_1s.csv, bikesafe.gps); for a ride without GPS, its first
    and last NO_GPS_PRIVATE_S, where it starts and ends (usually home or work)."""
    gps = corpus / video / "gps_1s.csv"
    if gps.exists():
        g = pd.read_csv(gps, usecols=["second", "in_privacy_zone"])
        return set(g.second[g.in_privacy_zone.astype(bool)].astype(int))
    timeline = corpus / video / "timeline_1s.csv"
    n = len(pd.read_csv(timeline, usecols=["second"])) if timeline.exists() else 0
    return set(range(0, min(n, NO_GPS_PRIVATE_S))) | set(range(max(0, n - NO_GPS_PRIVATE_S), n))


def item_seconds(it: dict) -> range:
    """The seconds of the ride an item's media show."""
    start = float(it.get("t") or 0.0)
    return range(int(start - CLIP_PAD_S - 1), int(start + float(it.get("duration") or 0.0) + CLIP_PAD_S) + 2)


def export(store: Store, out: Path, per_queue: int) -> None:
    """The page as files that need no server (GitHub Pages, any web host): the first `per_queue` items of each queue
    with their clips and crops, leaving out those filmed in a ride's privacy zones (private_seconds) - the copy is
    public. Decisions made there stay in the viewer's browser until downloaded; --import adds them to the labels."""
    shutil.rmtree(out / "media", ignore_errors=True)
    queues = {"vehicle_new": store.vehicle_queue("new"), "vehicle_verify": store.vehicle_queue("verify"),
              "sign": store.sign_queue(), "plate": store.plate_queue()}
    private: dict[str, set[int]] = {}
    for name, items in queues.items():
        kept, hidden = [], 0
        for it in items:
            if len(kept) >= per_queue:
                break
            zone = private.setdefault(it["video"], private_seconds(store.corpus, it["video"]))
            if any(sec in zone for sec in item_seconds(it)):
                hidden += 1
                continue
            files = {kind: store.image(kind, it["video"], it["item"], it.get("row")) for kind in MEDIA[it["kind"]]}
            if not all(files.values()):
                continue
            for kind, data in files.items():
                path = out / "media" / it["video"] / f"{kind}_{it['item']}.{'mp4' if kind == 'clip' else 'jpg'}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            kept.append(it)
        (out / f"queue_{name}.json").write_text(json.dumps(clean(kept), default=str), encoding="utf-8")
        print(f"{name}: {len(kept)} items ({hidden} left out: privacy zones)", flush=True)
    (out / "index.html").write_text(page(static=True), encoding="utf-8")


VALID_LABELS = {"vehicle": set(RELATIONS) | {"unclear", "not_vehicle", "skip"}, "sign": set(SIGN_TYPES_UI) | {"skip"},
                "plate": {"plate", "skip"}}


def import_decisions(store: Store, path: Path) -> tuple[int, int]:
    """Decisions downloaded from an exported page, saved as if they had been made here (reviewer names and times
    kept). All are checked before any is saved, and a decision already in the review log (same reviewer, time, kind,
    ride and item: the same file imported twice) is skipped. Returns (added, skipped)."""
    decisions = json.loads(path.read_text(encoding="utf-8"))
    unknown = {d["video"] for d in decisions if not store.known_video(d["video"])}
    if unknown:
        raise SystemExit(f"rides not in {store.perception}: {', '.join(sorted(unknown))}")
    bad = [d for d in decisions if d.get("label") not in VALID_LABELS.get(d.get("kind"), set())]
    if bad:
        raise SystemExit(f"{len(bad)} decisions with an unknown kind or label, e.g. {bad[0]}; nothing imported")
    log = store._csv("review_log.csv", REVIEW_LOG_COLUMNS)
    done = set(zip(log.reviewer.astype(str), log.time.astype(str), log.kind.astype(str), log.video.astype(str),
                   log["item"].astype(str)))
    added = 0
    for d in decisions:
        key = (str(d.get("reviewer") or "reviewer")[:40], str(d.get("time")), d["kind"], d["video"], str(d["item"]))
        if key in done:
            continue
        store.save(d)
        done.add(key)
        added += 1
    return added, len(decisions) - added


def import_run(store: Store, run: Path) -> str:
    """A run of the Colab notebook (My Drive/bikesafe_runs/<video>): its ride joins the rides here (perception,
    analysis and results, unless already here) and the decisions reviewed on it join the labels, so that Retrain
    learns from them. Importing a run again replaces its vehicles' labels with the run's."""
    rides = [meta.parent.name for meta in sorted((run / "work" / "perception").glob("*/meta.json"))]
    if not rides:
        raise SystemExit(f"{run} is not a run of the pipeline (no work/perception/<video>/meta.json)")
    for video in rides:
        for source, target in ((run / "work" / "perception" / video, store.perception / video),
                               (run / "work" / "analysis" / video, store.analysis / video),
                               (run / "results" / video, store.corpus / video)):
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
    labels, counts = run / "labels", {}
    theirs = pd.read_csv(labels / "track_labels.csv") if (labels / "track_labels.csv").exists() else pd.DataFrame()
    if len(theirs):
        # card ids are renumbered: the run numbered its cards on its own, and they key the anchors
        their_anchors = load_anchors(labels / "track_anchors.csv")
        mine = store._csv("track_labels.csv", ["card_id", "video", "track_id", "label", "labeler", "notes"])
        anchors = load_anchors(store.labels_dir / "track_anchors.csv")
        replaced = mine.merge(theirs[["video", "track_id"]], on=["video", "track_id"]).card_id
        mine, anchors = mine[~mine.card_id.isin(replaced)], anchors[~anchors.card_id.isin(replaced)]
        for row in theirs.to_dict("records"):
            card = next_card_id(str(row["video"]), mine, anchors)
            anchors = pd.concat([anchors, their_anchors[their_anchors.card_id == row["card_id"]].assign(card_id=card)],
                                ignore_index=True)
            mine = pd.concat([mine, pd.DataFrame([{**row, "card_id": card}])], ignore_index=True)
        mine.to_csv(store.labels_dir / "track_labels.csv", index=False)
        anchors.to_csv(store.labels_dir / "track_anchors.csv", index=False)
        counts["vehicle"] = len(theirs)
    for name, key in (("sign_labels.csv", "object_id"), ("plate_labels.csv", "track_id"), ("review_log.csv", None)):
        if (labels / name).exists():
            new = pd.read_csv(labels / name)
            merged = pd.concat([store._csv(name), new], ignore_index=True)
            merged = merged.drop_duplicates(["video", key], keep="last") if key else merged.drop_duplicates()
            merged.to_csv(store.labels_dir / name, index=False)
            counts[name.split("_")[0]] = len(new)
    decided = ", ".join(f"{n} {kind}" for kind, n in counts.items() if kind != "review") or "no"
    return f"{', '.join(rides)}: ride added, {decided} decisions added to {store.labels_dir}"


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
                        help="Add decisions to the labels: decisions.json downloaded from an exported page, or a Colab "
                             "run's folder (My Drive/bikesafe_runs/<video>), whose ride joins the rides here")
    args = parser.parse_args()
    args.labels.mkdir(parents=True, exist_ok=True)
    store = Store(args.analysis, args.perception, args.corpus, args.videos, labels_dir=args.labels)
    if args.export is not None:
        export(store, args.export, args.per_queue)
        print(f"wrote {args.export / 'index.html'}")
        return
    if args.import_ is not None:
        if args.import_.is_dir():
            print(import_run(store, args.import_))
        else:
            added, skipped = import_decisions(store, args.import_)
            print(f"added {added} decisions to {args.labels}" + (f" ({skipped} already there)" if skipped else ""))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, page()))
    print(f"review at http://127.0.0.1:{args.port}  ({len(store.tracks)} tracks from {args.corpus})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
