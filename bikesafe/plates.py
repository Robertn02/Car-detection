"""Licence plates: find them on the vehicle boxes and read them as text.

A small plate detector (YOLOv9-tiny, 7 MB ONNX, from open-image-models; downloaded on first use) runs on every vehicle
box at least MIN_VEHICLE_H px tall, cropped from the full-resolution frame so a plate is large enough to find, and a
small OCR model (fast-plate-ocr, global model including US plates) reads it.

    python -m bikesafe.plates work/perception/VID_... --analysis work/analysis --videos D:/rides

reads each track's plate on up to READS_PER_TRACK frames - the largest view of the vehicle in each part of the track -
and writes work/analysis/<video>/plates.parquet: the consensus text per track (majority over the confident reads,
character by character when they disagree), how many reads agree, and every read. bikesafe.exposure puts the plate on
each vehicle in tracks_typed.csv and checks the vehicle ids from bikesafe.stitch against it (identity_audit): two
tracks with one plate should share a vehicle_id unless the vehicle left the view and came back, and one vehicle_id
should not carry two different plates. bikesafe.render draws each plate's box and text on the overlay.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import ROOT, VIDEOS, open_video, read_json, resolve_video

PLATE_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-s-v2-global-model"
PLATE_CONF = 0.25
MIN_VEHICLE_H = 40  # px; a plate on a smaller vehicle box is a few pixels wide and unreadable anyway
CROP_MARGIN = 0.1
PLATE_REL_U = (0.15, 0.85)  # plausible plate centre across its vehicle's box
PLATE_REL_V = (0.3, 1.0)  # and down it (plates are mounted low)
MIN_READ_VEHICLE_H = 60
READS_PER_TRACK = 10
READ_CONF = 0.75  # mean per-character confidence of a kept read
SURE_READ_CONF = 0.92  # a single read this confident stands on its own
MIN_AGREEING_READS = 2
PLATE_LEN = (4, 8)
PROVIDERS = ["CPUExecutionProvider"]  # the models are tiny; CPU avoids optional runtime providers


class PlateFinder:
    """Plate boxes on the vehicle boxes of one frame."""

    def __init__(self, model: str = PLATE_MODEL, conf: float = PLATE_CONF) -> None:
        try:
            from open_image_models.detection.factory import create_detector
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise RuntimeError("licence plates need `pip install open-image-models onnxruntime`") from error

        logging.getLogger("open_image_models").setLevel(logging.WARNING)
        self.detector = create_detector(model, conf_thresh=conf, providers=PROVIDERS)

    def find(self, image: np.ndarray, boxes: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """(vehicle index, plate box x1 y1 x2 y2 in image px, confidence) for every plate found."""
        height, width = image.shape[:2]
        out = []
        for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, float).reshape(-1, 4)):
            w, h = x2 - x1, y2 - y1
            if h < MIN_VEHICLE_H:
                continue
            cx1, cy1 = int(max(0, x1 - CROP_MARGIN * w)), int(max(0, y1 - CROP_MARGIN * h))
            cx2, cy2 = int(min(width, x2 + CROP_MARGIN * w)), int(min(height, y2 + CROP_MARGIN * h))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            for result in self.detector.predict(image[cy1:cy2, cx1:cx2]):
                b = result.bounding_box
                out.append((i, np.array([cx1 + b.x1, cy1 + b.y1, cx1 + b.x2, cy1 + b.y2], float), float(result.confidence)))
        return out


def plate_owner(plate: np.ndarray, boxes: np.ndarray) -> int | None:
    """Index of the vehicle box a plate belongs to, or None. A plate sits in the lower middle of its own vehicle's
    box; a crop around one vehicle often also shows the plate of the vehicle beside or in front of it, so the plate is
    given to the box in which it is most central, and to none if no box has it in a plausible place."""
    cu, cv = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    best, best_offset = None, np.inf
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, float).reshape(-1, 4)):
        rel_u, rel_v = (cu - x1) / max(x2 - x1, 1.0), (cv - y1) / max(y2 - y1, 1.0)
        if PLATE_REL_U[0] <= rel_u <= PLATE_REL_U[1] and PLATE_REL_V[0] <= rel_v <= PLATE_REL_V[1]:
            if abs(rel_u - 0.5) < best_offset:
                best, best_offset = i, abs(rel_u - 0.5)
    return best


def normalise(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def consensus(reads: list[tuple[str, float]]) -> tuple[str | None, int]:
    """Plate text from several (text, confidence) reads of one vehicle, and how many reads agree with it.

    The most frequent text wins (ties by summed confidence). When no text was read twice, reads of the most common
    length vote character by character, weighted by confidence, which recovers a plate whose reads each got one
    character wrong. A single read stands only if it is very confident."""
    if not reads:
        return None, 0
    counts = Counter(text for text, _ in reads)
    weight = Counter()
    for text, conf in reads:
        weight[text] += conf
    best = max(counts, key=lambda t: (counts[t], weight[t]))
    if counts[best] >= MIN_AGREEING_READS:
        return best, counts[best]
    length = Counter(len(text) for text, _ in reads).most_common(1)[0][0]
    same = [(text, conf) for text, conf in reads if len(text) == length]
    if len(same) >= MIN_AGREEING_READS + 1:
        voted = "".join(max({t[i] for t, _ in same}, key=lambda ch: sum(c for t, c in same if t[i] == ch))
                        for i in range(length))
        support = sum(sum(t[i] == voted[i] for i in range(length)) >= length - 1 for t, _ in same)
        if support >= MIN_AGREEING_READS:
            return voted, support
    sure = [(text, conf) for text, conf in reads if conf >= SURE_READ_CONF]
    if sure:
        return max(sure, key=lambda r: r[1])[0], 1
    return None, 0


def frames_to_read(dets: pd.DataFrame) -> pd.DataFrame:
    """Up to READS_PER_TRACK detections per track where the vehicle is large: the track is cut into that many parts
    and the largest box of each part is read, so the reads are both close up and spread over the track."""
    big = dets[(dets.y2 - dets.y1) >= MIN_READ_VEHICLE_H]
    picks = []
    for _, t in big.groupby("track_id"):
        t = t.sort_values("frame")
        part = np.minimum(np.arange(len(t)) * READS_PER_TRACK // max(len(t), 1), READS_PER_TRACK - 1)
        heights = (t.y2 - t.y1).to_numpy()
        picks.append(t.iloc[[int(np.flatnonzero(part == k)[np.argmax(heights[part == k])]) for k in np.unique(part)]])
    return pd.concat(picks) if picks else big.iloc[:0]


class PlateReader:
    def __init__(self, finder: PlateFinder | None = None, ocr_model: str = OCR_MODEL) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        logging.getLogger("fast_plate_ocr").setLevel(logging.WARNING)
        self.finder = finder or PlateFinder()
        self.ocr = LicensePlateRecognizer(ocr_model, device="cpu", providers=PROVIDERS)

    def read_crop(self, image: np.ndarray, plate: np.ndarray) -> tuple[str | None, float]:
        """Text and mean character confidence of the plate at `plate` (x1 y1 x2 y2), or (None, conf) if unreadable."""
        x1, y1, x2, y2 = plate.astype(int)
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None, 0.0
        prediction = self.ocr.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_confidence=True)[0]
        text = normalise(prediction.plate)
        probs = prediction.char_probs if prediction.char_probs is not None else np.zeros(1)
        confidence = float(np.mean(probs[: max(len(prediction.plate), 1)]))
        if confidence < READ_CONF or not PLATE_LEN[0] <= len(text) <= PLATE_LEN[1]:
            return None, confidence
        return text, confidence

    def read(self, image: np.ndarray, boxes: np.ndarray, index: int) -> tuple[np.ndarray | None, str | None, float]:
        """(plate box or None, text if read confidently, confidence) for vehicle `index`. `boxes` are all vehicle
        boxes of the frame, so a neighbour's plate showing in this vehicle's crop is not taken for its own."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        found = [(plate, conf) for _, plate, conf in self.finder.find(image, boxes[index:index + 1])
                 if plate_owner(plate, boxes) == index]
        if not found:
            return None, None, 0.0
        plate, _ = max(found, key=lambda item: item[1])
        text, confidence = self.read_crop(image, plate)
        return plate, text, confidence


def read_ride(perception_dir: Path, analysis_root: Path, videos: Path, reader: PlateReader | None = None) -> Path:
    meta = read_json(perception_dir / "meta.json")
    dets = pd.read_parquet(perception_dir / "detections.parquet")
    dets = dets[dets.track_id >= 0]
    wanted = frames_to_read(dets)
    by_frame = {int(f): g for f, g in wanted.groupby("frame")}
    all_boxes = {int(f): g for f, g in dets[dets.frame.isin(by_frame)].groupby("frame")}
    reader = reader or PlateReader()
    found: Counter = Counter()
    reads: dict[int, list[tuple[str, float]]] = {}
    best: dict[int, tuple] = {}  # track -> (rank, frame, plate box) of its best plate view, for bikesafe.review
    cap = open_video(resolve_video(meta, videos))
    targets = sorted(by_frame)
    index = 0
    if targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
        index = targets[0]
    for target in targets:
        while index < target:
            cap.grab()
            index += 1
        ok, image = cap.read()
        index += 1
        if not ok:
            break
        frame_dets = all_boxes[target]
        boxes = frame_dets[["x1", "y1", "x2", "y2"]].to_numpy(float)
        position = {tid: i for i, tid in enumerate(frame_dets.track_id.astype(int))}
        for det in by_frame[target].itertuples():
            plate, text, confidence = reader.read(image, boxes, position[int(det.track_id)])
            tid = int(det.track_id)
            found[tid] += int(plate is not None)
            if text:
                reads.setdefault(tid, []).append((text, confidence))
            if plate is not None:
                rank = (text is not None, confidence, float((plate[2] - plate[0]) * (plate[3] - plate[1])))
                if tid not in best or rank > best[tid][0]:
                    best[tid] = (rank, target, plate)
    cap.release()
    rows = []
    for tid in sorted(set(wanted.track_id.astype(int))):
        track_reads = reads.get(tid, [])
        text, agree = consensus(track_reads)
        confs = [conf for t, conf in track_reads if t == text]
        view = best.get(tid)
        box = view[2] if view else np.full(4, np.nan)
        rows.append({"track_id": tid, "frames_checked": int((wanted.track_id == tid).sum()), "plates_found": found[tid],
                     "reads": len(track_reads), "agreeing_reads": int(agree), "plate_text": text,
                     "plate_conf": round(float(np.mean(confs)), 3) if confs else np.nan,
                     "all_reads": "|".join(f"{t}:{c:.2f}" for t, c in track_reads),
                     "plate_frame": view[1] if view else np.nan,
                     **{k: round(float(v), 1) for k, v in zip(("plate_x1", "plate_y1", "plate_x2", "plate_y2"), box)}})
    out_dir = analysis_root / meta["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "plates.parquet"
    pd.DataFrame(rows, columns=["track_id", "frames_checked", "plates_found", "reads", "agreeing_reads", "plate_text",
                                "plate_conf", "all_reads", "plate_frame", "plate_x1", "plate_y1", "plate_x2",
                                "plate_y2"]).to_parquet(out, index=False)
    return out


def plates_on_pieces(plates: pd.DataFrame, kin: pd.DataFrame) -> pd.DataFrame:
    """plates.parquet keys tracks by the tracker's id; where bikesafe.identity cut a track, its plate goes to the piece
    that held its best plate view (plate_frame). Older tables without plate_frame keep the tracker's id (which the
    first piece of every track keeps)."""
    if plates.empty or "raw_track_id" not in kin or "plate_frame" not in plates:
        return plates
    from bikesafe.identity import remap_ids

    id_map = kin[["frame", "raw_track_id", "track_id"]]
    has_view = plates.plate_frame.notna()
    mapped = remap_ids(plates[has_view].assign(frame=plates.plate_frame[has_view].astype(np.int64)), id_map)
    return pd.concat([pd.DataFrame(mapped.drop(columns="frame")), pd.DataFrame(plates[~has_view])], ignore_index=True)


def vehicle_plates(plates: pd.DataFrame, vehicle_of: dict[int, int]) -> pd.Series:
    """One plate per vehicle id: the text read on most of its tracks' agreeing reads (index vehicle_id)."""
    keyed = plates.dropna(subset=["plate_text"])
    if keyed.empty:
        return pd.Series(dtype=object)
    keyed = keyed.assign(vehicle_id=keyed.track_id.map(vehicle_of))
    votes = keyed.groupby(["vehicle_id", "plate_text"]).agreeing_reads.sum().reset_index()
    best = votes.sort_values("agreeing_reads").drop_duplicates("vehicle_id", keep="last")
    return best.set_index("vehicle_id").plate_text


def identity_audit(plates: pd.DataFrame, vehicle_of: dict[int, int], spans: pd.DataFrame,
                   rejoin_gap_s: float = 2.0) -> dict:
    """Check vehicle ids against plates. `spans` is indexed by track_id with t_start / t_end columns."""
    keyed = plates.dropna(subset=["plate_text"])
    keyed = keyed.assign(vehicle_id=keyed.track_id.map(vehicle_of))
    audit = {"tracks_with_plate": int(len(keyed)), "vehicles_with_plate": int(keyed.vehicle_id.nunique()),
             "vehicles_with_two_plates": 0, "same_plate_joined_pairs": 0, "same_plate_split_pairs": 0,
             "same_plate_split_within_gap": 0, "vehicles_seen_again_later": 0}
    audit["vehicles_with_two_plates"] = int((keyed.groupby("vehicle_id").plate_text.nunique() > 1).sum())
    for _, group in keyed.groupby("plate_text"):
        if len(group) < 2:
            continue
        rows = group.sort_values("track_id").to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["vehicle_id"] == b["vehicle_id"]:
                    audit["same_plate_joined_pairs"] += 1
                    continue
                audit["same_plate_split_pairs"] += 1
                sa, sb = spans.loc[a["track_id"]], spans.loc[b["track_id"]]
                gap = max(sb.t_start - sa.t_end, sa.t_start - sb.t_end)
                if gap <= rejoin_gap_s:
                    audit["same_plate_split_within_gap"] += 1
        vehicles = group.vehicle_id.unique()
        if len(vehicles) > 1:
            starts = group.groupby("vehicle_id").track_id.apply(lambda ids: spans.loc[list(ids), "t_start"].min())
            ends = group.groupby("vehicle_id").track_id.apply(lambda ids: spans.loc[list(ids), "t_end"].max())
            order = starts.sort_values().index
            if any(starts[order[k + 1]] - ends[order[k]] > rejoin_gap_s for k in range(len(order) - 1)):
                audit["vehicles_seen_again_later"] += 1
    return audit


class PlateAnnotator:
    """Draws each readable plate's box and text on overlay frames: the track's consensus text from plates.parquet
    when there is one, otherwise a confident read of this frame. Plates that cannot be read are not boxed.

    Plates are looked for on the frames the vehicles were detected on (find=True). On the frames in between (find=False)
    each plate found on the last such frame is drawn at the same place within its vehicle's (interpolated) box - the
    plate finder is most of the time an overlay clip takes to render."""

    def __init__(self, known: dict[int, str] | None = None, reader: PlateReader | None = None) -> None:
        self.known = known or {}
        self.reader = reader or PlateReader()
        self.plates_drawn = 0
        self.last: dict[int, tuple[np.ndarray, str]] = {}  # track id -> plate box relative to its vehicle box, text

    def __call__(self, image: np.ndarray, boxes: np.ndarray, track_ids: list[int], scale: float = 1.0,
                 find: bool = True) -> None:
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if find:
            self.last = {}
            for i, plate, _ in self.reader.finder.find(image, boxes):
                if plate_owner(plate, boxes) != i:
                    continue
                text = self.known.get(int(track_ids[i]))
                if text is None:
                    text, _ = self.reader.read_crop(image, plate)
                if text:
                    size = np.maximum(boxes[i, 2:] - boxes[i, :2], 1.0)
                    self.last[int(track_ids[i])] = ((plate - np.tile(boxes[i, :2], 2)) / np.tile(size, 2), text)
        for box, track_id in zip(boxes, track_ids):
            if int(track_id) in self.last:
                relative, text = self.last[int(track_id)]
                self._draw(image, np.tile(box[:2], 2) + relative * np.tile(box[2:] - box[:2], 2), text, scale)

    def _draw(self, image: np.ndarray, plate: np.ndarray, text: str, scale: float) -> None:
        thick = max(2, int(round(2 / scale)))
        font = 0.6 / scale
        x1, y1, x2, y2 = plate.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), thick)
        self.plates_drawn += 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        top = min(image.shape[0] - 2, y2 + th + 8)
        cv2.rectangle(image, (x1, top - th - 6), (x1 + tw + 6, top + 2), (0, 0, 0), -1)
        cv2.putText(image, text, (x1 + 3, top - 2), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 255), thick, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("perception_dirs", nargs="+", type=Path)
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reader = PlateReader()
    for path in args.perception_dirs:
        if (args.analysis / path.name / "plates.parquet").exists() and not args.overwrite:
            print(f"skip {path.name}: plates.parquet exists")
            continue
        out = read_ride(path, args.analysis, args.videos, reader)
        table = pd.read_parquet(out)
        print(f"{path.name}: {len(table)} tracks checked, {int((table.plates_found > 0).sum())} with a plate found, "
              f"{int(table.plate_text.notna().sum())} read")


if __name__ == "__main__":
    main()
