"""Licence plates: blur them in everything shown, and (only when asked) use them to check vehicle identities.

Blurring. A small plate detector (YOLOv9-tiny, 7 MB ONNX, from open-image-models; downloaded on first use) runs on
every vehicle box at least MIN_VEHICLE_H px tall, cropped from the full-resolution frame so a plate is large enough
to find. A plate found on a track stays blurred for PLATE_HOLD frames after the detector last saw it, at the same
place relative to the vehicle box, which covers single missed frames. bikesafe.render blurs by default, and

    python -m bikesafe.plates blur demos/clip.mp4 [more videos or images] [--out-dir DIR]

blurs existing videos and images in place (or into DIR), finding vehicles with the project's YOLO11n detector.

Identity check (opt-in).

    python -m bikesafe.plates read work/perception/VID_... --analysis work/analysis --videos D:/rides

reads each vehicle's plate on up to READS_PER_TRACK frames where the vehicle is large, with a small OCR model
(fast-plate-ocr, global model including US plates). Plate text never leaves the process: a read becomes a keyed hash
(HMAC-SHA256 under a random key made for that run and never stored), so hashes can be compared only within one ride,
cannot be turned back into a plate, and cannot link a vehicle across rides. Only hashes and counts are written, to
work/analysis/<video>/plates.parquet, which stays on the processing machine. bikesafe.tracks then audits the vehicle
ids from bikesafe.stitch with them (identity_audit): two tracks with one plate should share a vehicle_id unless the
vehicle left the view and came back, and one vehicle_id should not carry two different plates.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import ROOT, VEHICLE_CLASSES, open_video, read_json, resolve_video

PLATE_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-s-v2-global-model"
PLATE_CONF = 0.25
MIN_VEHICLE_H = 40  # px; a plate on a smaller vehicle box is a few pixels wide and unreadable anyway
CROP_MARGIN = 0.1
BLUR_PAD = 0.3  # the blurred area extends the plate box by this fraction on each side
PLATE_HOLD = 6  # frames a plate stays blurred after its last detection on the track
PLATE_REL_U = (0.15, 0.85)  # plausible plate centre across its vehicle's box
PLATE_REL_V = (0.3, 1.0)  # and down it (plates are mounted low)
MIN_READ_VEHICLE_H = 80
READS_PER_TRACK = 6
READ_CONF = 0.8  # mean per-character confidence of a kept read
MIN_AGREEING_READS = 2
KEY_HEX = 16
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PROVIDERS = ["CPUExecutionProvider"]  # the models are tiny; CPU avoids optional runtime providers


class PlateFinder:
    """Plate boxes on the vehicle boxes of one frame."""

    def __init__(self, model: str = PLATE_MODEL, conf: float = PLATE_CONF) -> None:
        try:
            from open_image_models import create_detector
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise RuntimeError("licence-plate blurring needs `pip install open-image-models onnxruntime` "
                               "(or render with --no-blur)") from error

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


def blur_region(image: np.ndarray, box: np.ndarray, pad: float = BLUR_PAD) -> None:
    """Pixelate and blur a padded box in place (irreversible at the sizes involved)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ix1, iy1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    ix2, iy2 = int(min(image.shape[1], x2 + pad * w)), int(min(image.shape[0], y2 + pad * h))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return
    region = image[iy1:iy2, ix1:ix2]
    small = cv2.resize(region, (max(1, (ix2 - ix1) // 8), max(1, (iy2 - iy1) // 8)), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (ix2 - ix1, iy2 - iy1), interpolation=cv2.INTER_NEAREST)
    k = max(3, (min(ix2 - ix1, iy2 - iy1) // 2) | 1)
    image[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(coarse, (k, k), 0)


class PlateBlurrer:
    """Blurs plates frame by frame, remembering where each track's plate was for PLATE_HOLD frames."""

    def __init__(self, finder: PlateFinder | None = None) -> None:
        self.finder = finder or PlateFinder()
        self.memory: dict[int, tuple[np.ndarray, int]] = {}  # track id -> (plate box relative to vehicle box, age)
        self.plates_found = 0
        self.plates_held = 0

    def __call__(self, image: np.ndarray, boxes: np.ndarray, track_ids=None) -> np.ndarray:
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        ids = list(track_ids) if track_ids is not None else [None] * len(boxes)
        seen = set()
        for i, plate, _ in self.finder.find(image, boxes):
            blur_region(image, plate)
            self.plates_found += 1
            if ids[i] is not None:
                x1, y1, x2, y2 = boxes[i]
                w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                self.memory[ids[i]] = (np.array([(plate[0] - x1) / w, (plate[1] - y1) / h,
                                                 (plate[2] - x1) / w, (plate[3] - y1) / h]), 0)
                seen.add(ids[i])
        for i, tid in enumerate(ids):
            if tid is None or tid in seen or tid not in self.memory:
                continue
            rel, age = self.memory[tid]
            if age >= PLATE_HOLD:
                del self.memory[tid]
                continue
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            blur_region(image, np.array([x1 + rel[0] * w, y1 + rel[1] * h, x1 + rel[2] * w, y1 + rel[3] * h]))
            self.memory[tid] = (rel, age + 1)
            self.plates_held += 1
        return image


def _vehicle_model():
    from ultralytics import YOLO

    return YOLO(str(ROOT / "models" / "yolo11n.pt"))


def blur_image_file(path: Path, out: Path, model=None, finder: PlateFinder | None = None) -> int:
    model = model or _vehicle_model()
    image = cv2.imread(str(path))
    result = model.predict(image, imgsz=1280, conf=0.10, classes=list(VEHICLE_CLASSES), verbose=False)[0]
    blurrer = PlateBlurrer(finder)
    blurrer(image, result.boxes.xyxy.cpu().numpy())
    cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 92] if out.suffix.lower() in (".jpg", ".jpeg") else [])
    return blurrer.plates_found


def blur_video_file(path: Path, out: Path, model=None, finder: PlateFinder | None = None) -> tuple[int, int]:
    """Blur plates in every frame; vehicles and their ids come from YOLO11n + the tuned tracker."""
    model = model or _vehicle_model()
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    tmp = out.with_name(out.stem + ".blurring" + out.suffix)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    blurrer = PlateBlurrer(finder)
    tracker = str(ROOT / "configs" / "botsort_vehicle_tuned.yaml")
    frames = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        result = model.track(image, persist=True, tracker=tracker, imgsz=1280, conf=0.10,
                             classes=list(VEHICLE_CLASSES), verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else None
        writer.write(blurrer(image, boxes, ids))
        frames += 1
    cap.release()
    writer.release()
    os.replace(tmp, out)
    return frames, blurrer.plates_found + blurrer.plates_held


# ----------------------------------------------------------------------------------------------- identity check

def plate_key(text: str, key: bytes) -> str:
    return hmac.new(key, text.encode("ascii"), hashlib.sha256).hexdigest()[:KEY_HEX]


def normalise(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def frames_to_read(dets: pd.DataFrame) -> pd.DataFrame:
    """Up to READS_PER_TRACK detections per track where the vehicle is large, spread over the track."""
    big = dets[(dets.y2 - dets.y1) >= MIN_READ_VEHICLE_H]
    picks = []
    for _, t in big.groupby("track_id"):
        t = t.sort_values("frame")
        picks.append(t.iloc[np.unique(np.linspace(0, len(t) - 1, min(READS_PER_TRACK, len(t))).round().astype(int))])
    return pd.concat(picks) if picks else big.iloc[:0]


class PlateReader:
    def __init__(self, finder: PlateFinder | None = None, ocr_model: str = OCR_MODEL) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        logging.getLogger("fast_plate_ocr").setLevel(logging.WARNING)
        self.finder = finder or PlateFinder()
        self.ocr = LicensePlateRecognizer(ocr_model, device="cpu", providers=PROVIDERS)

    def read(self, image: np.ndarray, boxes: np.ndarray, index: int) -> tuple[bool, str | None]:
        """(a plate of vehicle `index` was found, its text if read confidently). `boxes` are all vehicle boxes of the
        frame, so a neighbour's plate showing in this vehicle's crop is not taken for its own. Text is for hashing."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        found = [(plate, conf) for _, plate, conf in self.finder.find(image, boxes[index:index + 1])
                 if plate_owner(plate, boxes) == index]
        if not found:
            return False, None
        plate, _ = max(found, key=lambda item: item[1])
        x1, y1, x2, y2 = plate.astype(int)
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return True, None
        prediction = self.ocr.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_confidence=True)[0]
        text = normalise(prediction.plate)
        confidence = float(np.mean(prediction.char_probs[: max(len(prediction.plate), 1)]))
        if confidence < READ_CONF or not 4 <= len(text) <= 8:
            return True, None
        return True, text


def read_ride(perception_dir: Path, analysis_root: Path, videos: Path, reader: PlateReader | None = None) -> Path:
    meta = read_json(perception_dir / "meta.json")
    dets = pd.read_parquet(perception_dir / "detections.parquet")
    dets = dets[dets.track_id >= 0]
    wanted = frames_to_read(dets)
    by_frame = {int(f): g for f, g in wanted.groupby("frame")}
    all_boxes = {int(f): g for f, g in dets[dets.frame.isin(by_frame)].groupby("frame")}
    reader = reader or PlateReader()
    key = os.urandom(32)  # this run only; never written anywhere
    found: Counter = Counter()
    reads: dict[int, list[str]] = {}
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
            has_plate, text = reader.read(image, boxes, position[int(det.track_id)])
            found[int(det.track_id)] += int(has_plate)
            if text:
                reads.setdefault(int(det.track_id), []).append(plate_key(text, key))
    cap.release()
    rows = []
    for tid in sorted(set(wanted.track_id.astype(int))):
        keys = Counter(reads.get(tid, []))
        best, agree = keys.most_common(1)[0] if keys else (None, 0)
        rows.append({"track_id": tid, "frames_checked": int((wanted.track_id == tid).sum()), "plates_found": found[tid],
                     "reads": int(sum(keys.values())), "agreeing_reads": int(agree),
                     "plate_key": best if agree >= MIN_AGREEING_READS else None})
    out_dir = analysis_root / meta["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "plates.parquet"
    pd.DataFrame(rows, columns=["track_id", "frames_checked", "plates_found", "reads", "agreeing_reads",
                                "plate_key"]).to_parquet(out, index=False)
    return out


def identity_audit(plates: pd.DataFrame, vehicle_of: dict[int, int], spans: pd.DataFrame,
                   rejoin_gap_s: float = 2.0) -> dict:
    """Check vehicle ids against plate keys. `spans` is indexed by track_id with t_start / t_end columns."""
    keyed = plates.dropna(subset=["plate_key"])
    keyed = keyed.assign(vehicle_id=keyed.track_id.map(vehicle_of))
    audit = {"tracks_with_plate_key": int(len(keyed)), "vehicles_with_plate_key": int(keyed.vehicle_id.nunique()),
             "vehicles_with_two_plates": 0, "same_plate_joined_pairs": 0, "same_plate_split_pairs": 0,
             "same_plate_split_within_gap": 0, "vehicles_seen_again_later": 0}
    audit["vehicles_with_two_plates"] = int((keyed.groupby("vehicle_id").plate_key.nunique() > 1).sum())
    for _, group in keyed.groupby("plate_key"):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    blur = sub.add_parser("blur", help="Blur plates in existing videos / images")
    blur.add_argument("paths", nargs="+", type=Path)
    blur.add_argument("--out-dir", type=Path, default=None, help="Write here instead of replacing the files")
    read = sub.add_parser("read", help="Opt-in: plate keys per track for the identity audit (text is never stored)")
    read.add_argument("perception_dirs", nargs="+", type=Path)
    read.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    read.add_argument("--videos", type=Path, default=ROOT / "videos")
    read.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command == "blur":
        model, finder = _vehicle_model(), PlateFinder()
        for path in args.paths:
            out = (args.out_dir / path.name) if args.out_dir else path
            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                print(f"{path}: {blur_image_file(path, out, model, finder)} plates blurred")
            else:  # a fresh tracker per video, so ids do not carry over between files
                frames, plates = blur_video_file(path, out, _vehicle_model(), finder)
                print(f"{path}: {frames} frames, {plates} plate regions blurred")
        return
    reader = PlateReader()
    for path in args.perception_dirs:
        if (args.analysis / path.name / "plates.parquet").exists() and not args.overwrite:
            print(f"skip {path.name}: plates.parquet exists")
            continue
        out = read_ride(path, args.analysis, args.videos, reader)
        table = pd.read_parquet(out)
        print(f"{path.name}: {len(table)} tracks checked, {int((table.plates_found > 0).sum())} with a plate found, "
              f"{int(table.plate_key.notna().sum())} with a consistent read")


if __name__ == "__main__":
    main()
