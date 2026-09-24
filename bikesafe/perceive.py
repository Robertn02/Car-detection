"""GPU perception pass over ride videos.

For every processed frame (default ~12 fps) this stores:
  * vehicle detections with tuned BoT-SORT track IDs,
  * dense-optical-flow measurements for each detection (body motion, body expansion, ground motion just below it),
  * a coarse 16x9 background flow grid (vehicles masked out) used later for ego-motion and horizon estimates,
and once per second CLIP embeddings of the whole frame, the road ahead, and each sizeable vehicle crop.

Everything downstream (geometry, typology, exposure metrics) is CPU-only and reads these files, so the
expensive pass runs once per video. Re-running skips videos whose meta.json is marked complete.

    python -m bikesafe.perceive videos --out work/perception
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from bikesafe.common import (ROOT, VEHICLE_CLASSES, fp16_kwargs, is_cuda, list_videos, open_video, probe_video, read_json,
                             write_json)

FLOW_W, FLOW_H = 480, 270
GRID_W, GRID_H = 16, 9
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
MIN_CROP_HEIGHT = 40
MAX_CROPS_PER_SECOND = 6


def frame_reader(path: Path, stride: int, start: int, end: int, out: queue.Queue) -> None:
    """Decode, downscale, and compute dense flow off the main thread (OpenCV releases the GIL)."""
    cap = open_video(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev_small = None
    idx = start
    while idx < end:
        if (idx - start) % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.cvtColor(cv2.resize(frame, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            # Flow on the current grid pointing back to the previous frame, negated: motion prev->current at current pixels.
            flow = None if prev_small is None else -dis.calc(small, prev_small, None)
            prev_small = small
            out.put((idx, frame, flow, flow_affine(flow, frame.shape[1], frame.shape[0])))
        elif not cap.grab():
            break
        idx += 1
    cap.release()
    out.put(None)


def flow_affine(flow: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Camera-motion similarity transform (previous -> current, full-res px) fitted with RANSAC to sampled dense flow.

    Replaces BoT-SORT's own sparse optical-flow GMC, which would recompute features on the full frame.
    """
    if flow is None:
        return np.eye(2, 3)
    sx, sy = width / FLOW_W, height / FLOW_H
    ys, xs = np.mgrid[6:FLOW_H:12, 6:FLOW_W:12]
    sampled = flow[ys, xs]
    curr = np.stack([xs * sx, ys * sy], axis=-1).reshape(-1, 2).astype(np.float32)
    prev = curr - (sampled * np.array([sx, sy], dtype=np.float32)).reshape(-1, 2)
    warp, _ = cv2.estimateAffinePartial2D(prev, curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return np.eye(2, 3) if warp is None else warp


def scaled_tracker_config(base: Path, fps: float, buffer_s: float, out_dir: Path) -> Path:
    """Ultralytics counts track_buffer in processed frames, so rescale the tuned 2 s buffer to the sampling rate."""
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["track_buffer"] = max(5, round(buffer_s * fps))
    path = out_dir / "tracker.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


class ClipEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str, fp16: bool = True) -> None:
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.eval().to(device)
        self.device = device
        self.mean, self.std = CLIP_MEAN.to(device), CLIP_STD.to(device)
        # embeddings are stored as float16 anyway, so autocast costs nothing in what is kept
        self.autocast = fp16 and is_cuda(device)

    @torch.inference_mode()
    def encode(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([cv2.cvtColor(cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for im in images_bgr])
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.autocast):
            feats = self.model.encode_image((x - self.mean) / self.std)
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float16)


def detection_flow(flow: np.ndarray, mask: np.ndarray, box: np.ndarray, sx: float, sy: float) -> tuple[float, ...]:
    """Body displacement, body isotropic expansion, and ground displacement just below the box (full-res px/step)."""
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bw, bh = x2 - x1, y2 - y1
    nan = float("nan")
    obj_dx = obj_dy = obj_scale = nan
    ix1, ix2 = int(round(x1 + 0.15 * bw)), int(round(x2 - 0.15 * bw))
    iy1, iy2 = int(round(y1 + 0.15 * bh)), int(round(y2 - 0.15 * bh))
    ix1, iy1 = max(ix1, 0), max(iy1, 0)
    ix2, iy2 = min(ix2, FLOW_W), min(iy2, FLOW_H)
    obj_n = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if obj_n >= 9:
        region = flow[iy1:iy2, ix1:ix2]
        dx, dy = region[..., 0].ravel(), region[..., 1].ravel()
        obj_dx, obj_dy = float(np.median(dx)) / sx, float(np.median(dy)) / sy
        if obj_n >= 36:
            gx, gy = np.meshgrid(np.arange(ix1, ix2, dtype=np.float32), np.arange(iy1, iy2, dtype=np.float32))
            gx, gy = gx.ravel() - gx.mean(), gy.ravel() - gy.mean()
            denom = float((gx * gx).sum() + (gy * gy).sum())
            if denom > 0:
                obj_scale = float((gx * (dx - dx.mean())).sum() + (gy * (dy - dy.mean())).sum()) / denom
    gy1 = int(round(y2))
    gy2 = min(FLOW_H, gy1 + max(3, int(round(0.3 * bh))))
    gx1, gx2 = max(0, int(round(x1))), min(FLOW_W, int(round(x2)))
    bg_dx = bg_dy = nan
    bg_n = 0
    if gy1 < FLOW_H and gx2 > gx1 and gy2 > gy1:
        valid = ~mask[gy1:gy2, gx1:gx2]
        bg_n = int(valid.sum())
        if bg_n >= 6:
            patch = flow[gy1:gy2, gx1:gx2][valid]
            bg_dx, bg_dy = float(np.median(patch[:, 0])) / sx, float(np.median(patch[:, 1])) / sy
    return obj_dx, obj_dy, obj_scale, float(obj_n), bg_dx, bg_dy, float(bg_n)


def flow_grid(flow: np.ndarray, mask: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Mean background flow per 16x9 cell in full-res px; NaN where vehicles cover >70% of a cell."""
    ch, cw = FLOW_H // GRID_H, FLOW_W // GRID_W
    valid = (~mask[: ch * GRID_H, : cw * GRID_W]).astype(np.float32)
    f = flow[: ch * GRID_H, : cw * GRID_W] * valid[..., None]
    sums = f.reshape(GRID_H, ch, GRID_W, cw, 2).sum(axis=(1, 3))
    counts = valid.reshape(GRID_H, ch, GRID_W, cw).sum(axis=(1, 3))
    with np.errstate(all="ignore"):
        grid = sums / counts[..., None]
    grid[counts < 0.3 * ch * cw] = np.nan
    grid[..., 0] /= sx
    grid[..., 1] /= sy
    return grid.astype(np.float32)


def process_video(path: Path, args: argparse.Namespace, clip: ClipEncoder | None) -> None:
    info = probe_video(path)
    out_dir = args.out / info.stem
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and read_json(meta_path).get("complete") and not args.overwrite:
        print(f"skip {info.stem}: already complete")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, round(info.fps / args.target_fps))
    proc_fps = info.fps / stride
    start = int(args.start_s * info.fps) if args.start_s else 0
    end = min(info.frames, int(args.end_s * info.fps)) if args.end_s else info.frames
    tracker_cfg = scaled_tracker_config(args.tracker, proc_fps, args.track_buffer_s, out_dir)
    model = YOLO(str(args.model))
    classes = list(VEHICLE_CLASSES)
    precision = {} if args.fp32 else fp16_kwargs(args.device)

    frames_q: queue.Queue = queue.Queue(maxsize=16)
    reader = threading.Thread(target=frame_reader, args=(path, stride, start, end, frames_q), daemon=True)
    reader.start()

    sx, sy = FLOW_W / info.width, FLOW_H / info.height
    warp_state = {"warp": np.eye(2, 3)}
    det_rows: list[tuple] = []
    frame_idx: list[int] = []
    grids: list[np.ndarray] = []
    clip_frames: list[int] = []
    clip_full: list[np.ndarray] = []
    clip_road: list[np.ndarray] = []
    crop_keys: list[tuple[int, int]] = []
    crop_embs: list[np.ndarray] = []
    last_crop_height: dict[int, float] = {}
    next_clip_t = start / info.fps
    started = time.perf_counter()
    last_report = started

    while (item := frames_q.get()) is not None:
        idx, frame, flow, warp_state["warp"] = item
        t = idx / info.fps
        result = model.track(
            frame, persist=True, tracker=str(tracker_cfg), imgsz=args.imgsz, conf=args.conf,
            classes=classes, device=args.device, verbose=False, **precision,
        )[0]
        assert model.predictor is not None
        tracker = model.predictor.trackers[0]
        if args.gmc == "dense" and not getattr(tracker.gmc, "uses_dense_flow", False):
            tracker.gmc.apply = lambda raw_frame, detections=None: warp_state["warp"]
            tracker.gmc.uses_dense_flow = True
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.int().cpu().numpy()
            ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.full(len(boxes), -1)
        else:
            xyxy, confs, cls, ids = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int)

        mask = np.zeros((FLOW_H, FLOW_W), dtype=bool)
        for b in xyxy:
            mask[max(0, int(b[1] * sy) - 1): int(b[3] * sy) + 2, max(0, int(b[0] * sx) - 1): int(b[2] * sx) + 2] = True
        if flow is not None:
            grids.append(flow_grid(flow, mask, sx, sy))
        else:
            grids.append(np.full((GRID_H, GRID_W, 2), np.nan, dtype=np.float32))
        frame_idx.append(idx)

        for b, c, k, tid in zip(xyxy, confs, cls, ids):
            feats = detection_flow(flow, mask, b, sx, sy) if flow is not None else (float("nan"),) * 7
            det_rows.append((idx, t, int(tid), int(k), float(c), *map(float, b), *feats))

        if clip is not None and t >= next_clip_t:
            next_clip_t += args.clip_every_s
            h, w = frame.shape[:2]
            images = [frame, frame[int(0.55 * h):, int(0.25 * w): int(0.75 * w)]]
            keys = []
            # Throttle crops: embed a track when first seen or once it has grown 1.5x since its last embedding.
            order = np.argsort(-(xyxy[:, 3] - xyxy[:, 1])) if len(xyxy) else []
            for j in order:
                b, tid = xyxy[j], int(ids[j])
                bh, bw = b[3] - b[1], b[2] - b[0]
                if tid < 0 or bh < MIN_CROP_HEIGHT or len(keys) >= MAX_CROPS_PER_SECOND:
                    continue
                if bh < 1.5 * last_crop_height.get(tid, 0.0):
                    continue
                last_crop_height[tid] = float(bh)
                px, py = 0.1 * bw, 0.1 * bh
                crop = frame[max(0, int(b[1] - py)): int(b[3] + py), max(0, int(b[0] - px)): int(b[2] + px)]
                if crop.size:
                    images.append(crop)
                    keys.append((idx, tid))
            embs = clip.encode(images)
            clip_frames.append(idx)
            clip_full.append(embs[0])
            clip_road.append(embs[1])
            crop_keys.extend(keys)
            crop_embs.extend(embs[2:])

        now = time.perf_counter()
        if now - last_report > 60:
            last_report = now
            done = (idx - start) / max(1, end - start)
            rate = len(frame_idx) / (now - started)
            eta = (now - started) * (1 - done) / max(done, 1e-6)
            print(f"{info.stem}: {done:5.1%} t={t:7.1f}s {rate:5.1f} proc-fps dets={len(det_rows)} eta={eta/60:5.1f} min", flush=True)

    reader.join()
    columns = [
        "frame", "time_s", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2",
        "obj_dx", "obj_dy", "obj_scale", "obj_n", "bg_dx", "bg_dy", "bg_n",
    ]
    dets = pd.DataFrame(det_rows, columns=columns)
    for col in dets.columns:
        if dets[col].dtype == np.float64:
            dets[col] = dets[col].astype(np.float32)
    dets.to_parquet(out_dir / "detections.parquet", index=False)
    np.savez_compressed(out_dir / "flow_grid.npz", frame=np.array(frame_idx), grid=np.stack(grids) if grids else np.zeros((0, GRID_H, GRID_W, 2)))
    if clip is not None:
        np.savez_compressed(
            out_dir / "clip.npz",
            frame=np.array(clip_frames), full=np.stack(clip_full) if clip_full else np.zeros((0, 512)),
            road=np.stack(clip_road) if clip_road else np.zeros((0, 512)),
            crop_frame=np.array([k[0] for k in crop_keys]), crop_track=np.array([k[1] for k in crop_keys]),
            crop=np.stack(crop_embs) if crop_embs else np.zeros((0, 512)),
        )
    elapsed = time.perf_counter() - started
    write_json(meta_path, {
        "video": str(path), "stem": info.stem, "start_local": info.start_local,
        "source_fps": info.fps, "source_frames": info.frames, "width": info.width, "height": info.height,
        "stride": stride, "processed_fps": proc_fps, "start_frame": start, "end_frame": end,
        "processed_frames": len(frame_idx), "detections": len(dets), "tracks": int(dets.loc[dets.track_id >= 0, "track_id"].nunique()),
        "model": str(args.model), "imgsz": args.imgsz, "conf": args.conf, "tracker": str(args.tracker),
        "track_buffer_frames": yaml.safe_load(tracker_cfg.read_text())["track_buffer"],
        "clip": None if clip is None else f"{args.clip_model}/{args.clip_pretrained}",
        "flow": {"size": [FLOW_W, FLOW_H], "grid": [GRID_W, GRID_H], "preset": "DIS medium"},
        "gmc": "dense-flow RANSAC similarity" if args.gmc == "dense" else "BoT-SORT sparseOptFlow",
        "precision": "fp16" if precision else "fp32",
        "elapsed_s": round(elapsed, 1), "processed_fps_achieved": round(len(frame_idx) / max(elapsed, 1e-6), 2),
        "complete": args.start_s == 0 and args.end_s is None,
    })
    print(f"{info.stem}: {len(frame_idx)} frames, {len(dets)} detections in {elapsed/60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path, help="Video files or folders")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--tracker", type=Path, default=ROOT / "configs" / "botsort_vehicle_tuned.yaml")
    parser.add_argument("--track-buffer-s", type=float, default=2.0)
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--gmc", choices=["dense", "sparse"], default="dense",
                        help="dense reuses the DIS flow (fast); sparse is BoT-SORT's own GMC (slightly better IDF1 at full frame rate)")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp32", action="store_true",
                        help="Disable half precision on the GPU (FP16 is the default on CUDA: ~1.5-2x faster detector)")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip-every-s", type=float, default=1.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.videos:
        paths.extend(list_videos(item) if item.is_dir() else [item])
    # skip finished videos before loading any model, so re-running the corpus costs nothing for them
    pending = []
    for path in paths:
        meta_path = args.out / path.stem / "meta.json"
        if meta_path.exists() and read_json(meta_path).get("complete") and not args.overwrite:
            print(f"skip {path.stem}: already complete")
        else:
            pending.append(path)
    if not pending:
        return
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    clip = None if args.no_clip else ClipEncoder(args.clip_model, args.clip_pretrained, device, fp16=not args.fp32)
    for path in pending:
        process_video(path, args, clip)


if __name__ == "__main__":
    main()
