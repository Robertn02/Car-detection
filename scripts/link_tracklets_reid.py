"""Link non-overlapping tracker fragments with the trained triplet-loss embedding."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


def evenly_spaced_rows(group: pd.DataFrame, count: int) -> pd.DataFrame:
    group = group.sort_values("frame").drop_duplicates("frame")
    if len(group) <= count:
        return group
    indexes = np.linspace(0, len(group) - 1, count).round().astype(int)
    return group.iloc[np.unique(indexes)]


def load_track_embeddings(
    source: Path,
    tracks: pd.DataFrame,
    checkpoint_path: Path,
    samples_per_track: int,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    preprocess = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval()
    projection = nn.Linear(checkpoint["feature_size"], checkpoint["embedding_size"], bias=False)
    projection.load_state_dict(checkpoint["projection"])
    projection.eval()

    selected = []
    for track_id, group in tracks.groupby("track_id"):
        for row in evenly_spaced_rows(group, samples_per_track).to_dict("records"):
            selected.append(row)
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        by_frame[int(row["frame"])].append(row)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}")
    crops: list[tuple[int, Image.Image]] = []
    target_frames = set(by_frame)
    for frame_no in range(1, max(target_frames, default=0) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no not in target_frames:
            continue
        height, width = frame.shape[:2]
        for row in by_frame[frame_no]:
            x1 = max(0, min(width - 1, round(row["x1"])))
            y1 = max(0, min(height - 1, round(row["y1"])))
            x2 = max(x1 + 1, min(width, round(row["x2"])))
            y2 = max(y1 + 1, min(height, round(row["y2"])))
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append((int(row["track_id"]), Image.fromarray(crop)))
    cap.release()

    encoded: dict[int, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch_items = crops[start : start + batch_size]
            images = torch.stack([preprocess(image) for _, image in batch_items])
            features = F.normalize(backbone(images), dim=1)
            vectors = F.normalize(projection(features), dim=1).cpu().numpy()
            for (track_id, _), vector in zip(batch_items, vectors):
                encoded[track_id].append(vector)
            print(f"Encoded {min(start + len(batch_items), len(crops))}/{len(crops)} crops", flush=True)

    embeddings = {}
    sample_counts = {}
    for track_id, vectors in encoded.items():
        mean = np.mean(vectors, axis=0)
        embeddings[track_id] = mean / max(np.linalg.norm(mean), 1e-12)
        sample_counts[track_id] = len(vectors)
    return embeddings, sample_counts


def endpoint_state(group: pd.DataFrame, end: str) -> tuple[int, float, float, float, float, float]:
    ordered = group.sort_values("frame").drop_duplicates("frame")
    rows = ordered.tail(8) if end == "last" else ordered.head(8)
    frames = rows["frame"].to_numpy(dtype=float)
    xs = rows["x_center"].to_numpy(dtype=float)
    ys = rows["y_center"].to_numpy(dtype=float)
    if end == "last" and len(rows) >= 3 and frames[-1] > frames[0]:
        vx = float(np.polyfit(frames, xs, 1)[0])
        vy = float(np.polyfit(frames, ys, 1)[0])
    else:
        vx = vy = 0.0
    row = rows.iloc[-1] if end == "last" else rows.iloc[0]
    area = float(row["width"] * row["height"])
    return int(row["frame"]), float(row["x_center"]), float(row["y_center"]), area, vx, vy


class DisjointSet:
    def __init__(self, ids: list[int], frames: dict[int, set[int]]) -> None:
        self.parent = {track_id: track_id for track_id in ids}
        self.frames = {track_id: set(frames[track_id]) for track_id in ids}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def merge(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b or self.frames[a].intersection(self.frames[b]):
            return False
        if b < a:
            a, b = b, a
        self.parent[b] = a
        self.frames[a].update(self.frames.pop(b))
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--thresholds", default="0.2685,0.45,0.60,0.72,0.82,0.90")
    parser.add_argument("--max-gap", type=int, default=90)
    parser.add_argument("--motion-base", type=float, default=0.05)
    parser.add_argument("--motion-per-frame", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)
    parser.add_argument("--samples-per-track", type=int, default=8)
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(args.tracks_csv)
    tracks = tracks[tracks["track_id"] >= 0].copy()
    grouped = {int(track_id): group.copy() for track_id, group in tracks.groupby("track_id")}
    eligible = {
        track_id: group
        for track_id, group in grouped.items()
        if group["frame"].nunique() >= args.min_track_hits
    }
    embeddings, sample_counts = load_track_embeddings(
        args.source, tracks[tracks["track_id"].isin(eligible)], args.checkpoint,
        args.samples_per_track, args.batch_size,
    )

    metadata = {}
    frame_sets = {}
    for track_id, group in eligible.items():
        first = endpoint_state(group, "first")
        last = endpoint_state(group, "last")
        metadata[track_id] = {
            "first": first,
            "last": last,
            "class_id": int(group["class_id"].mode().iloc[0]),
            "hits": int(group["frame"].nunique()),
        }
        frame_sets[track_id] = set(group["frame"].astype(int))

    candidates = []
    ids = sorted(embeddings)
    for left in ids:
        a = metadata[left]
        for right in ids:
            if left == right:
                continue
            b = metadata[right]
            gap = b["first"][0] - a["last"][0]
            if gap <= 0 or gap > args.max_gap or a["class_id"] != b["class_id"]:
                continue
            predicted_x = a["last"][1] + a["last"][4] * gap
            predicted_y = a["last"][2] + a["last"][5] * gap
            motion_residual = float(np.hypot(predicted_x - b["first"][1], predicted_y - b["first"][2]))
            allowed_motion = min(0.35, args.motion_base + args.motion_per_frame * gap)
            area_ratio = max(a["last"][3], b["first"][3]) / max(min(a["last"][3], b["first"][3]), 1e-9)
            if motion_residual > allowed_motion or area_ratio > args.max_area_ratio:
                continue
            similarity = float(np.dot(embeddings[left], embeddings[right]))
            candidates.append({
                "left_track_id": left,
                "right_track_id": right,
                "cosine_similarity": similarity,
                "gap_frames": gap,
                "motion_residual": motion_residual,
                "allowed_motion": allowed_motion,
                "area_ratio": area_ratio,
                "left_hits": a["hits"],
                "right_hits": b["hits"],
                "left_embedding_samples": sample_counts[left],
                "right_embedding_samples": sample_counts[right],
            })
    candidates_frame = pd.DataFrame(candidates).sort_values("cosine_similarity", ascending=False)
    candidates_frame.to_csv(args.output_dir / "reid_link_candidates.csv", index=False)

    results = []
    thresholds = [float(value) for value in args.thresholds.split(",")]
    all_ids = sorted(grouped)
    all_frame_sets = {track_id: set(group["frame"].astype(int)) for track_id, group in grouped.items()}
    for threshold in thresholds:
        sets = DisjointSet(all_ids, all_frame_sets)
        links = []
        for candidate in candidates_frame.itertuples(index=False):
            if candidate.cosine_similarity < threshold:
                break
            if sets.merge(int(candidate.left_track_id), int(candidate.right_track_id)):
                links.append({
                    "left_track_id": int(candidate.left_track_id),
                    "right_track_id": int(candidate.right_track_id),
                    "cosine_similarity": float(candidate.cosine_similarity),
                    "gap_frames": int(candidate.gap_frames),
                    "motion_residual": float(candidate.motion_residual),
                })
        linked = tracks.copy()
        linked.insert(linked.columns.get_loc("track_id") + 1, "original_track_id", linked["track_id"].astype(int))
        linked["track_id"] = linked["track_id"].astype(int).map(sets.find)
        label = f"{threshold:.4f}".replace(".", "p")
        output_csv = args.output_dir / f"tracks_reid_t{label}.csv"
        linked.to_csv(output_csv, index=False)
        pd.DataFrame(links).to_csv(args.output_dir / f"links_t{label}.csv", index=False)
        results.append({
            "threshold": threshold,
            "links_accepted": len(links),
            "input_track_ids": len(all_ids),
            "output_track_ids": int(linked["track_id"].nunique()),
            "output_csv": str(output_csv),
        })

    summary = {
        "source": str(args.source),
        "tracks_csv": str(args.tracks_csv),
        "checkpoint": str(args.checkpoint),
        "eligible_tracks": len(eligible),
        "candidate_links_after_motion_gating": len(candidates_frame),
        "maximum_gap_frames": args.max_gap,
        "threshold_runs": results,
    }
    (args.output_dir / "reid_linking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
