"""Render review cards for annotating track relations and riding scenes.

Track cards (2 per sheet, 2x2 tiles): [start] [middle] / [end with the wheel-point trajectory] [zoomed crop].
Scene cards (9 per sheet): one frame every --every-s seconds with an ID stamp.
Card IDs and sheets are per video (T006_012, S006_031) so videos can be labelled as their processing finishes.
The manifest CSVs carry the IDs; labels go into data/typology/*.csv (see LABELING_GUIDE.md).

    python -m bikesafe.cards tracks --per-video 60
    python -m bikesafe.cards scenes --every-s 20
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import RELATIONS, ROOT, read_json

TILE_W, TILE_H = 640, 360


def read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    return image if ok else np.zeros((1080, 1920, 3), np.uint8)


def stamp(tile: np.ndarray, text: str) -> None:
    cv2.rectangle(tile, (0, 0), (min(TILE_W, 14 + 11 * len(text)), 30), (0, 0, 0), -1)
    cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def context_tile(image: np.ndarray, others: pd.DataFrame, box: pd.Series, path: np.ndarray | None) -> np.ndarray:
    canvas = image.copy()
    for o in others.itertuples():
        cv2.rectangle(canvas, (int(o.x1), int(o.y1)), (int(o.x2), int(o.y2)), (200, 200, 200), 1)
    if path is not None and len(path) > 1:
        cv2.polylines(canvas, [path.astype(np.int32)], False, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(path[-2].astype(int)), tuple(path[-1].astype(int)), (0, 255, 255), 5, tipLength=2.0)
    cv2.rectangle(canvas, (int(box.x1) - 3, int(box.y1) - 3), (int(box.x2) + 3, int(box.y2) + 3), (255, 0, 255), 6)
    return cv2.resize(canvas, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def zoom_tile(image: np.ndarray, box: pd.Series) -> np.ndarray:
    h, w = image.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    half_w = max(bw * 1.6, bh * 1.6 * TILE_W / TILE_H, 80) / 2
    half_h = half_w * TILE_H / TILE_W
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    x1, x2 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y1, y2 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    crop = image[y1:y2, x1:x2].copy()
    cv2.rectangle(crop, (int(box.x1 - x1), int(box.y1 - y1)), (int(box.x2 - x1), int(box.y2 - y1)), (255, 0, 255), 2)
    scale = min(TILE_W / max(1, crop.shape[1]), TILE_H / max(1, crop.shape[0]))
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    oy, ox = (TILE_H - resized.shape[0]) // 2, (TILE_W - resized.shape[1]) // 2
    tile[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return tile


def sample_strata(group: pd.DataFrame) -> dict[str, pd.Index]:
    """Strata that guarantee coverage of the safety-relevant cases, not just of what the rules already predict."""
    strata = {f"rule_{name}": idx.index for name, idx in group.groupby("rule_relation")}
    strata["near_path"] = group[(group.x_center_med.abs() <= 2.5) & (group.z_min <= 20) & (group.duration_s >= 1.5)].index
    strata["moving_same_way"] = group[group.v_along_med >= 3].index
    strata["moving_opposite"] = group[group.v_along_med <= -3].index
    strata["crossing_fast"] = group[group.v_lat_absmed >= 4].index
    strata["close_and_large"] = group[group.box_h_max >= 250].index
    strata["ahead_and_moving"] = group[(group.x_center_med.abs() < 3) & (group.duration_s >= 3)
                                       & group.z_med.between(5, 40) & (group.frac_static < 0.5)].index
    strata["overtaking_rider"] = group[(group.enter_edge) & (group.u_rel_first < -0.2) & (group.log_growth < 0)].index
    return {k: v for k, v in strata.items() if len(v)}


def model_strata(group: pd.DataFrame, model: dict, per_class: int = 12) -> dict[str, pd.Index]:
    """Strata from the current relation model: the most likely candidates for each relation (not just the argmax,
    so rare classes still get cards) plus the tracks it is least sure about."""
    X = group.reindex(columns=model["features"]).astype(float).to_numpy()
    proba = model["model"].predict_proba(X)
    classes = list(model["model"].classes_)
    strata = {}
    for i, c in enumerate(classes):
        scores = pd.Series(proba[:, i], index=group.index)
        strata[f"likely_{c}"] = scores.nlargest(min(per_class, len(scores))).index
    top2 = np.sort(proba, axis=1)[:, -2:]
    margin = pd.Series(top2[:, 1] - top2[:, 0], index=group.index)
    strata["uncertain"] = margin.nsmallest(max(10, len(group) // 50)).index
    return {k: v for k, v in strata.items() if len(v)}


def sample_tracks(tracks: pd.DataFrame, per_video: int, seed: int, exclude: set[tuple[str, int]] | None = None,
                  model: dict | None = None) -> pd.DataFrame:
    eligible = tracks[(tracks.n_det >= 8) & (tracks.box_h_max >= 35)]
    if exclude:
        keys = list(zip(eligible.video, eligible.track_id))
        eligible = eligible[[k not in exclude for k in keys]]
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in eligible.groupby("video"):
        strata = model_strata(group, model) if model else sample_strata(group)
        quota = max(4, int(np.ceil(per_video / max(1, len(strata)))))
        chosen: list = []
        for index in strata.values():
            remaining = index.difference(chosen).to_numpy()
            if len(remaining):
                chosen.extend(rng.choice(remaining, min(len(remaining), quota), replace=False))
        extra = max(0, per_video - len(chosen))
        remaining = group.index.difference(chosen).to_numpy()
        if extra and len(remaining):
            chosen.extend(rng.choice(remaining, min(extra, len(remaining)), replace=False))
        picks.append(group.loc[chosen[:per_video]].sort_values("t_start"))
    return pd.concat(picks) if picks else eligible.iloc[:0]


def video_tag(video: str) -> str:
    return video.rsplit("_", 1)[-1]


def render_track_cards(analysis: Path, videos: Path, out: Path, per_video: int, seed: int, only: list[str] | None,
                       labels_csv: Path | None = None, model: dict | None = None, per_sheet: int = 2) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(analysis.glob("*/tracks.parquet"))
    if only:
        paths = [p for p in paths if p.parent.name in only]
    done: set[tuple[str, int]] = set()
    offsets: dict[str, int] = {}
    if labels_csv and labels_csv.exists():
        labelled = pd.read_csv(labels_csv)
        done = set(zip(labelled.video, labelled.track_id))
        offsets = {str(v): int(g.card_id.str.split("_").str[1].astype(int).max()) for v, g in labelled.groupby("video")}
    sample = sample_tracks(pd.concat([pd.read_parquet(p) for p in paths]), per_video, seed, exclude=done, model=model)
    for video, chosen in sample.groupby("video", sort=True):
        tag = video_tag(str(video))
        kin = pd.read_parquet(analysis / str(video) / "kinematics.parquet")
        meta = read_json(analysis / str(video) / "calibration.json")
        fps = meta["source_fps"]
        cap = cv2.VideoCapture(str(videos / f"{video}.mp4"))
        rows, tiles, sheet_no = [], [], 0
        offset = offsets.get(str(video), 0)
        for card_no, row in enumerate(chosen.itertuples(), start=offset + 1):
            t = kin[kin.track_id == row.track_id].sort_values("frame")
            first, last = t.iloc[max(0, int(0.1 * len(t)))], t.iloc[min(len(t) - 1, int(0.9 * len(t)))]
            middle = t.iloc[len(t) // 2]
            biggest = t.loc[t.box_h.idxmax()]
            path = np.column_stack([t.u_c.to_numpy(), t.y2.to_numpy()])
            path = path[(t.frame >= first.frame).to_numpy() & (t.frame <= last.frame).to_numpy()]
            card_id = f"T{tag}_{card_no:03d}"
            tile_a = context_tile(read_frame(cap, first.frame), kin[kin.frame == first.frame], first, None)
            stamp(tile_a, f"{card_id} start t={first.time_s:.1f}s")
            tile_b = context_tile(read_frame(cap, middle.frame), kin[kin.frame == middle.frame], middle, None)
            stamp(tile_b, f"{card_id} middle +{(middle.frame - first.frame) / fps:.1f}s")
            tile_c = context_tile(read_frame(cap, last.frame), kin[kin.frame == last.frame], last, path)
            stamp(tile_c, f"{card_id} end +{(last.frame - first.frame) / fps:.1f}s (yellow = path)")
            tile_d = zoom_tile(read_frame(cap, biggest.frame), biggest)
            stamp(tile_d, f"{card_id} zoom")
            tiles.append(np.vstack([np.hstack([tile_a, tile_b]), np.hstack([tile_c, tile_d])]))
            rows.append({"card_id": card_id, "sheet": sheet_no + 1, "video": video, "track_id": row.track_id,
                         "t_start": row.t_start, "t_end": row.t_end, "frame_first": int(first.frame),
                         "frame_middle": int(middle.frame), "frame_last": int(last.frame), "frame_zoom": int(biggest.frame)})
            if len(tiles) == per_sheet or card_no == offset + len(chosen):
                sheet_no += 1
                while len(tiles) < per_sheet:
                    tiles.append(np.zeros_like(tiles[0]))
                cv2.imwrite(str(out / f"tracks_{tag}_{offset + sheet_no:02d}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
                tiles = []
        cap.release()
        manifest = out / f"track_cards_{tag}.csv"
        frame = pd.DataFrame(rows)
        if manifest.exists():
            frame = pd.concat([pd.read_csv(manifest), frame], ignore_index=True)
        frame.to_csv(manifest, index=False)
        print(f"{video}: {len(rows)} track cards on {sheet_no} sheets -> {out}")


def render_scene_cards(perception: Path, videos: Path, out: Path, every_s: float, only: list[str] | None,
                       per_sheet: int = 9) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = 3
    for meta_path in sorted(perception.glob("*/meta.json")):
        meta = read_json(meta_path)
        if not meta.get("complete") or (only and meta["stem"] not in only):
            continue
        tag = video_tag(meta["stem"])
        clip_frames = np.load(meta_path.parent / "clip.npz")["frame"]
        fps = meta["source_fps"]
        cap = cv2.VideoCapture(str(videos / f"{meta['stem']}.mp4"))
        times = np.arange(every_s / 2, meta["source_frames"] / fps, every_s)
        rows, tiles, sheet_no = [], [], 0
        for card_no, t in enumerate(times, start=1):
            frame = int(clip_frames[np.abs(clip_frames - t * fps).argmin()])
            card_id = f"S{tag}_{card_no:03d}"
            tile = cv2.resize(read_frame(cap, frame), (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            stamp(tile, card_id)
            tiles.append(tile)
            rows.append({"card_id": card_id, "sheet": sheet_no + 1, "video": meta["stem"], "frame": frame, "time_s": frame / fps})
            if len(tiles) == per_sheet or card_no == len(times):
                sheet_no += 1
                while len(tiles) < per_sheet:
                    tiles.append(np.zeros_like(tiles[0]))
                grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, per_sheet, cols)])
                cv2.imwrite(str(out / f"scenes_{tag}_{sheet_no:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
                tiles = []
        cap.release()
        pd.DataFrame(rows).to_csv(out / f"scene_cards_{tag}.csv", index=False)
        print(f"{meta['stem']}: {len(rows)} scene cards on {sheet_no} sheets -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["tracks", "scenes"])
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--perception", type=Path, default=ROOT / "work" / "perception")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "labeling")
    parser.add_argument("--per-video", type=int, default=60)
    parser.add_argument("--every-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", nargs="*", help="Video stems to render (default: all available)")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "typology" / "track_labels.csv",
                        help="Skip tracks already labelled here and continue the card numbering")
    parser.add_argument("--balance-with-model", type=Path, default=None,
                        help="models/typology/relation_model.pkl: sample evenly across predicted relations and uncertain tracks")
    args = parser.parse_args()
    if args.kind == "tracks":
        model = pickle.loads(args.balance_with_model.read_bytes()) if args.balance_with_model else None
        render_track_cards(args.analysis, args.videos, args.out, args.per_video, args.seed, args.only, args.labels, model)
    else:
        render_scene_cards(args.perception, args.videos, args.out, args.every_s, args.only)


if __name__ == "__main__":
    main()
