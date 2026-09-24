"""Render a typology overlay clip: boxes coloured by relation to the rider, scene and ego-speed banner, legend.

    python -m bikesafe.render VID_20260224_162848_00_006 --start-s 300 --duration-s 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import RELATION_COLOURS_BGR, RELATIONS, ROOT, read_json

LABELS = {
    "ego_lane": "in my lane", "adjacent_same": "other lane, same way", "oncoming": "oncoming",
    "parked": "parked", "cross_side": "side road / other roadway",
}


def draw_legend(frame: np.ndarray) -> None:
    x, y = 16, frame.shape[0] - 16 - 30 * len(RELATIONS)
    cv2.rectangle(frame, (x - 8, y - 26), (x + 300, frame.shape[0] - 8), (20, 20, 20), -1)
    for i, rel in enumerate(RELATIONS):
        cy = y + 30 * i
        cv2.rectangle(frame, (x, cy - 14), (x + 22, cy + 6), RELATION_COLOURS_BGR[rel], -1)
        cv2.putText(frame, LABELS[rel], (x + 32, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results" / "corpus")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--min-box-h", type=float, default=18)
    args = parser.parse_args()

    calib = read_json(args.analysis / args.video / "calibration.json")
    fps = calib["source_fps"]
    kin = pd.read_parquet(args.analysis / args.video / "kinematics.parquet")
    tracks = pd.read_csv(args.corpus / args.video / "tracks_typed.csv")
    line = pd.read_csv(args.corpus / args.video / "timeline_1s.csv").set_index("second")
    kin = kin.merge(tracks[["track_id", "relation", "relation_confidence"]], on="track_id", how="left")
    end_s = args.start_s + args.duration_s
    kin = kin[(kin.time_s >= args.start_s) & (kin.time_s < end_s)]
    frames = kin.groupby("frame")
    processed = sorted(kin.frame.unique())

    out = args.out or ROOT / "demos" / f"typology_{args.video}_{int(args.start_s)}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = args.width / calib["width"]
    size = (args.width, int(round(calib["height"] * scale)))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), calib["fps"], size)
    cap = cv2.VideoCapture(str(args.videos / f"{args.video}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_s * fps))
    idx = int(args.start_s * fps)
    targets = iter(processed)
    target = next(targets, None)
    while target is not None:
        ok = cap.grab()
        if not ok:
            break
        if idx < target:
            idx += 1
            continue
        _, image = cap.retrieve()
        idx += 1
        rows = frames.get_group(target)
        for r in rows.itertuples():
            if r.box_h < args.min_box_h or not isinstance(r.relation, str):
                continue
            colour = RELATION_COLOURS_BGR.get(r.relation, (255, 255, 255))
            thick = 4 if r.relation in ("ego_lane", "adjacent_same", "oncoming") else 2
            cv2.rectangle(image, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, thick)
            if r.box_h >= 40:
                label = f"{LABELS[r.relation]} {r.z_m:.0f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                top = max(0, int(r.y1) - th - 10)
                cv2.rectangle(image, (int(r.x1), top), (int(r.x1) + tw + 8, top + th + 10), colour, -1)
                cv2.putText(image, label, (int(r.x1) + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2, cv2.LINE_AA)
        second = int(target / fps)
        info = line.loc[second] if second in line.index else None
        banner = f"t={target / fps:7.1f}s"
        if info is not None:
            banner += f" | scene: {str(info.scene).replace('_', ' ')} | ego {info.ego_speed_mps:.1f} m/s"
        cv2.rectangle(image, (0, 0), (calib["width"], 54), (20, 20, 20), -1)
        cv2.putText(image, banner, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
        draw_legend(image)
        writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        target = next(targets, None)
    writer.release()
    cap.release()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
