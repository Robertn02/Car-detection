"""Render a typology overlay clip: boxes coloured by relation to the rider, scene and ego-speed banner, legend.

When bikesafe.infrastructure has run, the clip also shows the painted lane lines it found (green when the rider is in
a bike lane), traffic lights with their state, traffic signs, and the bike-lane / signal status in the banner.
Licence plates are blurred (bikesafe.plates; --no-blur to switch off), and a vehicle boxed twice by the detector
(car + truck) is drawn once, in the colour of its vehicle-level relation.

    python -m bikesafe.render VID_20260224_162848_00_006 --start-s 300 --duration-s 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bikesafe.common import RELATION_COLOURS_BGR, RELATIONS, ROOT, open_video, read_json
from bikesafe.infrastructure import REF_Z, parse_lines
from bikesafe.plates import PlateBlurrer

LABELS = {
    "ego_lane": "in my lane", "adjacent_same": "other lane, same way", "oncoming": "oncoming",
    "parked": "parked", "cross_side": "side road / other roadway",
}
LIGHT_COLOURS_BGR = {"red": (60, 60, 255), "yellow": (0, 215, 255), "green": (90, 220, 60), "off": (200, 200, 200)}
SIGN_COLOUR_BGR = (255, 255, 0)
LANE_COLOURS_BGR = {"bike": (80, 230, 80), "solid": (255, 255, 255), "broken": (180, 180, 180), "yellow": (0, 220, 255)}
SAMPLE_HOLD_S = 0.3  # draw an infrastructure sample's boxes and lines on frames up to this far from it


def draw_legend(frame: np.ndarray) -> None:
    x, y = 16, frame.shape[0] - 16 - 30 * len(RELATIONS)
    cv2.rectangle(frame, (x - 8, y - 26), (x + 300, frame.shape[0] - 8), (20, 20, 20), -1)
    for i, rel in enumerate(RELATIONS):
        cy = y + 30 * i
        cv2.rectangle(frame, (x, cy - 14), (x + 22, cy + 6), RELATION_COLOURS_BGR[rel], -1)
        cv2.putText(frame, LABELS[rel], (x + 32, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)


def project_line(x0: float, shear: float, u_foe: float, calib: dict, z_range: tuple[float, float] = (2.5, 14.0)) -> np.ndarray:
    """Image polyline of a lane line X(Z) = x0 + shear*(Z - REF_Z) on the flat road (inverse of the bird's-eye warp)."""
    zs = np.linspace(z_range[0], z_range[1], 12)
    xs = x0 + shear * (zs - REF_Z)
    rows_below = calib["focal_px"] * calib["cam_height_m"] / zs
    u = u_foe + xs * rows_below / calib["lateral_height_m"]
    v = calib["horizon_v"] + rows_below
    return np.column_stack([u, v]).astype(np.int32)


def draw_lane_lines(image: np.ndarray, sample: pd.Series, calib: dict, in_bike_lane: bool) -> None:
    for x, solid, yellow, shear in parse_lines(sample.lines_json):
        if abs(x) > 3.0:
            continue
        colour = LANE_COLOURS_BGR["yellow"] if yellow >= 0.5 else LANE_COLOURS_BGR["solid" if solid else "broken"]
        if in_bike_lane and -2.4 <= x <= -0.3 and yellow < 0.5:
            colour = LANE_COLOURS_BGR["bike"]
        pts = project_line(x, shear, float(sample.u_foe), calib)
        cv2.polylines(image, [pts], False, colour, 6 if solid else 3, cv2.LINE_AA)


def draw_objects(image: np.ndarray, objects: pd.DataFrame) -> None:
    for o in objects.itertuples():
        if o.kind == "light":
            colour = LIGHT_COLOURS_BGR.get(o.light_state or "off", LIGHT_COLOURS_BGR["off"])
            label = "signal" + (f" {o.light_state}" if o.light_state in ("red", "yellow", "green") else "")
        elif o.kind == "sign":
            colour = SIGN_COLOUR_BGR
            label = o.category.replace("_", " ")
        else:
            continue
        p1, p2 = (int(o.x1), int(o.y1)), (int(o.x2), int(o.y2))
        cv2.rectangle(image, p1, p2, colour, 3)
        if o.y2 - o.y1 >= 14:
            cv2.putText(image, label, (p1[0], max(66, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


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
    parser.add_argument("--no-infrastructure", action="store_true", help="Do not draw lane lines, signals and signs")
    parser.add_argument("--no-vehicles", action="store_true", help="Draw only the road infrastructure and the banner")
    parser.add_argument("--no-blur", action="store_true", help="Do not blur licence plates")
    args = parser.parse_args()

    calib = read_json(args.analysis / args.video / "calibration.json")
    fps = calib["source_fps"]
    kin = pd.read_parquet(args.analysis / args.video / "kinematics.parquet")
    tracks = pd.read_csv(args.corpus / args.video / "tracks_typed.csv")
    line = pd.read_csv(args.corpus / args.video / "timeline_1s.csv").set_index("second")
    if "vehicle_relation" in tracks:  # one colour per physical vehicle
        tracks = tracks.assign(relation=tracks.vehicle_relation)
    extra = [c for c in ("n_det",) if c in tracks]
    kin = kin.merge(tracks[["track_id", "relation", "relation_confidence", *extra]], on="track_id", how="left")
    if "vehicle_id" not in kin:
        kin["vehicle_id"] = kin.track_id
    end_s = args.start_s + args.duration_s
    kin = kin[(kin.time_s >= args.start_s) & (kin.time_s < end_s)]
    frames = kin.groupby("frame")
    processed = sorted(kin.frame.unique())

    lanes = objects = None
    lanes_path = args.analysis / args.video / "lanes.parquet"
    if lanes_path.exists() and not args.no_infrastructure:
        lanes = pd.read_parquet(lanes_path)
        lanes = lanes[(lanes.time_s >= args.start_s - 1) & (lanes.time_s < end_s + 1)].reset_index(drop=True)
        objects = pd.read_parquet(args.analysis / args.video / "infrastructure.parquet")
        objects = objects[(objects.time_s >= args.start_s - 1) & (objects.time_s < end_s + 1)]

    out = args.out or ROOT / "demos" / f"typology_{args.video}_{int(args.start_s)}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = args.width / calib["width"]
    size = (args.width, int(round(calib["height"] * scale)))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), calib["fps"], size)
    cap = open_video(args.videos / f"{args.video}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_s * fps))
    idx = int(args.start_s * fps)
    blurrer = None if args.no_blur else PlateBlurrer()
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
        in_frame = frames.get_group(target)
        if blurrer is not None:  # before anything is drawn, on the full-resolution frame
            blurrer(image, in_frame[["x1", "y1", "x2", "y2"]].to_numpy(float), in_frame.track_id.tolist())
        second = int(target / fps)
        info = line.loc[second] if second in line.index else None
        in_bike_lane = bool(info is not None and "in_bike_lane" in line and str(info.in_bike_lane) == "True")
        if lanes is not None and len(lanes):
            nearest = int(np.abs(lanes.time_s.to_numpy() - target / fps).argmin())
            sample = lanes.iloc[nearest]
            if abs(sample.time_s - target / fps) <= SAMPLE_HOLD_S:
                draw_lane_lines(image, sample, calib, in_bike_lane)
                draw_objects(image, objects[objects.frame == sample.frame])
        rows = in_frame if not args.no_vehicles else in_frame.iloc[:0]
        if "n_det" in rows:  # a vehicle boxed twice is drawn once, with its longest track's box
            rows = rows.sort_values("n_det", ascending=False).drop_duplicates("vehicle_id")
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
        banner = f"t={target / fps:7.1f}s"
        if info is not None:
            banner += f" | scene: {str(info.scene).replace('_', ' ')} | ego {info.ego_speed_mps:.1f} m/s"
            if "in_bike_lane" in line:
                banner += " | bike lane" if in_bike_lane else ""
                state = info.get("traffic_light_state")
                if isinstance(state, str):
                    banner += f" | signal {state}"
        cv2.rectangle(image, (0, 0), (calib["width"], 54), (20, 20, 20), -1)
        cv2.putText(image, banner, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
        if not args.no_vehicles:
            draw_legend(image)
        writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        target = next(targets, None)
    writer.release()
    cap.release()
    blurred = "" if blurrer is None else f"; {blurrer.plates_found + blurrer.plates_held} plate regions blurred"
    print(f"wrote {out}{blurred}")


if __name__ == "__main__":
    main()
