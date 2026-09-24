"""Shared constants, video metadata, and path helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Fine-grained relation of a vehicle to the rider, and the professor's four reporting groups.
RELATIONS = ["ego_lane", "adjacent_same", "oncoming", "parked", "cross_side"]
RELATION_GROUP = {
    "ego_lane": "in_my_lane",
    "adjacent_same": "sharing_road",
    "oncoming": "sharing_road",
    "parked": "parked",
    # side streets, crossing or turning traffic, and anything on another roadway (across a median, frontage road)
    "cross_side": "not_sharing",
}
RELATION_COLOURS_BGR = {
    "ego_lane": (40, 40, 235),
    "adjacent_same": (0, 150, 255),
    "oncoming": (200, 60, 200),
    "parked": (170, 170, 170),
    "cross_side": (230, 170, 30),
    "unknown": (255, 255, 255),
}
SCENES = ["separated_path", "bike_lane", "shared_road"]

FILENAME_TIME = re.compile(r"VID_(\d{8})_(\d{6})")


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frames: int
    width: int
    height: int

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def start_local(self) -> str | None:
        """Recording start parsed from the Insta360 file name (camera local time)."""
        match = FILENAME_TIME.search(self.path.name)
        if not match:
            return None
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").isoformat()


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    info = VideoInfo(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def list_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
