"""Shared constants, video metadata, and path helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
# Hugging Face models (segmenter, CLIP, Florence-2, depth) live in the project's cache unless HF_HOME says otherwise,
# so every stage, the CARC job and a fresh shell find the ~4 GB already downloaded (set before any import of
# huggingface_hub, which reads it once).
os.environ.setdefault("HF_HOME", str(ROOT / "_cache" / "hf"))
# Set by `bikesafe.run --hw-decode` (or by hand) so every stage decodes with the GPU's video engine when it can.
HW_DECODE_ENV = "BIKESAFE_HW_DECODE"
# The ride videos: the project's videos/ folder, or wherever BIKESAFE_VIDEOS points - such as a Google Drive folder the
# processed rides are archived in (G:/My Drive/bikesafe/videos through Google Drive for desktop). Every --videos
# option defaults to it.
VIDEOS = Path(os.environ.get("BIKESAFE_VIDEOS", ROOT / "videos"))
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
# How the relations read in tables and on the overlay
RELATION_TEXT = {
    "ego_lane": "in my lane", "adjacent_same": "other lane, same way", "oncoming": "oncoming",
    "parked": "parked", "cross_side": "side road / other roadway",
}
VEHICLE_CLASS_TEXT = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
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


def open_video(path: Path, hw: bool | None = None) -> cv2.VideoCapture:
    """VideoCapture with FFmpeg hardware decoding (NVDEC / D3D11 / VAAPI) when requested and available.

    Every stage decodes the whole ride once, so on high-bitrate 360 exports decoding, not the models, can dominate.
    Falls back to software decoding silently; frames are identical either way.
    """
    if hw is None:
        hw = os.environ.get(HW_DECODE_ENV) == "1"
    if hw:
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(str(path))


class VideoOut:
    """Writes BGR frames to an .mp4 as H.264 (libx264 from the ffmpeg bundled with imageio-ffmpeg), or as OpenCV's
    MPEG-4 when imageio-ffmpeg is not installed.

    On 12 fps overlay clips H.264 at CRF 20 is as sharp as OpenCV's MPEG-4 (PSNR 35.8 vs 35.7 dB, banner text 38.4 vs
    37.8 dB) at 70% of the size and, unlike it, plays in browsers and slide software. Frames are piped to ffmpeg as
    raw pixels, so each clip is encoded once.
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int], crf: int = 20):
        self.path = path
        try:
            import imageio_ffmpeg
        except ImportError:
            self._ffmpeg = None
            self._opencv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self._opencv.isOpened():
                raise SystemExit(f"Could not create {path}")
            return
        self._ffmpeg = imageio_ffmpeg.write_frames(
            str(path), size, pix_fmt_in="bgr24", fps=fps, codec="libx264", quality=None, macro_block_size=2,
            output_params=["-crf", str(crf), "-preset", "medium", "-movflags", "+faststart"])
        self._ffmpeg.send(None)

    def write(self, frame) -> None:
        if self._ffmpeg is None:
            self._opencv.write(frame)
        else:
            self._ffmpeg.send(frame)

    def release(self) -> None:
        if self._ffmpeg is None:
            self._opencv.release()
        else:
            self._ffmpeg.close()


def is_cuda(device: str) -> bool:
    device = str(device)
    return device.startswith("cuda") or device.isdigit()


def fp16_kwargs(device: str) -> dict:
    """Half-precision inference arguments for ultralytics predict/track on CUDA, in the form the installed version
    accepts (newer releases replaced `half=True` with `quantize=16`). Empty on CPU."""
    if not is_cuda(device):
        return {}
    from ultralytics.cfg import DEFAULT_CFG_DICT  # pyright: ignore[reportPrivateImportUsage]  (not in __all__)

    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


def resolve_video(meta: dict, videos: Path) -> Path:
    """The ride video recorded in a perception meta.json, or the same file name in `videos` if it has moved."""
    video = Path(meta["video"])
    return video if video.exists() else videos / f"{meta['stem']}.mp4"


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
