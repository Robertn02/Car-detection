"""Stage bookkeeping of bikesafe.run and the clip writer (no video or model needed)."""

import os

import cv2
import numpy as np
import pytest

from bikesafe.common import VideoOut, write_json
from bikesafe.run import tracks_outdated
from bikesafe.tracks import FEATURES_VERSION


def finished_tracks(analysis_dir):
    analysis_dir.mkdir()
    write_json(analysis_dir / "calibration.json",
               {"features_version": FEATURES_VERSION, "lane_features": True, "context_features": True})
    (analysis_dir / "tracks.parquet").write_bytes(b"")
    return analysis_dir


def test_tracks_outdated_when_an_input_is_newer(tmp_path):
    ride = finished_tracks(tmp_path / "ride")
    lanes = ride / "lanes.parquet"
    lanes.write_bytes(b"")
    built = os.stat(ride / "tracks.parquet").st_mtime
    os.utime(lanes, (built - 60, built - 60))
    assert not tracks_outdated(ride)
    os.utime(lanes, (built + 60, built + 60))  # the lane pass was re-run after the track features
    assert tracks_outdated(ride)


def test_tracks_outdated_on_old_feature_version(tmp_path):
    ride = finished_tracks(tmp_path / "ride")
    write_json(ride / "calibration.json", {"features_version": FEATURES_VERSION - 1})
    assert tracks_outdated(ride)


def test_video_out_writes_h264(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "clip.mp4"
    writer = VideoOut(path, 12.0, (64, 48))
    for i in range(6):
        writer.write(np.full((48, 64, 3), 40 * i, np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    assert int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little") in (b"h264", b"avc1")
    ok, frame = cap.read()
    assert ok and frame.shape == (48, 64, 3)
    cap.release()
