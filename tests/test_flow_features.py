"""Depth-free flow motion cues on synthetic detections with known motion."""

import numpy as np
import pandas as pd
import pytest

from bikesafe.geometry import CAR_HEIGHT_M, Calibration, add_flow_motion, flow_grid_reference
from bikesafe.tracks import flow_motion_summary


def calibration() -> Calibration:
    return Calibration(width=1920, height=1080, fps=12.0, stride=1, horizon_v=600.0, height_ratio=0.75,
                       cam_height_m=1.2, lateral_height_m=1.2, focal_px=672.0, inliers=1000, residual_px=3.0)


def detection(z: float, v_obj_along: float, v_obj_lat: float, u_offset: float = 300.0, ego: float = 5.0) -> dict:
    """One detection of a car at distance z ahead, with the flow a pinhole camera moving forward at `ego` m/s sees."""
    calib = calibration()
    fps = calib.fps
    d = calib.focal_px * calib.cam_height_m / z  # rows below the horizon of its wheels
    box_h = calib.focal_px * CAR_HEIGHT_M / z
    u_c = calib.width / 2 + u_offset
    x = u_offset * z / calib.focal_px  # lateral offset (m)
    closing = ego - v_obj_along
    # image motion per step: static ground at the same place, plus the car's own motion
    ground_du = u_offset * ego / z / fps
    body_du = (u_offset * closing / z + calib.focal_px * v_obj_lat / z) / fps
    return {"frame": 0, "track_id": 1, "x1": u_c - box_h, "x2": u_c + box_h, "y1": calib.horizon_v + d - box_h,
            "y2": calib.horizon_v + d, "box_h": box_h, "d_eff": d, "trunc_top": False, "trunc_bottom": False,
            "geom_ok": True, "obj_n": 400.0, "bg_n": 100.0, "obj_dx": body_du, "bg_dx": ground_du,
            "obj_dy": 0.0, "bg_dy": 0.0, "obj_scale": closing / z / fps, "g_smooth": ego / (calib.focal_px * calib.cam_height_m),
            "ego_moving": True, "u_foe": calib.width / 2, "x_center_m": x, "ego_speed_mps": ego,
            "v_along_flow_mps": np.nan}


@pytest.mark.parametrize("v_along, v_lat, ratio", [
    (0.0, 0.0, 1.0),    # parked: expands like the road, no motion of its own
    (5.0, 0.0, 0.0),    # travelling with the rider
    (-5.0, 0.0, 2.0),   # oncoming at the rider's speed
    (0.0, 4.0, 1.0),    # crossing at 4 m/s
])
def test_flow_motion_recovers_known_motion(v_along, v_lat, ratio):
    u_offset = 300.0
    df = pd.DataFrame([detection(10.0, v_along, v_lat, u_offset=u_offset)])
    add_flow_motion(df, calibration())
    assert df.closing_ratio.iloc[0] == pytest.approx(ratio, abs=0.02)
    # lateral motion plus along-road motion seen at a bearing: v_lat - tan(bearing) * v_along
    expected = v_lat - u_offset / calibration().focal_px * v_along
    assert df.v_lat_flow_mps.iloc[0] == pytest.approx(expected, abs=0.05)


def test_flow_motion_needs_visible_ground_and_a_moving_rider():
    row = detection(10.0, 0.0, 0.0)
    row.update(bg_n=0.0, ego_moving=False)
    df = pd.DataFrame([row])
    add_flow_motion(df, calibration())
    assert np.isnan(df.v_lat_flow_mps.iloc[0]) and np.isnan(df.closing_ratio.iloc[0])


def test_flow_grid_reference_reads_cells_on_the_riders_side():
    calib = calibration()
    grid = np.full((1, 9, 16, 2), np.nan, np.float32)
    grid[0, :, :8] = (-3.0, 1.0)  # left half of the image
    grid[0, :, 8:] = (3.0, 1.0)   # right half
    kin = pd.DataFrame([{"frame": 0, "x1": 1500.0, "y1": 500.0, "x2": 1700.0, "y2": 700.0, "u_foe": 960.0}])
    ref = flow_grid_reference(kin, np.array([0]), grid, calib.width, calib.height)
    assert ref[0] == pytest.approx([3.0, 1.0])


def test_flow_motion_summary_aggregates_per_track():
    rows = [detection(z, 0.0, 0.0) for z in (14.0, 12.0, 10.0, 8.0)]
    df = pd.DataFrame(rows)
    add_flow_motion(df, calibration())
    df["grid_proj"] = 1.0
    summary = flow_motion_summary(df)
    assert summary["closing_ratio_med"] == pytest.approx(1.0, abs=0.02)
    assert summary["v_lat_flow_absmed"] == pytest.approx(0.0, abs=0.05)
    assert summary["closing_n"] == 4 and summary["grid_n"] == 4
