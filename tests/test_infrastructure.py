"""Lights, lane paint, bike-lane evidence and object linking on synthetic inputs (no video or model needed)."""

import cv2
import numpy as np
import pandas as pd
import pytest

from bikesafe.geometry import Calibration
from bikesafe.infrastructure import (BEV_RES, BEV_X, BEV_Z, GroundView, bev_paint, bike_lane_evidence, lane_lines,
                                     lane_record, light_state, link_objects, per_second_summary)


def calibration() -> Calibration:
    return Calibration(width=1920, height=1080, fps=12.0, stride=1, horizon_v=600.0, height_ratio=0.75,
                       cam_height_m=1.2, lateral_height_m=1.2, focal_px=672.0, inliers=1000, residual_px=3.0)


def signal_head(lamp: str | None, sky_behind: bool = False) -> np.ndarray:
    crop = np.full((60, 24, 3), (40, 40, 40), np.uint8)  # dark housing (BGR)
    if sky_behind:
        crop[:, :4] = crop[:, -4:] = (235, 206, 135)  # light blue sky strips down both sides
    centres = {"red": 10, "yellow": 30, "green": 50}
    colours = {"red": (130, 60, 250), "yellow": (0, 200, 250), "green": (170, 230, 0)}  # pink-red and cyan-green LEDs
    if lamp:
        cv2.circle(crop, (12, centres[lamp]), 6, colours[lamp], -1)
    return crop


@pytest.mark.parametrize("lamp", ["red", "yellow", "green"])
def test_light_state_reads_the_lit_lamp(lamp):
    assert light_state(signal_head(lamp))[0] == lamp


def test_light_state_ignores_sky_behind_an_unlit_head():
    assert light_state(signal_head(None, sky_behind=True))[0] == "off"
    assert light_state(signal_head("red", sky_behind=True))[0] == "red"


def paint_bev(lines: list[tuple[float, bool]], shape: tuple[int, int]) -> np.ndarray:
    """Grey asphalt with 0.15 m white lines at the given lateral offsets (solid, or 3 m dashes every 9 m)."""
    bev = np.full((*shape, 3), 90, np.uint8)
    zs = BEV_Z[1] - (np.arange(shape[0]) + 0.5) * BEV_RES
    for x, solid in lines:
        col = int((x - BEV_X[0]) / BEV_RES)
        rows = np.ones(shape[0], bool) if solid else ((zs % 9.0) < 3.0)
        bev[rows, col - 3: col + 4] = 230
    return bev


def test_lane_lines_finds_solid_and_broken_lines():
    view = GroundView(calibration())
    shape = (len(view.zs), len(view.xs))
    bev = paint_bev([(-1.0, True), (2.3, False)], shape)
    valid = np.ones(shape, bool)
    white, yellow = bev_paint(bev, valid)
    lines = lane_lines(white, yellow, valid, view.zs)
    assert len(lines) == 2
    left, right = lines
    assert left["x"] == pytest.approx(-1.0, abs=0.15) and left["solid"]
    assert right["x"] == pytest.approx(2.3, abs=0.15) and not right["solid"]
    assert 0.15 < right["cover"] < 0.6


def test_lane_lines_ignores_wide_bright_regions():
    """Kerb concrete and crosswalk bars are wider than paint lines and must not become lines."""
    view = GroundView(calibration())
    shape = (len(view.zs), len(view.xs))
    bev = np.full((*shape, 3), 90, np.uint8)
    bev[:, 300:340] = 200  # a 0.8 m wide bright strip
    valid = np.ones(shape, bool)
    white, yellow = bev_paint(bev, valid)
    assert lane_lines(white, yellow, valid, view.zs) == []


def test_ground_view_matches_the_flat_road_projection():
    calib = calibration()
    view = GroundView(calib)
    image = np.zeros((calib.height, calib.width), np.uint8)
    x, z, u_foe = 1.5, 8.0, 1000.0
    d = calib.focal_px * calib.cam_height_m / z
    u, v = u_foe + x * d / calib.lateral_height_m, calib.horizon_v + d
    cv2.circle(image, (int(round(u)), int(round(v))), 3, 255, -1)
    bev = view.warp(image, u_foe)
    rows, cols = np.nonzero(bev > 128)
    gx, gz = view.ground(cols.mean(), rows.mean())
    assert gx == pytest.approx(x, abs=0.1)
    assert gz == pytest.approx(z, abs=0.3)


def record(**overrides) -> dict:
    lines = overrides.pop("lines", [])
    base = lane_record(lines, overrides.pop("stencils", []), overrides.pop("green", 0.0),
                       np.array(overrides.pop("vehicle_x", [])), np.array(overrides.pop("vehicle_z", [])))
    base.update(overrides)
    return base


def line(x: float, solid: bool, yellow: float = 0.0, cover: float | None = None) -> dict:
    return {"x": x, "shear": 0.0, "cover": cover if cover is not None else (0.95 if solid else 0.35),
            "visible_m": 11.0, "solid": solid, "yellow": yellow}


def test_bike_lane_evidence_rules():
    kerbside_lane = record(lines=[line(-0.9, True), line(2.6, False, cover=0.4)])  # bright gutter on the right
    assert bike_lane_evidence(kerbside_lane) > 0.5
    traffic_lane = record(lines=[line(-1.7, False), line(1.8, False)])
    assert bike_lane_evidence(traffic_lane) < 0
    beside_centre_line = record(lines=[line(-1.2, True, yellow=0.9)])
    assert bike_lane_evidence(beside_centre_line) < 0
    stencil = record(stencils=[{"x": 0.2, "z": 6.0, "conf": 0.2}])
    assert bike_lane_evidence(stencil) > 0.5
    assert np.isnan(bike_lane_evidence(record()))  # no paint in view says nothing
    crosswalk = record(lines=[line(x, False) for x in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)])
    assert np.isnan(bike_lane_evidence(crosswalk))


def test_per_second_summary_smooths_and_holds_through_junctions():
    times = np.arange(0, 40, 0.5)
    evidence = np.where(times < 15, -0.7, 0.7)
    evidence[(times >= 24) & (times < 27)] = np.nan  # a junction: no paint in view
    lanes = pd.DataFrame({"time_s": times, "frame": (times * 12).astype(int), "bike_lane_evidence": evidence,
                          "n_lines": 2})
    summary = per_second_summary(lanes, pd.DataFrame(), 40)
    in_lane = pd.to_numeric(summary.in_bike_lane, errors="coerce")
    assert (in_lane.loc[0:10] == 0).all()
    assert (in_lane.loc[19:38] == 1).all()  # including the junction at 24-26 s


def test_link_objects_follows_a_static_sign_and_separates_two():
    foe = (960.0, 600.0)
    rows = []
    for k, scale in enumerate([1.0, 1.25, 1.6, 2.1]):
        box = np.array([1100.0, 450.0, 1130.0, 480.0])
        pts = np.array(foe * 2) + (box - np.array(foe * 2)) * scale
        rows.append({"frame": 6 * k, "time_s": k * 0.5, "kind": "sign", "category": "stop", "x1": pts[0], "y1": pts[1],
                     "x2": pts[2], "y2": pts[3]})
        rows.append({"frame": 6 * k, "time_s": k * 0.5, "kind": "sign", "category": "stop", "x1": 300.0, "y1": 400.0,
                     "x2": 320.0, "y2": 420.0})
    objects = pd.DataFrame(rows)
    ids = link_objects(objects, {6 * k: foe[0] for k in range(4)}, foe[1])
    assert ids.nunique() == 2
    assert ids[objects.x1 >= 1000].nunique() == 1
