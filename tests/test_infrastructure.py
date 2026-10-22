"""Lights, lane paint, bike-lane evidence and object linking on synthetic inputs (no video or model needed)."""

import cv2
import numpy as np
import pandas as pd
import pytest

from bikesafe.geometry import Calibration
from bikesafe.infrastructure import (BEV_RES, BEV_X, BEV_Z, GroundView, bev_paint, bike_lane_evidence, lane_lines,
                                     lane_record, light_state, line_extents, link_objects, parse_lines,
                                     per_second_summary)
from bikesafe.render import lane_line_tracks, lines_at, static_box


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


def test_segmented_paint_keeps_only_the_marked_strokes_and_lines_carry_their_extent():
    """With the segmenter's lane-marking mask, a bright strip that is not marked (sunlit asphalt, a kerb) is ignored
    even where it is as thin as paint, and each line records how far ahead its paint was seen."""
    view = GroundView(calibration())
    shape = (len(view.zs), len(view.xs))
    bev = paint_bev([(-1.0, True), (1.5, True)], shape)
    marking = np.zeros(shape, bool)
    col = int((-1.0 - BEV_X[0]) / BEV_RES)
    near = view.zs <= 9.0
    marking[near, col - 3: col + 4] = True  # only the left line, and only up to 9 m ahead, is marked
    valid = np.ones(shape, bool)
    white, yellow = bev_paint(bev, valid, marking)
    lines = lane_lines(white, yellow, valid, view.zs)
    assert len(lines) == 1 and lines[0]["x"] == pytest.approx(-1.0, abs=0.15)
    assert lines[0]["z_far"] <= 9.5 and lines[0]["z_near"] <= 3.5
    rec = lane_record(lines, [], float("nan"), np.zeros(0), np.zeros(0), bike_surface=0.4, crosswalk=0.0)
    assert line_extents(rec["lines_json"])[0][1] == pytest.approx(lines[0]["z_far"], abs=0.05)
    assert parse_lines(rec["lines_json"])[0][0] == pytest.approx(lines[0]["x"], abs=0.01)
    assert bike_lane_evidence(rec) > 0.5  # solid line on the left plus bike-lane surface


def test_bike_lane_surface_counts_and_a_crosswalk_silences_the_sample():
    assert bike_lane_evidence(record(bike_lane_surface=0.5)) > 0.5
    assert np.isnan(bike_lane_evidence(record(lines=[line(-0.9, True)], crosswalk_share=0.3)))


def lanes_table(xs_per_sample: list[list[float]], crosswalk: list[float] | None = None) -> pd.DataFrame:
    rows = []
    for i, xs in enumerate(xs_per_sample):
        rows.append({"time_s": 0.5 * i, "frame": 6 * i, "u_foe": 960.0, "n_lines": len(xs),
                     "crosswalk_share": (crosswalk or [0.0] * len(xs_per_sample))[i],
                     "lines_json": ";".join(f"{x:.2f}:1:0.00:0.000:3.0:12.0" for x in xs)})
    return pd.DataFrame(rows)


def test_lane_lines_are_drawn_only_when_confirmed_and_interpolated_between_samples():
    # a line at -1.0 m drifting to -1.2 m; a spurious line at 2.0 m in one sample only
    lanes = lanes_table([[-1.0], [-1.1, 2.0], [-1.2], [-1.2]])
    tracks = lane_line_tracks(lanes)
    assert tracks.groupby("line_id").size().max() == 4  # the drifting line keeps one id
    drawn, u_foe = lines_at(0.75, lanes, tracks)  # between samples 1 and 2
    assert u_foe == 960.0 and len(drawn) == 1  # the one-off line is not confirmed
    assert -1.2 <= drawn[0]["x"] <= -1.0


def test_no_lane_lines_are_drawn_across_a_crosswalk():
    lanes = lanes_table([[-1.0], [-1.0], [-1.0]], crosswalk=[0.0, 0.4, 0.0])
    tracks = lane_line_tracks(lanes)
    assert lines_at(0.5, lanes, tracks)[0] == []
    assert len(lines_at(0.0, lanes, tracks)[0]) == 1


def test_static_box_follows_a_sign_growing_about_the_focus_of_expansion():
    foe = np.array([960.0, 600.0])
    def at(scale):  # a sign 1.5x as far right as up from the FOE, 30 px tall at scale 1
        centre = foe + scale * np.array([300.0, -200.0])
        half = scale * np.array([10.0, 15.0])
        return np.concatenate([centre - half, centre + half])
    b0, b1 = at(1.0), at(1.6)
    assert np.allclose(static_box(b0, b1, 0.5), at(1.6 ** 0.5))  # in between
    assert np.allclose(static_box(b0, b1, 1.4), at(1.6 ** 1.4))  # after the last sample
    assert np.allclose(static_box(b0, b0 + 2.0, 0.5), b0 + 1.0)  # no growth: linear


def test_short_box_gaps_are_bridged_and_long_ones_are_not():
    from bikesafe.render import fill_box_gaps

    kin = pd.DataFrame({"track_id": [1, 1, 1, 1], "frame": [10, 12, 16, 30], "time_s": [5.0, 6.0, 8.0, 15.0],
                        "x1": [0.0, 10.0, 30.0, 40.0], "y1": 0.0, "x2": 50.0, "y2": 50.0, "box_h": 50.0, "z_m": 5.0,
                        "relation": "ego_lane"})
    out = fill_box_gaps(kin, stride=2).sort_values("frame")
    assert out.frame.tolist() == [10, 12, 14, 16, 30]  # one missing frame filled; the 14-frame gap left alone
    filled = out[out.frame == 14].iloc[0]
    assert filled.x1 == 20.0 and filled.time_s == 7.0 and filled.relation == "ego_lane"


def test_every_source_frame_gets_a_box_between_detections():
    from bikesafe.render import fill_box_gaps

    kin = pd.DataFrame({"track_id": [1, 1, 1], "frame": [10, 12, 30], "time_s": [5.0, 6.0, 15.0],
                        "x1": [0.0, 10.0, 40.0], "y1": 0.0, "x2": 50.0, "y2": 50.0, "box_h": 50.0, "z_m": 5.0})
    out = fill_box_gaps(kin, stride=2, step=1).sort_values("frame")
    assert out.frame.tolist() == [10, 11, 12, 30]  # frame 11 drawn; the 18-frame gap is a lost track, not filled
    assert out[out.frame == 11].x1.iloc[0] == 5.0
