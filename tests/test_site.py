"""The map data of scripts/build_site.py (no video, model or network needed)."""

import importlib
import sys

import numpy as np
import pandas as pd

from bikesafe.common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
build_site = importlib.import_module("build_site")
DEMOS, DURATION_S, HEAT = build_site.DEMOS, build_site.DURATION_S, build_site.HEAT
demo_gps, exposure_data, polylines = build_site.demo_gps, build_site.exposure_data, build_site.polylines


def test_routes_break_where_positions_are_hidden():
    t = pd.DataFrame({"lat": [1.0, 2.0, np.nan, np.nan, 3.0, 4.0, np.nan, 5.0],
                      "lon": [9.0, 9.0, np.nan, np.nan, 9.0, 9.0, np.nan, 9.0]})
    assert polylines(t, every=1) == [[[1.0, 9.0], [2.0, 9.0]], [[3.0, 9.0], [4.0, 9.0]]]  # a lone point is no line


def test_a_demo_gets_the_position_of_every_second_of_its_clip():
    name, ride, start, _ = DEMOS[0]
    seconds = np.arange(start - 10, start + DURATION_S + 10)
    t = pd.DataFrame({"second": seconds, "lat": 34.0 + seconds * 1e-5, "lon": -118.0})
    t.loc[t.second == start + 3, ["lat", "lon"]] = np.nan  # hidden (privacy zone)
    clip = demo_gps({ride: t})[name]["clip"]
    assert len(clip) == DURATION_S + 1 and clip[3] is None
    assert clip[0] == [round(34.0 + start * 1e-5, 6), -118.0]


def test_the_exposure_map_has_only_placed_seconds_with_their_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "CORPUS", tmp_path)  # no events.csv there
    t = pd.DataFrame({"second": [0, 1, 2], "timestamp_local": ["2026-02-24T16:28:48"] * 3,
                      "lat": [34.0, np.nan, 34.1], "lon": [-118.0, np.nan, -118.1], "scene": ["bike_lane"] * 3,
                      "n_ego_lane": [1, 5, 0], "n_adjacent_same": [2, 0, 0], "n_oncoming": [0, 0, 3],
                      "door_zone_parked": [0, 0, 1], "n_parked": [4, 4, 4]})
    data = exposure_data({"VID_20260224_162848_00_006": t})
    assert len(data["points"]) == 2  # the second without a position is left out
    traffic = 4 + list(HEAT).index("traffic")
    assert [p[traffic] for p in data["points"]] == [3.0, 3.0]
    assert data["routes"][0]["label"].startswith("Ride 6")
