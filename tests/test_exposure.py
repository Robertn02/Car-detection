import pytest
"""Infrastructure events on small hand-made timelines."""

import numpy as np
import pandas as pd

from bikesafe.exposure import infrastructure_events


def test_red_light_wait_is_one_event_per_stop_even_when_the_signal_is_hidden():
    # a bus hides the signal head for four seconds in the middle of one wait
    states = ["red"] * 5 + [None] * 4 + ["red"] * 3 + ["green"] + [None] * 5
    moving = [False] * 13 + [True] * 5
    line = pd.DataFrame({"second": np.arange(len(states)), "traffic_light_state": states, "ego_moving": moving})
    ev = infrastructure_events(pd.DataFrame(), line)
    waits = ev[ev.event == "red_light_wait"]
    assert len(waits) == 1
    assert waits.iloc[0].t_start == 0 and waits.iloc[0].duration_s == 13 and waits.iloc[0].red_seconds == 8


def test_a_stop_without_a_red_signal_is_not_a_red_light_wait():
    line = pd.DataFrame({"second": np.arange(10), "traffic_light_state": [None] * 6 + ["green"] * 4,
                         "ego_moving": [False] * 8 + [True] * 2})
    assert infrastructure_events(pd.DataFrame(), line).empty


def test_stop_signs_count_once_when_close_enough():
    signs = pd.DataFrame({"object_id": [1, 2, 3], "kind": "sign", "category": ["stop", "stop", "speed_limit"],
                          "t_first": [1.0, 5.0, 2.0], "t_last": [3.0, 5.5, 4.0], "n_samples": [5, 2, 4],
                          "conf_max": [0.9, 0.4, 0.8], "box_h_max": [60.0, 10.0, 50.0], "states": ""})
    line = pd.DataFrame({"second": [0], "traffic_light_state": [None], "ego_moving": [True]})
    ev = infrastructure_events(signs, line)
    assert ev.event.tolist() == ["stop_sign"] and ev.object_id.tolist() == [1]


def test_vehicle_summary_gives_one_readable_row_per_vehicle():
    from bikesafe.exposure import vehicle_summary

    tracks = pd.DataFrame({
        "track_id": [1, 2, 3], "vehicle_id": [10, 10, 20], "vehicle_relation": ["parked", "parked", "oncoming"],
        "n_det": [30, 10, 20], "class_id": [2, 2, 7], "t_start": [5.0, 9.0, 1.0], "t_end": [8.0, 12.0, 4.0],
        "z_min": [4.0, 6.0, 30.0], "x_inner_at_min_z": [1.2, 1.5, -3.0], "x_at_min_z": [2.0, 2.2, -4.0],
        "box_h_max": [120.0, 90.0, 30.0], "proba_parked": [0.9, 0.5, 0.2], "proba_oncoming": [0.1, 0.5, 0.8],
        "frac_rider_stopped": [0.0, 1.0, 0.0], "not_vehicle": [0.0, 0.0, 0.9]})
    v = vehicle_summary(tracks, "2026-02-24T16:28:48").set_index("vehicle_id")
    assert v.loc[10, "relation_text"] == "parked" and v.loc[10, "n_tracks"] == 2 and v.loc[10, "track_ids"] == "1 2"
    assert v.loc[10, "confidence"] == pytest.approx((0.9 * 30 + 0.5 * 10) / 40, abs=1e-3)
    assert v.loc[10, "closest_m"] == 4.0 and v.loc[10, "side"] == "right" and v.loc[10, "counted"]
    assert v.loc[10, "seen_while_rider_stopped"] == 0.25 and v.loc[10, "time_local"] == "2026-02-24T16:28:53"
    assert not v.loc[20, "counted"] and "may not be a vehicle" in v.loc[20, "check"]
    assert list(v.reset_index().vehicle_id) == [20, 10]  # in time order
