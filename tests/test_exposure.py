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


def track(**columns) -> dict:
    """A track that overtakes the rider: seen 2 s, passes 1.2 m to the left, 3 m/s faster than the rider."""
    base = {"track_id": 1, "vehicle_id": 1, "vehicle_relation": "adjacent_same", "t_start": 10.0, "t_end": 12.0,
            "z_first": 10.0, "z_min": 2.0, "log_growth": -0.5, "n_det": 24, "ego_speed_med": 5.0, "v_along_med": 8.0,
            "x_inner_at_min_z": -1.2, "x_at_min_z": -2.0, "x_inner_absmin": 1.2}
    return {**base, **columns}


def test_an_overtake_is_a_vehicle_followed_for_a_second_passing_beside_the_rider():
    from bikesafe.exposure import events

    rows = [track(),  # a close pass
            track(track_id=2, vehicle_id=2, x_inner_at_min_z=2.5),  # an overtake with room
            track(track_id=3, vehicle_id=3, n_det=6, vehicle_relation="parked"),  # a fragment beside parked cars
            track(track_id=4, vehicle_id=4, x_inner_at_min_z=0.0),  # straddles the rider's heading: in the path
            track(track_id=5, vehicle_id=5, x_inner_at_min_z=6.0),  # two lanes over
            track(track_id=6, vehicle_id=6, v_along_med=60.0)]  # implausibly fast: a mismeasured track
    ev = events(pd.DataFrame(rows), pd.DataFrame())
    assert sorted(ev[ev.event == "overtaken_by_vehicle"].vehicle_id) == [1, 2]
    assert ev[ev.event == "close_pass_under_1p5m"].vehicle_id.tolist() == [1]


def test_stop_signs_split_by_the_linker_are_one_event():
    signs = pd.DataFrame({"object_id": [1, 2, 3], "kind": "sign", "category": "stop", "t_first": [10.0, 11.5, 40.0],
                          "t_last": [11.0, 12.0, 41.0], "n_samples": [3, 2, 2], "conf_max": [0.6, 0.9, 0.8],
                          "box_h_max": 60.0, "states": ""})
    line = pd.DataFrame({"second": [0], "traffic_light_state": [None], "ego_moving": [True]})
    ev = infrastructure_events(signs, line)
    assert ev.t_start.tolist() == [10.0, 40.0] and ev.t_end.iloc[0] == 12.0 and ev.score.iloc[0] == 0.9


def test_the_timeline_speed_is_never_negative_and_moving_follows_it():
    from bikesafe.exposure import timeline

    ego = pd.DataFrame({"time_s": [0.1, 0.6, 1.1, 1.6, 2.1], "ego_speed_mps": [-2.0, -1.0, 4.0, 4.2, 0.5],
                        "ego_moving": [True, True, False, False, True]})
    kin = pd.DataFrame({"track_id": [1], "frame": [0], "time_s": [0.2], "box_h": [50.0], "z_m": [5.0],
                        "x_inner_m": [1.0]})
    tracks = pd.DataFrame({"track_id": [1], "vehicle_id": [1], "vehicle_relation": ["parked"]})
    scene = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "p_separated_path": 0.1, "p_bike_lane": 0.2, "p_shared_road": 0.7})
    out = timeline(kin, tracks, ego, scene, None)
    assert out.ego_speed_mps.min() == 0.0 and out.ego_moving.tolist() == [False, True, False]
    assert out.n_parked.dtype.kind == "i"


def test_a_vehicle_is_counted_the_same_in_vehicles_csv_and_the_summary():
    from bikesafe.exposure import counted, vehicle_summary, vehicle_table

    tracks = pd.DataFrame({"track_id": [1], "vehicle_id": [1], "vehicle_relation": ["parked"], "n_det": [12],
                           "class_id": [2], "t_start": [0.0], "t_end": [0.996], "z_min": [14.97],
                           "x_inner_at_min_z": [1.0], "x_at_min_z": [2.0], "box_h_max": [34.6]})
    # rounded, it would look like 1.0 s, 15.0 m and 35 px - inside every limit - but it is outside all three
    assert not vehicle_summary(tracks, None).counted.iloc[0] and counted(vehicle_table(tracks)).empty


def test_a_vehicle_boxed_as_car_and_truck_takes_the_class_it_was_seen_as_longest():
    from bikesafe.exposure import vehicle_summary

    tracks = pd.DataFrame({"track_id": [1, 2, 3], "vehicle_id": 1, "vehicle_relation": "parked", "n_det": [5, 6, 40],
                           "class_id": [2, 2, 7], "t_start": 0.0, "t_end": 3.0, "z_min": 5.0, "x_inner_at_min_z": 1.0,
                           "x_at_min_z": 2.0, "box_h_max": 80.0})
    assert vehicle_summary(tracks, None).vehicle_class.iloc[0] == "truck"


def test_calibration_says_why_when_a_clip_has_too_few_cars():
    from bikesafe.geometry import fit_horizon

    empty = pd.DataFrame(columns=["class_id", "confidence", "x1", "y1", "x2", "y2"], dtype=float)
    with pytest.raises(ValueError, match="cannot calibrate"):
        fit_horizon(empty, 1920, 1080, 12.0, 2)
    same = pd.DataFrame({"class_id": 2, "confidence": 0.9, "x1": 100.0, "x2": 200.0, "y1": np.full(30, 600.0),
                         "y2": np.full(30, 680.0)})  # thirty cars of one size: no slope to fit
    with pytest.raises(ValueError, match="cannot calibrate"):
        fit_horizon(same, 1920, 1080, 12.0, 2)


def test_a_vehicle_never_placed_on_the_road_still_gets_a_row():
    from bikesafe.exposure import vehicle_summary

    tracks = pd.DataFrame({"track_id": [1, 2], "vehicle_id": [1, 2], "vehicle_relation": "parked", "n_det": 10,
                           "class_id": 2, "t_start": 0.0, "t_end": 2.0, "z_min": [np.nan, 5.0],
                           "x_inner_at_min_z": [np.nan, 1.0], "x_at_min_z": [np.nan, 2.0], "box_h_max": 80.0})
    v = vehicle_summary(tracks, None).set_index("vehicle_id")
    assert len(v) == 2 and np.isnan(v.loc[1, "closest_m"]) and not v.loc[1, "counted"] and v.loc[2, "closest_m"] == 5.0


def test_each_vehicle_keeps_its_own_side_whatever_the_order_of_closest_approach():
    from bikesafe.exposure import vehicle_summary

    tracks = pd.DataFrame({"track_id": [1, 2, 3], "vehicle_id": [1, 2, 3], "vehicle_relation": "parked", "n_det": 10,
                           "class_id": 2, "t_start": 0.0, "t_end": 2.0, "z_min": [9.0, 3.0, 6.0],
                           "x_inner_at_min_z": [-1.0, 1.0, -2.0], "x_at_min_z": [-2.0, 2.0, -3.0], "box_h_max": 80.0})
    v = vehicle_summary(tracks, None).set_index("vehicle_id")
    assert v.side.to_dict() == {1: "left", 2: "right", 3: "left"}
