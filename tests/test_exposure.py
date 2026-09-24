"""Riding-context fusion and infrastructure events on small hand-made timelines."""

import numpy as np
import pandas as pd

from bikesafe.exposure import fuse_scene, infrastructure_events


def test_fuse_scene_lets_paint_decide_between_lane_and_road():
    line = pd.DataFrame({
        "scene_clip": ["shared_road", "bike_lane", "separated_path", "shared_road", "bike_lane"],
        "p_separated_path": [0.1, 0.2, 0.8, 0.1, 0.3],
        "in_bike_lane": [True, False, True, None, None],
    })
    assert fuse_scene(line).tolist() == ["bike_lane", "shared_road", "separated_path", "shared_road", "bike_lane"]


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
