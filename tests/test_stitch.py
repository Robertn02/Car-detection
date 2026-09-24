import numpy as np
import pandas as pd

from bikesafe.stitch import duplicates, stitch, vehicle_ids

FPS = 12.0
WIDTH = 1920


def boxes(track_id, times, u, v=600.0, h=120.0, cls=2):
    """Detections of a box h px tall and 1.6 h wide centred at (u, v) at the given times."""
    times = np.asarray(times, float)
    u = np.broadcast_to(np.asarray(u, float), times.shape)
    return pd.DataFrame({"frame": np.round(times * FPS).astype(int), "time_s": times, "track_id": track_id,
                         "class_id": cls, "x1": u - 0.8 * h, "y1": v - h / 2, "x2": u + 0.8 * h, "y2": v + h / 2})


def track(track_id, t0, t1, u0, vu, v=600.0, h=120.0, cls=2):
    """A box moving vu px/s from u0 between t0 and t1."""
    times = np.arange(t0, t1 + 1e-9, 1 / FPS)
    return boxes(track_id, times, u0 + vu * (times - t0), v, h, cls)


def gap_links(links):
    return set(zip(links.A[links.kind == "gap"], links.B[links.kind == "gap"]))


def test_a_break_is_joined_where_the_vehicle_reappears_on_its_path():
    dets = pd.concat([track(1, 0.0, 1.0, 600, 150), track(2, 1.5, 2.5, 600 + 150 * 1.5, 150)])
    links = stitch(dets, WIDTH, FPS)
    assert gap_links(links) == {(1, 2)}
    assert vehicle_ids([1, 2], links) == {1: 1, 2: 1}


def test_a_break_is_not_joined_to_a_vehicle_that_is_still_visible():
    parked = track(1, 0.0, 3.0, 1400, 0)
    times = np.arange(0.0, 1.0 + 1e-9, 1 / FPS)
    drifting = boxes(4, times, np.where(times <= 0.6, 1400, 1400 - 500 * (times - 0.6)), cls=7)  # 2nd box on it
    other = track(2, 1.3, 2.3, 1400 - 500 * 0.7, -500)  # continues the drift: must be another vehicle
    links = stitch(pd.concat([parked, drifting, other]), WIDTH, FPS)
    assert (1, 4) in set(zip(links.A[links.kind == "duplicate"], links.B[links.kind == "duplicate"]))
    assert not gap_links(links)


def test_a_vehicle_leaving_at_the_side_edge_is_not_continued_by_the_next_one():
    leaving = track(1, 0.0, 1.0, WIDTH - 300, 200)  # its box reaches the right border as it ends
    entering = track(2, 1.3, 2.3, WIDTH - 60, -50)
    assert stitch(pd.concat([leaving, entering]), WIDTH, FPS).empty


def test_motion_that_does_not_match_is_not_joined():
    going_right = track(1, 0.0, 1.0, 600, 300)
    coming_back = track(2, 1.25, 2.25, 600 + 300 * 1.25, -300)  # the right place, the opposite direction
    assert stitch(pd.concat([going_right, coming_back]), WIDTH, FPS).empty


def test_an_ambiguous_break_is_left_alone():
    ended = track(1, 0.0, 1.0, 900, 0)
    left_twin = track(2, 1.4, 2.4, 860, 0)
    right_twin = track(3, 1.4, 2.4, 940, 0)  # two different vehicles fit the break equally well
    assert not gap_links(stitch(pd.concat([ended, left_twin, right_twin]), WIDTH, FPS))


def test_duplicate_boxes_on_one_vehicle_are_merged_into_the_longer_track():
    main = track(7, 0.0, 3.0, 800, 20)
    second_box = track(9, 1.0, 1.5, 800 + 20 * 1.0, 20, cls=7).assign(x1=lambda d: d.x1 + 3, y2=lambda d: d.y2 - 2)
    dup = duplicates(pd.concat([main, second_box]))
    assert list(zip(dup.A, dup.B)) == [(7, 9)]
    links = stitch(pd.concat([main, second_box]), WIDTH, FPS)
    assert vehicle_ids([7, 9], links) == {7: 7, 9: 7}


def test_boxes_that_only_touch_are_not_duplicates():
    a = track(1, 0.0, 1.0, 800, 0)
    b = track(2, 0.0, 1.0, 800 + 192, 0)  # side by side, edges touching
    assert duplicates(pd.concat([a, b])).empty


def test_vehicle_ids_take_the_smallest_track_id_of_a_chain():
    links = pd.DataFrame({"A": [5, 9], "B": [9, 12]})
    assert vehicle_ids([5, 9, 12, 20], links) == {5: 5, 9: 5, 12: 5, 20: 20}
