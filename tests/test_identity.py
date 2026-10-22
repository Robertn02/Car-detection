import numpy as np
import pandas as pd

from bikesafe.identity import bhattacharyya, descriptor, remap_ids, split_switches, step_scores
from bikesafe.tracks import own_motion_while_stopped


def patch(bgr) -> np.ndarray:
    image = np.zeros((40, 60, 3), np.uint8)
    image[:] = bgr
    image[::4] = (40, 40, 40)  # some texture, like a car body's panel lines
    return image


def test_descriptors_tell_colours_apart():
    red, red_again, blue = descriptor(patch((30, 30, 200))), descriptor(patch((35, 30, 205))), descriptor(patch((200, 40, 30)))
    assert bhattacharyya(red, red_again) < 0.2 < 0.5 < bhattacharyya(red, blue)
    assert np.isnan(bhattacharyya(red, np.full_like(red, np.nan)))


def track(frames, boxes, colours, track_id=7):
    dets = pd.DataFrame({"frame": frames, "track_id": track_id,
                         "x1": [b[0] for b in boxes], "y1": [b[1] for b in boxes],
                         "x2": [b[2] for b in boxes], "y2": [b[3] for b in boxes]})
    return dets, np.stack([descriptor(patch(c)) for c in colours])


def test_a_hop_to_another_vehicle_across_a_gap_is_cut_and_a_smooth_track_is_not():
    red, blue = (30, 30, 200), (200, 40, 30)
    frames = [0, 2, 4, 6, 8, 16, 18, 20]  # the tracker lost the box for 3 frames, then picked up another vehicle
    smooth = [(100 + 5 * i, 300, 160 + 5 * i, 340) for i in range(5)]
    jumped = [(400, 280, 470, 330), (402, 280, 472, 330), (404, 280, 474, 330)]
    dets, desc = track(frames, smooth + jumped, [red] * 5 + [blue] * 3)
    steps = step_scores(dets, desc)
    assert steps.split.tolist() == [False] * 4 + [True, False, False]
    out = split_switches(dets, desc)
    assert out.raw_track_id.eq(7).all()
    assert out.track_id.tolist()[:5] == [7] * 5 and len(set(out.track_id.tolist()[5:])) == 1
    assert out.track_id.iloc[5] != 7
    same, desc_same = track(frames, smooth + [(140 + 5 * i, 300, 200 + 5 * i, 340) for i in range(3)], [red] * 8)
    assert not step_scores(same, desc_same).split.any()


def test_remap_ids_follows_the_pieces_by_frame():
    id_map = pd.DataFrame({"frame": [0, 2, 16], "raw_track_id": [7, 7, 7], "track_id": [7, 7, 9]})
    table = pd.DataFrame({"frame": [0, 16, 30], "track_id": [7, 7, 5], "value": [1, 2, 3]})
    assert remap_ids(table, id_map).track_id.tolist() == [7, 9, 5]


def test_own_motion_while_stopped_tells_a_departing_car_from_a_parked_one():
    n = 24
    base = pd.DataFrame({"ego_speed_mps": [0.0] * n, "obj_scale": [0.0] * n, "obj_n": 50, "z_m": 8.0,
                         "geom_ok": True, "v_lat_flow_mps": [0.05] * n})
    parked = own_motion_while_stopped(base, fps=12.0)
    assert parked["frac_rider_stopped"] == 1.0 and parked["frac_stopped_own_moving"] == 0.0
    departing = base.assign(obj_scale=[0.0] * 12 + [-0.03] * 12)  # shrinks: drives off when the light turns
    moved = own_motion_while_stopped(departing, fps=12.0)
    assert moved["stopped_own_speed_p90"] > 2.0 and moved["frac_stopped_own_moving"] > 0.3
    riding = own_motion_while_stopped(base.assign(ego_speed_mps=5.0), fps=12.0)
    assert riding["stopped_obs"] == 0 and np.isnan(riding["stopped_own_speed_p90"])
