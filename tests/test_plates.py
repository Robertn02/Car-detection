import numpy as np
import pandas as pd

from bikesafe.plates import (KEY_HEX, PLATE_HOLD, PlateBlurrer, blur_region, frames_to_read, identity_audit,
                             normalise, plate_key, plate_owner)


def test_a_plate_belongs_to_the_vehicle_in_whose_box_it_is_central():
    van = np.array([100, 300, 500, 600])  # a plate at x = 470 sits at the far right of the van's box ...
    car = np.array([380, 400, 560, 560])  # ... and in the lower middle of the car's box
    plate = np.array([450, 500, 490, 515])
    assert plate_owner(plate, np.stack([van, car])) == 1
    assert plate_owner(np.array([280, 520, 320, 535]), np.stack([van, car])) == 0


def test_a_plate_at_an_implausible_place_belongs_to_no_vehicle():
    car = np.array([[100, 100, 300, 250]])
    assert plate_owner(np.array([180, 105, 220, 115]), car) is None  # on the roof


def test_blurring_removes_detail_inside_the_box_only():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    before = image.copy()
    blur_region(image, np.array([100, 80, 180, 110]), pad=0.0)
    assert image[80:110, 100:180].std() < 0.3 * before[80:110, 100:180].std()
    assert np.array_equal(image[:70], before[:70]) and np.array_equal(image[:, :90], before[:, :90])


def test_plate_keys_depend_on_the_run_key_and_do_not_contain_the_text():
    key_a, key_b = b"a" * 32, b"b" * 32
    text = normalise(" 7abc-123 ")
    assert text == "7ABC123"
    assert plate_key(text, key_a) == plate_key(text, key_a)
    assert plate_key(text, key_a) != plate_key(text, key_b)
    assert len(plate_key(text, key_a)) == KEY_HEX and text.lower() not in plate_key(text, key_a)


class OneShotFinder:
    """Finds a plate on the first vehicle box once, then never again."""

    def __init__(self):
        self.calls = 0

    def find(self, image, boxes):
        self.calls += 1
        if self.calls > 1 or not len(boxes):
            return []
        x1, y1, x2, y2 = boxes[0]
        return [(0, np.array([x1 + 0.4 * (x2 - x1), y1 + 0.7 * (y2 - y1), x1 + 0.6 * (x2 - x1), y1 + 0.8 * (y2 - y1)]), 0.9)]


def test_a_plate_stays_blurred_for_a_few_frames_after_the_detector_loses_it():
    blurrer = PlateBlurrer(OneShotFinder())
    image = np.zeros((400, 600, 3), np.uint8)
    for _ in range(PLATE_HOLD + 5):
        blurrer(image, np.array([[100, 100, 300, 250]]), [7])
    assert blurrer.plates_found == 1 and blurrer.plates_held == PLATE_HOLD


def test_frames_to_read_picks_large_detections_spread_over_each_track():
    frames = np.arange(20)
    dets = pd.DataFrame({"track_id": 1, "frame": frames, "x1": 0.0, "x2": 100.0, "y1": 0.0,
                         "y2": np.where(frames < 5, 50.0, 120.0)})  # too small for the first five frames
    picked = frames_to_read(dets)
    assert picked.frame.min() == 5 and picked.frame.max() == 19 and len(picked) == 6


def test_identity_audit_counts_agreement_misses_and_conflicts():
    plates = pd.DataFrame({"track_id": [1, 2, 3, 4, 5, 6],
                           "plate_key": ["k1", "k1", "k1", "k2", "k3", None]})
    vehicle_of = {1: 1, 2: 1, 3: 3, 4: 4, 5: 4, 6: 6}  # 3 should have been joined to 1; 4 and 5 are two cars
    spans = pd.DataFrame({"t_start": [0, 5, 6, 20, 20, 30], "t_end": [4, 5.5, 8, 25, 25, 31]}, index=[1, 2, 3, 4, 5, 6])
    audit = identity_audit(plates, vehicle_of, spans)
    assert audit["tracks_with_plate_key"] == 5
    assert audit["same_plate_joined_pairs"] == 1  # 1-2
    assert audit["same_plate_split_pairs"] == 2 and audit["same_plate_split_within_gap"] == 2  # 1-3, 2-3
    assert audit["vehicles_with_two_plates"] == 1
    assert audit["vehicles_seen_again_later"] == 0
