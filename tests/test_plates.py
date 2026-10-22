import numpy as np
import pandas as pd

from bikesafe.plates import READS_PER_TRACK, consensus, frames_to_read, identity_audit, normalise, plate_owner, vehicle_plates


def test_a_plate_belongs_to_the_vehicle_in_whose_box_it_is_central():
    van = np.array([100, 300, 500, 600])  # a plate at x = 470 sits at the far right of the van's box ...
    car = np.array([380, 400, 560, 560])  # ... and in the lower middle of the car's box
    plate = np.array([450, 500, 490, 515])
    assert plate_owner(plate, np.stack([van, car])) == 1
    assert plate_owner(np.array([280, 520, 320, 535]), np.stack([van, car])) == 0


def test_a_plate_at_an_implausible_place_belongs_to_no_vehicle():
    car = np.array([[100, 100, 300, 250]])
    assert plate_owner(np.array([180, 105, 220, 115]), car) is None  # on the roof


def test_consensus_takes_the_text_read_most_often():
    assert consensus([("7ABC123", 0.9), ("7ABC123", 0.8), ("7ABG123", 0.95)]) == ("7ABC123", 2)


def test_consensus_votes_per_character_when_every_read_differs():
    reads = [("7ABC123", 0.8), ("7A8C123", 0.8), ("7ABC128", 0.8)]
    assert consensus(reads) == ("7ABC123", 3)


def test_a_single_read_stands_only_when_very_confident():
    assert consensus([("7ABC123", 0.8)]) == (None, 0)
    assert consensus([("7ABC123", 0.95)]) == ("7ABC123", 1)
    assert consensus([]) == (None, 0)


def test_frames_to_read_picks_the_largest_view_in_each_part_of_a_track():
    frames = np.arange(40)
    dets = pd.DataFrame({"track_id": 1, "frame": frames, "x1": 0.0, "x2": 100.0, "y1": 0.0,
                         "y2": np.where(frames < 10, 50.0, 60.0 + frames)})  # too small for the first ten frames
    picked = frames_to_read(dets)
    assert len(picked) == READS_PER_TRACK and picked.frame.min() >= 10 and picked.frame.max() == 39
    assert picked.frame.is_unique and (np.diff(np.sort(picked.frame.to_numpy())) >= 2).all()


def test_normalise_keeps_letters_and_digits():
    assert normalise(" 7abc-123 ") == "7ABC123"


def test_identity_audit_counts_agreement_misses_and_conflicts():
    plates = pd.DataFrame({"track_id": [1, 2, 3, 4, 5, 6],
                           "plate_text": ["P1", "P1", "P1", "P2", "P3", None]})
    vehicle_of = {1: 1, 2: 1, 3: 3, 4: 4, 5: 4, 6: 6}  # 3 should have been joined to 1; 4 and 5 are two cars
    spans = pd.DataFrame({"t_start": [0, 5, 6, 20, 20, 30], "t_end": [4, 5.5, 8, 25, 25, 31]}, index=[1, 2, 3, 4, 5, 6])
    audit = identity_audit(plates, vehicle_of, spans)
    assert audit["tracks_with_plate"] == 5
    assert audit["same_plate_joined_pairs"] == 1  # 1-2
    assert audit["same_plate_split_pairs"] == 2 and audit["same_plate_split_within_gap"] == 2  # 1-3, 2-3
    assert audit["vehicles_with_two_plates"] == 1
    assert audit["vehicles_seen_again_later"] == 0


def test_each_vehicle_gets_the_plate_most_of_its_reads_agree_on():
    plates = pd.DataFrame({"track_id": [1, 2, 3], "plate_text": ["P1", "P1", "P9"], "agreeing_reads": [2, 3, 2]})
    assert vehicle_plates(plates, {1: 10, 2: 10, 3: 10}).to_dict() == {10: "P1"}
