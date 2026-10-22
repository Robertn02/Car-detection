"""Labels follow their vehicle when the perception pass is re-run and the tracker numbers vehicles differently."""

import pandas as pd

from bikesafe.labels import align_track_labels, anchor_rows


def detections(track_ids: dict[str, int], frames=range(0, 40, 2), shift_px: float = 0.0) -> pd.DataFrame:
    """Two vehicles, "left" and "right", each with its own box path; track_ids gives each one's tracker id."""
    rows = []
    for f in frames:
        rows.append((f, track_ids["left"], 100 + f + shift_px, 400, 300 + f + shift_px, 500))
        rows.append((f, track_ids["right"], 1200 - f, 380, 1420 - f, 520))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])


def write_run(root, video: str, dets: pd.DataFrame) -> None:
    (root / video).mkdir(parents=True, exist_ok=True)
    dets.to_parquet(root / video / "detections.parquet", index=False)


def labels_and_anchors(original: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame({"card_id": ["T001_001", "T001_002"], "video": ["v1", "v1"], "track_id": [7, 9],
                           "label": ["parked", "oncoming"]})
    rows = []
    for card, tid in (("T001_001", 7), ("T001_002", 9)):
        rows += anchor_rows(card, "v1", original[original.track_id == tid], [4, 16, 30, 36])
    return labels, pd.DataFrame(rows)


def test_unchanged_run_keeps_the_labels(tmp_path):
    original = detections({"left": 7, "right": 9})
    write_run(tmp_path, "v1", original)
    labels, anchors = labels_and_anchors(original)
    aligned = align_track_labels(labels, tmp_path, anchors)
    assert aligned.alignment.tolist() == ["same", "same"]
    assert aligned.track_id.tolist() == [7, 9]


def test_renumbered_run_moves_each_label_to_its_vehicle(tmp_path):
    labels, anchors = labels_and_anchors(detections({"left": 7, "right": 9}))
    write_run(tmp_path, "v1", detections({"left": 131, "right": 7}))  # id 7 now belongs to the other vehicle
    aligned = align_track_labels(labels, tmp_path, anchors)
    assert aligned.alignment.tolist() == ["remapped", "remapped"]
    assert aligned.track_id.tolist() == [131, 7]
    assert aligned.labelled_track_id.tolist() == [7, 9]


def test_run_at_another_frame_rate_matches_the_nearest_frame(tmp_path):
    labels, anchors = labels_and_anchors(detections({"left": 7, "right": 9}))
    write_run(tmp_path, "v1", detections({"left": 3, "right": 4}, frames=range(1, 40, 3), shift_px=1.0))
    aligned = align_track_labels(labels, tmp_path, anchors)
    assert aligned.track_id.tolist() == [3, 4]


def test_vehicle_missing_from_the_new_run_is_lost_and_unanchored_labels_pass_through(tmp_path):
    original = detections({"left": 7, "right": 9})
    labels, anchors = labels_and_anchors(original)
    rerun = original[original.track_id == 9]  # the left vehicle is no longer detected
    write_run(tmp_path, "v1", rerun)
    extra = pd.DataFrame({"card_id": ["T002_001"], "video": ["v2"], "track_id": [5], "label": ["parked"]})
    aligned = align_track_labels(pd.concat([labels, extra], ignore_index=True), tmp_path, anchors)
    assert aligned.alignment.tolist() == ["lost", "same", "no_perception"]
    unanchored = align_track_labels(labels, tmp_path, anchors.iloc[:0])
    assert unanchored.alignment.tolist() == ["no_anchor", "no_anchor"]
