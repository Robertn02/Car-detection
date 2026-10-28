import json

import numpy as np
import pandas as pd
import pytest

import bikesafe.labels as labels_module
from bikesafe.review import Store, agreement, clean, next_card_id, spread, vehicle_priority


def tracks(**columns) -> pd.DataFrame:
    base = {"video": "VID_A_001", "track_id": [1, 2, 3], "vehicle_relation": ["parked", "parked", "oncoming"],
            "rule_relation": ["parked", "parked", "oncoming"], "proba_parked": [0.9, 0.55, 0.1],
            "proba_oncoming": [0.1, 0.45, 0.9], "z_min": [10.0, 10.0, 10.0], "t_start": [0.0, 10.0, 20.0],
            "frac_in_path": [0.0, 0.0, 0.0], "view_rear": [0.2, 0.2, 0.2], "view_front": [0.6, 0.6, 0.6]}
    base.update(columns)
    return pd.DataFrame(base)


def test_uncertain_and_physically_suspicious_vehicles_come_first():
    t = vehicle_priority(tracks())
    assert t.track_id.tolist()[0] == 2  # 55 % is the least sure
    t = vehicle_priority(tracks(frac_in_path=[0.8, 0.0, 0.0]))
    assert t.track_id.iloc[0] == 1 and "in the rider's path" in t.reasons.iloc[0]
    t = vehicle_priority(tracks(view_rear=[0.2, 0.2, 0.9], view_front=[0.6, 0.6, 0.1]))
    assert t.track_id.iloc[0] == 3 and "rear" in t.reasons.iloc[0]


def test_spread_keeps_one_vehicle_per_moment_of_a_ride():
    items = pd.DataFrame({"video": ["a", "a", "a", "b"], "t_start": [0.0, 1.0, 10.0, 0.5]})
    assert spread(items, 10, gap_s=4).index.tolist() == [0, 2, 3]
    assert len(spread(items, 2, gap_s=4)) == 2


def test_next_card_id_never_reuses_a_number():
    labels = pd.DataFrame({"video": ["VID_X_006"], "card_id": ["T006_007"]})
    anchors = pd.DataFrame({"video": ["VID_X_006"], "card_id": ["T006_012"]})
    assert next_card_id("VID_X_006", labels, anchors) == "T006_013"
    assert next_card_id("VID_X_001", labels, anchors) == "T001_001"


def test_agreement_counts_kept_and_corrected_answers():
    log = pd.DataFrame({"kind": ["vehicle"] * 4, "model": ["parked", "parked", "oncoming", "parked"],
                        "human": ["parked", "adjacent_same", "oncoming", "skip"], "reviewer": "r",
                        "text": None, "model_text": None})
    a = agreement(log)["vehicle"]
    assert a["decisions"] == 3 and a["model_kept"] == pytest.approx(2 / 3, abs=1e-3)
    assert {"model": "parked", "human": "adjacent_same", "n": 1} in a["confusion"]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    video = "VID_20260101_000000_00_001"
    (tmp_path / "perception" / video).mkdir(parents=True)
    (tmp_path / "perception" / video / "meta.json").write_text(json.dumps({"video": "none.mp4"}))
    frames = np.arange(0, 40, 2)
    pd.DataFrame({"frame": frames, "track_id": 5, "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 180.0}).to_parquet(
        tmp_path / "perception" / video / "detections.parquet")
    (tmp_path / "corpus" / video).mkdir(parents=True)
    tracks(video=video, track_id=[5, 6, 7], vehicle_id=[5, 6, 7]).assign(n_det=10, box_h_max=80.0, duration_s=2.0,
                                                                     x_at_min_z=1.0).to_csv(
        tmp_path / "corpus" / video / "tracks_typed.csv", index=False)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    # anchors are written beside the labels (a run elsewhere, e.g. Colab with --labels, must not touch data/typology)
    monkeypatch.setattr(labels_module, "ANCHORS", tmp_path / "not_here.csv")
    s = Store(tmp_path / "analysis", tmp_path / "perception", tmp_path / "corpus", tmp_path / "videos", labels_dir,
              tmp_path / "cache")
    return s, video, labels_dir


def test_a_new_vehicle_label_is_anchored_logged_and_undone(store, tmp_path):
    s, video, labels_dir = store
    assert {i["item"] for i in s.vehicle_queue("new")} == {5, 6, 7}
    s.save({"kind": "vehicle", "video": video, "item": 5, "label": "adjacent_same", "model": "parked",
            "reviewer": "prof"})
    labels = pd.read_csv(labels_dir / "track_labels.csv")
    assert labels.iloc[0].label == "adjacent_same" and labels.iloc[0].labeler == "prof"
    assert len(pd.read_csv(labels_dir / "track_anchors.csv")) == 4
    assert not (tmp_path / "not_here.csv").exists()
    assert 5 not in {i["item"] for i in s.vehicle_queue("new")}
    assert s.undo("prof")["ok"]
    assert pd.read_csv(labels_dir / "track_labels.csv").empty
    assert pd.read_csv(labels_dir / "track_anchors.csv").empty  # the card's anchors go too


def test_a_plate_is_kept_with_the_labels_and_the_log_says_only_whether_it_was_read_right(store, tmp_path):
    s, video, labels_dir = store
    (tmp_path / "analysis" / video).mkdir(parents=True)
    pd.DataFrame({"track_id": [5, 6], "plate_frame": 10, "plate_x1": 1.0, "plate_y1": 2.0, "plate_x2": 30.0,
                  "plate_y2": 12.0}).to_parquet(tmp_path / "analysis" / video / "plates.parquet")
    for item, typed in ((5, "7ABC123"), (6, "8XYZ999")):
        s.save({"kind": "plate", "video": video, "item": item, "label": "plate", "model": "plate", "text": typed,
                "model_text": "7ABC123", "reviewer": "prof"})
    assert list(pd.read_csv(labels_dir / "plate_labels.csv").text) == ["7ABC123", "8XYZ999"]
    log = pd.read_csv(labels_dir / "review_log.csv")
    assert list(log.text) == ["kept", "corrected"] and "7ABC123" not in log.to_csv()
    assert agreement(log)["plate"]["model_kept"] == 0.5


def test_verifying_a_label_replaces_it_and_undo_restores_it(store):
    s, video, labels_dir = store
    pd.DataFrame([{"card_id": "T001_001", "video": video, "track_id": 6, "label": "parked", "labeler": "claude",
                   "notes": ""}]).to_csv(labels_dir / "track_labels.csv", index=False)
    queue = s.vehicle_queue("verify")
    assert [i["item"] for i in queue] == [6] and queue[0]["current"] == "parked"
    s.save({"kind": "vehicle", "video": video, "item": 6, "label": "oncoming", "model": "parked", "reviewer": "prof"})
    row = pd.read_csv(labels_dir / "track_labels.csv").iloc[0]
    assert row.label == "oncoming" and "was parked" in row.notes
    s.undo("prof")
    row = pd.read_csv(labels_dir / "track_labels.csv").iloc[0]
    assert row.label == "parked" and row.labeler == "claude"


def test_api_payloads_carry_no_nan():
    assert clean({"a": [1.0, float("nan")], "b": np.float64("inf"), "c": "NaN"}) == {"a": [1.0, None], "b": None,
                                                                                    "c": "NaN"}


def test_a_track_clip_plays_every_frame_with_the_box_in_between(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import cv2

    from bikesafe.review import track_clip

    video = tmp_path / "ride.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (320, 180))
    for _ in range(72):
        writer.write(np.full((180, 320, 3), 60, np.uint8))
    writer.release()
    frames = np.arange(24, 48, 2)  # detected on every second frame
    dets = pd.DataFrame({"frame": frames, "track_id": 5, "x1": 100.0, "y1": 60.0, "x2": 200.0, "y2": 140.0})
    out = tmp_path / "clip.mp4"
    track_clip(video, dets, 5, out)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == (46 - 24 + 1) + 2 * 24  # the track, a second either side
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24 + 3)  # frame 27: not detected, box interpolated
    ok, image = cap.read()
    cap.release()
    magenta = (image[..., 0] > 180) & (image[..., 1] < 90) & (image[..., 2] > 180)
    assert ok and magenta.sum() > 500


def test_an_exported_page_runs_without_a_server_and_its_decisions_import(store, tmp_path, monkeypatch):
    from bikesafe.review import export, import_decisions

    s, video, labels_dir = store
    monkeypatch.setattr(s, "image", lambda kind, video, item, row=None: b"media")
    export(s, tmp_path / "site", per_queue=2)
    queue = json.loads((tmp_path / "site" / "queue_vehicle_new.json").read_text())
    assert len(queue) == 2 and (tmp_path / "site" / "media" / video / f"clip_{queue[0]['item']}.mp4").exists()
    assert "const STATIC = true;" in (tmp_path / "site" / "index.html").read_text()
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps([{"kind": "vehicle", "video": video, "item": 5, "label": "oncoming",
                                      "model": "parked", "reviewer": "prof", "time": "2026-10-23T10:00:00"}]))
    assert import_decisions(s, decisions) == (1, 0)
    assert pd.read_csv(labels_dir / "track_labels.csv").iloc[0].label == "oncoming"
    assert import_decisions(s, decisions) == (0, 1)  # the same file again adds nothing
    assert len(pd.read_csv(labels_dir / "review_log.csv")) == 1
    assert pd.read_csv(labels_dir / "review_log.csv").time.iloc[0] == "2026-10-23T10:00:00"  # when it was decided
    decisions.write_text(json.dumps([{"kind": "vehicle", "video": video, "item": 6, "label": "car",
                                      "reviewer": "prof", "time": "2026-10-23T10:01:00"}]))
    with pytest.raises(SystemExit):
        import_decisions(s, decisions)


def test_undo_takes_back_only_the_decision_the_page_shows(store):
    s, video, labels_dir = store
    s.save({"kind": "vehicle", "video": video, "item": 5, "label": "oncoming", "model": "parked", "reviewer": "prof"})
    assert not s.undo("prof", {"kind": "vehicle", "video": video, "item": 6})["ok"]  # another item on the page
    assert len(pd.read_csv(labels_dir / "track_labels.csv")) == 1
    assert s.undo("prof", {"kind": "vehicle", "video": video, "item": 5})["ok"]
    assert pd.read_csv(labels_dir / "track_labels.csv").empty


def test_card_ids_stay_unique_across_videos_with_the_same_ending():
    labels = pd.DataFrame({"video": ["ride_a_from0s_3min"], "card_id": ["T3min_001"]})
    anchors = pd.DataFrame({"video": ["ride_a_from0s_3min"], "card_id": ["T3min_001"]})
    assert next_card_id("ride_b_from0s_3min", labels, anchors) == "T3min_002"


def test_decisions_saved_outside_the_labels_retrain_learns_from_are_not_retrained_on(store):
    s, _, _ = store  # labels in a folder of its own, as on Colab
    assert not s.progress()["retrainable"]
    assert s.start_retrain()["state"] == "unavailable"


def test_a_colab_run_and_its_decisions_join_the_rides_and_labels(store, tmp_path):
    from bikesafe.review import import_run

    s, video, labels_dir = store
    pd.DataFrame([{"card_id": "T3min_001", "video": "other_from0s_3min", "track_id": 1, "label": "parked",
                   "labeler": "prof", "notes": ""}]).to_csv(labels_dir / "track_labels.csv", index=False)
    run, ride = tmp_path / "run", "mine_from0s_3min"
    (run / "work" / "perception" / ride).mkdir(parents=True)
    (run / "work" / "perception" / ride / "meta.json").write_text("{}")
    (run / "work" / "analysis" / ride).mkdir(parents=True)
    (run / "results" / ride).mkdir(parents=True)
    (run / "labels").mkdir()
    pd.DataFrame([{"card_id": "T3min_001", "video": ride, "track_id": 9, "label": "oncoming", "labeler": "prof",
                   "notes": ""}]).to_csv(run / "labels" / "track_labels.csv", index=False)
    pd.DataFrame([{"card_id": "T3min_001", "video": ride, "track_id": 9, "frame": f, "x1": 0, "y1": 0, "x2": 9, "y2": 9}
                  for f in (2, 4)]).to_csv(run / "labels" / "track_anchors.csv", index=False)
    pd.DataFrame([{"video": ride, "object_id": 3, "label": "stop"}]).to_csv(run / "labels" / "sign_labels.csv",
                                                                         index=False)
    for _ in range(2):  # importing again changes nothing
        assert ride in import_run(s, run)
    assert (s.perception / ride / "meta.json").exists() and (s.corpus / ride).exists()
    labels = pd.read_csv(labels_dir / "track_labels.csv")
    mine = labels[labels.video == ride]
    assert len(labels) == 2 and mine.card_id.tolist() == ["T3min_002"]  # renumbered: T3min_001 was taken
    anchors = pd.read_csv(labels_dir / "track_anchors.csv")
    assert anchors.card_id.tolist() == ["T3min_002", "T3min_002"]
    assert len(pd.read_csv(labels_dir / "sign_labels.csv")) == 1


def test_the_public_copy_leaves_out_what_was_filmed_in_a_privacy_zone(store, tmp_path, monkeypatch):
    from bikesafe.review import export, private_seconds

    s, video, _ = store
    monkeypatch.setattr(s, "image", lambda kind, video, item, row=None: b"media")
    seconds = np.arange(40)
    pd.DataFrame({"second": seconds, "in_privacy_zone": seconds < 5}).to_csv(  # tracks start at 0, 10 and 20 s
        tmp_path / "corpus" / video / "gps_1s.csv", index=False)
    export(s, tmp_path / "site", per_queue=5)
    queue = json.loads((tmp_path / "site" / "queue_vehicle_new.json").read_text())
    assert {it["item"] for it in queue} == {6, 7}
    (tmp_path / "corpus" / video / "gps_1s.csv").unlink()  # no GPS: the start and end of the ride are left out
    pd.DataFrame({"second": np.arange(1000)}).to_csv(tmp_path / "corpus" / video / "timeline_1s.csv", index=False)
    assert private_seconds(tmp_path / "corpus", video) == set(range(180)) | set(range(820, 1000))


def test_cached_media_are_made_again_when_the_ride_s_tables_change(store, tmp_path):
    s, video, _ = store
    old = tmp_path / "cache" / video / "zoom_5.jpg"  # made before the cache was stamped
    old.parent.mkdir(parents=True)
    old.write_bytes(b"stale")
    s.fresh_cache(video)
    assert not old.exists() and (tmp_path / "cache" / video / "made_from.txt").exists()
    kept = tmp_path / "cache" / video / "zoom_6.jpg"
    kept.write_bytes(b"fresh")
    s._fresh.clear()
    s.fresh_cache(video)  # same tables: the cache stays
    assert kept.exists()
