import numpy as np
import pandas as pd

from bikesafe.signs import SIGN_TYPES, decide, normalise_text, read_signs, speed_limit, text_category
from bikesafe.streetseg import CURB, ROAD, SIDEWALK, SIGN_FRONT, sign_boxes, vehicle_context


def probs(**weights) -> np.ndarray:
    p = np.full(len(SIGN_TYPES), 0.01)
    for name, w in weights.items():
        p[list(SIGN_TYPES).index(name)] = w
    return p / p.sum()


def test_text_rules_type_the_signs_read_on_the_rides():
    assert text_category("SPELD LIMIT 35") == "speed_limit"
    assert text_category("Hoover St") == "street_name"
    assert text_category("Virgil Av") == "street_name"
    assert text_category("James M Wood 2800 W") == "street_name"
    assert text_category("North University Park") == "guide"  # a place name, not a parking sign
    assert text_category("PARKHIG") == "parking"
    assert text_category("8AME-10") == "parking"
    assert text_category("DIAGONAL CROSSING OK") == "pedestrian"
    assert text_category("CROSS ON WALK SIGRAL ONLY") == "pedestrian"
    assert text_category("NO STOPPING ANYTIME") == "parking"
    assert text_category("STOP") == "stop"
    assert text_category("DIKE LANE") == "bike"  # OCR slips on small signs
    assert text_category("RIKELANC") == "bike"
    assert text_category("Lake St") == "street_name"  # a near-keyword street name is left alone
    assert text_category("Pico BI") == "street_name" and text_category("OLYMPIC BLVO") == "street_name"
    assert text_category("BI") is None  # the start of BIKE LANE: CLIP types it
    assert text_category("PATKINL") == "parking" and text_category("RESIDENT LOADING") == "parking"
    assert text_category("WARNING") == "warning" and text_category("BANKING") is None  # not two edits from PARKING
    assert text_category("EMERGENCY ACCESS KEEP CLEAR") is None  # MERGE only as a word
    assert text_category("THRU TRAFFIC MERCE LEFT") == "warning"
    assert text_category("AHEAD") is None  # the plaque under a crossing sign: CLIP types the sign above
    assert text_category("RIGHT LANE TURNS RIGHT AHEAD") == "lane_use"
    assert text_category("ROAD WORK AHEAD") == "roadwork" and text_category("FWY DETOUR") == "roadwork"
    assert text_category("ROAD CLOSED AHEAD 7AM TO 4PM SCHOOL DAYS") == "warning"  # closed at set hours
    assert text_category("Metro 2 93") == "bus_stop" and text_category("BUS STOP") == "bus_stop"  # not "stop"
    assert text_category("BUS ONLY") == "lane_use"
    assert text_category("10 EAST") == "guide" and text_category("Santa Monica Foy") == "guide"
    assert text_category("END SCHOO ZONE SPE LIMI 3") == "speed_limit"


def test_normalise_text_fixes_keywords_only():
    assert normalise_text("SPELD LIMIT 35") == "SPEED LIMIT 35"
    assert normalise_text("Hoover St") == "HOOVER ST"


def test_speed_limit_reads_posted_values_only():
    assert speed_limit("SPELD LIMIT 35") == 35
    assert speed_limit("SPEED LIMIT 2O") == 20
    assert speed_limit("LIMIT 37") is None


def test_decide_prefers_text_then_clip_and_needs_evidence_for_stop():
    assert decide(probs(street_name=0.2, bike=0.5), "Hoover St", False) == ("street_name", 1.0)
    assert decide(probs(stop=0.9, do_not_enter=0.3), "", False)[0] == "do_not_enter"  # CLIP alone is not enough
    assert decide(probs(warning=0.2), "", True) == ("stop", 1.0)  # the COCO stop-sign check
    assert decide(probs(not_a_sign=0.8, street_name=0.1), "", False)[0] == "not_a_sign"
    assert decide(probs(not_a_sign=0.4, street_name=0.5), "", False)[0] == "street_name"
    assert decide(probs(street_name=0.9), "", False, height=15)[0] == "other_sign"  # never seen large enough
    assert decide(probs(speed_limit=0.4, warning=0.3), "", False)[0] == "other_sign"  # not sure enough for a limit
    assert decide(probs(street_name=0.9), "", False, aspect=0.2)[0] == "other_sign"  # seen edge-on
    assert decide(probs(speed_limit=0.9), "509", False)[0] == "other_sign"  # a fuel-price board
    assert decide(probs(speed_limit=0.9), "", False)[0] == "speed_limit"
    assert decide(probs(**{"yield": 0.95}), "YIELD", False)[0] == "yield"
    assert decide(probs(**{"yield": 0.95, "warning": 0.04}), "", False)[0] != "yield"  # needs its word
    assert decide(probs(warning=0.9), "", False, contrast=8)[0] == "other_sign"  # too dark to judge at night
    assert decide(probs(not_a_sign=0.9), "", False, contrast=8)[0] == "other_sign"  # nor to reject
    assert decide(probs(not_a_sign=0.9), "", False, aspect=0.2)[0] == "other_sign"  # an edge-on sign is still a sign
    assert text_category("WATCH FOR BICYCLISTS") == "bike"
    assert decide(probs(warning=0.9), "Hoover St", False, contrast=8)[0] == "street_name"  # text still decides


class FakeTyper:
    def __init__(self):
        self.embedded = 0

    def embed(self, crops):
        self.embedded += len(crops)
        return np.ones((len(crops), 2), np.float32)

    def probs(self, feats):
        return np.stack([probs(bike=0.6) for _ in feats])


class FakeOCR:
    def read_text(self, crop):
        return "SPEED LIMIT 25" if crop.shape[0] >= 40 else ""


def test_read_signs_types_each_object_once_from_its_largest_views():
    objects = pd.DataFrame({"object_id": [1, 1, 2], "x1": 0, "x2": 40, "y1": [0, 0, 0], "y2": [20, 50, 30],
                            "stop_confirmed": [False, False, False]})
    rng = np.random.default_rng(0)
    crops = {i: rng.integers(0, 255, (int(objects.y2[i]), 10, 3), dtype=np.uint8) for i in objects.index}
    typer, features = FakeTyper(), {}
    found = read_signs(objects, crops, typer=typer, ocr=FakeOCR(), features=features)
    assert found[1]["category"] == "speed_limit" and found[1]["speed_limit"] == 25
    assert found[2]["category"] == "bike" and found[2]["speed_limit"] is None
    assert typer.embedded == 3 and set(features) == {0, 1, 2}
    again = FakeTyper()  # typing again with the saved text and features needs no image embedding
    retyped = read_signs(objects, crops, typer=again, texts={1: "SPEED LIMIT 25"}, features=features)
    assert again.embedded == 0 and retyped == found


def test_saved_features_are_reused_only_for_the_same_crops_and_model(tmp_path):
    import os

    from bikesafe.signs import load_features, save_features

    crops_path, path = tmp_path / "sign_crops.parquet", tmp_path / "sign_clip.npz"
    crops_path.write_bytes(b"")
    save_features(path, {3: np.array([0.5, 0.25]), 7: np.array([1.0, 0.0])})
    loaded = load_features(path, crops_path)
    assert sorted(loaded) == [3, 7] and np.allclose(loaded[3], [0.5, 0.25])
    later = path.stat().st_mtime + 10
    os.utime(crops_path, (later, later))  # the pass ran again: new crops
    assert load_features(path, crops_path) == {}


def test_typing_fingerprint_follows_the_probe(tmp_path):
    from bikesafe.signs import typing_fingerprint

    probe = tmp_path / "sign_probe.pkl"
    before = typing_fingerprint(probe)
    probe.write_bytes(b"a probe")
    assert typing_fingerprint(probe) != before


def test_sign_boxes_scale_blobs_to_the_frame():
    labels = np.zeros((96, 160), np.uint8)
    labels[10:20, 30:36] = SIGN_FRONT
    labels[50:51, 50:52] = SIGN_FRONT  # too small
    prob = np.full(labels.shape, 0.8, np.float16)
    boxes = sign_boxes(labels, prob, (960, 1600, 3))
    assert len(boxes) == 1
    b = boxes[0]
    assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (300.0, 100.0, 360.0, 200.0) and abs(b["conf"] - 0.8) < 1e-3


def test_vehicle_context_tells_a_kerbside_car_from_one_in_a_lane():
    labels = np.full((400, 800), ROAD, np.uint8)
    labels[:, 700:705] = CURB
    labels[:, 705:] = SIDEWALK
    horizon, lateral_h, u_foe = 100.0, 1.2, 400.0
    parked = (600, 250, 690, 330)  # right of the rider's path, kerb 10 px outside its box
    in_lane = (400, 250, 490, 330)
    rows = vehicle_context(labels, np.array([parked, in_lane], float), [1, 2], u_foe, horizon, lateral_h)
    by_id = {r["track_id"]: r for r in rows}
    m_per_px = lateral_h / (330 - horizon)
    assert abs(by_id[1]["kerb_gap_m"] - 10 * m_per_px) < 1e-6
    assert by_id[2]["kerb_gap_m"] > 1.0  # 210 px of roadway between it and the kerb
    assert by_id[1]["under_offroad"] == 0.0 and by_id[1]["outer_vehicle"] == 0.0


class FakeFeatures:
    """Two kinds of sign the zero-shot text embedding cannot tell apart (both closest to "warning")."""

    def __init__(self):
        import torch

        names = list(SIGN_TYPES)
        self.text = torch.zeros((len(names), 3))
        self.text[names.index("warning")] = torch.tensor([0.0, 0.0, 1.0])

    def embed(self, crops):
        feats = np.array([[1.0, 0.0, 0.5] if c.mean() > 100 else [0.0, 1.0, 0.5] for c in crops])
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def test_the_probe_learns_reviewed_sign_types_zero_shot_confuses():
    from bikesafe.signs import blend_probe, train_probe

    rng = np.random.default_rng(0)
    bright = [np.full((30, 30, 3), 200, np.uint8) for _ in range(20)]
    dark = [np.full((30, 30, 3), 20, np.uint8) for _ in range(20)]
    labels = ["bike"] * 20 + ["parking"] * 20
    groups = rng.choice(["ride1", "ride2", "ride3"], 40)
    typer = FakeFeatures()
    probe, report = train_probe(labels, bright + dark, typer, groups)
    assert report["zero_shot_accuracy"] == 0.0 and report["with_probe_accuracy_lovo"] == 1.0
    names = list(SIGN_TYPES)
    zero = np.eye(len(names))[[names.index("warning")]]
    blended = blend_probe(zero, typer.embed(bright[:1]), probe)[0]
    assert probe is not None and probe["weight"] >= 0.75 and names[int(blended.argmax())] == "bike"  # best weight
    few, report = train_probe(labels[:5], bright[:5], typer, groups[:5])
    assert few is None and not report["trained"]
    one_ride, report = train_probe(labels, bright + dark, typer, ["ride1"] * 40)  # nothing to validate it on
    assert one_ride is None and not report["trained"]


def test_read_sign_objects_saves_and_reuses_clip_features(tmp_path, monkeypatch):
    import cv2

    import bikesafe.signs as signs
    from bikesafe.infrastructure import SIGN_CROPS, SIGN_FEATURES, read_sign_objects

    made = []

    class Typer(FakeTyper):
        def __init__(self, *args, **kwargs):
            super().__init__()
            made.append(self)

    monkeypatch.setattr(signs, "SignTyper", Typer)
    monkeypatch.setattr(signs, "load_probe", lambda: None)
    obj = pd.DataFrame({"kind": ["sign", "sign", "light"], "category": "other_sign", "object_id": [1, 1, 2],
                        "x1": 0.0, "x2": 40.0, "y1": 0.0, "y2": [30.0, 50.0, 40.0], "stop_confirmed": False,
                        "sign_text": "", "speed_limit": np.nan, "type_conf": np.nan})
    rng = np.random.default_rng(0)
    crops = {i: cv2.imencode(".jpg", rng.integers(0, 255, (int(obj.y2[i]), 40, 3), dtype=np.uint8))[1].tobytes()
             for i in (0, 1)}
    (tmp_path / SIGN_CROPS).write_bytes(b"")
    path = tmp_path / SIGN_FEATURES
    first = read_sign_objects(obj.copy(), crops, "cpu", False, ocr_model=None, features_path=path)
    assert path.exists() and made[-1].embedded == 2
    assert (first.category[:2] == "bike").all() and first.category[2] == "other_sign"  # the light is left alone
    again = read_sign_objects(obj.copy(), crops, "cpu", False, ocr_model=None, texts={}, features_path=path)
    assert made[-1].embedded == 0 and again.category.tolist() == first.category.tolist()
