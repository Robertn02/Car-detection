"""Read the traffic signs found by bikesafe.infrastructure: what kind of sign each one is, and what it says.

Two open models look at the crops of every physical sign (its few largest views):

* **CLIP ViT-L/14** (open_clip, DataComp-XL weights) compares each crop with descriptions of the US sign types
  in SIGN_TYPES - shape, colour and symbol - and the probabilities are averaged over the views. This is what types
  symbol-only signs (no left turn, pedestrian crossing, warning diamonds). Zero-shot on the first 46 hand-typed crops
  from the rides it was right on 34; the misses were mostly small, blurred street-name signs.
* **Florence-2** (Microsoft, MIT licence) reads the text of the largest view when the sign is at least
  MIN_OCR_H px tall: "SPEED LIMIT 35", "Hoover St", "NO PARKING 8AM-10AM". Text decides the type when a rule in
  TEXT_RULES matches it, and gives the posted speed limit. Without such text, small or edge-on signs and uncertain CLIP
  calls stay "other_sign", and stop and yield signs need their word.

Stop signs keep the separate check of bikesafe.infrastructure (the COCO-trained detector's stop-sign class): a sign
is a stop sign when that check confirmed it or its text reads STOP.

Signs typed by a reviewer (bikesafe.review -> data/typology/sign_labels.csv) train a linear probe on the same CLIP
image features (train_probe, run by bikesafe.train). Its probabilities are blended with the zero-shot ones, with the
weight that typed the reviewed signs best in leave-one-ride-out validation, so the typing learns the signs of these
streets as reviews accumulate - and stays zero-shot where the probe does not help yet.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import cv2
import numpy as np

from bikesafe.common import is_cuda

CLIP_MODEL = ("ViT-L-14", "datacomp_xl_s13b_b90k")
OCR_MODEL = "florence-community/Florence-2-large"
MIN_OCR_H = 28  # px; smaller signs gave no usable text on the ride frames
MIN_TYPE_H = 24  # px; a sign never seen larger than this is counted but not typed: CLIP guesses on blurred blobs
MIN_TYPE_CONF = 0.5  # share of the sign-type probability CLIP must give its best type when no text decides it
SURE_TYPE_CONF = {"speed_limit": 0.6}  # a speed-limit sign typed from appearance needs a surer CLIP call
TEXT_ONLY = {"stop", "yield"}  # US stop and yield signs carry the word; CLIP alone calls red markers and side views so
MIN_ASPECT = 0.3  # width / height; a narrower sign is seen edge-on and cannot be typed
# grey-level std of the largest view; under it the view is too dark or washed out to judge by appearance. On ride 006
# (day) 2% of typed signs fall under it; on evening ride 002, 32% - mostly blobs CLIP typed or rejected wrongly
MIN_CONTRAST = 20.0
VIEWS_PER_SIGN = 4
# crops per CLIP image batch: 32 on a GPU with room for it, 4 on a small one - a 2 GB GeForce MX250 holding CLIP-L
# with a batch of 32 spilled into system memory and slowed to a crawl
EMBED_BATCH, SMALL_GPU_BATCH, SMALL_GPU_BYTES = 32, 4, 4 * 2**30
OCR_REPORT = 250  # signs read between progress lines
CROP_PAD = 0.15
NOT_A_SIGN_MIN = 0.6  # the segmenter already says "traffic sign"; overrule it only when CLIP is sure
PROBE_PATH = Path(__file__).resolve().parents[1] / "models" / "typology" / "sign_probe.pkl"
PROBE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # probe weights tried in leave-one-ride-out validation
MIN_PROBE_LABELS = 20

TEMPLATES = ["a photo of {}.", "a close-up photo of {} on a pole.", "{}"]
SIGN_TYPES: dict[str, list[str]] = {
    "stop": ["a red octagonal STOP sign"],
    "yield": ["a red and white triangular YIELD sign"],
    "speed_limit": ["a white SPEED LIMIT sign with a number",
                    "a SCHOOL SPEED LIMIT 25 sign below a fluorescent yellow-green school sign"],
    "do_not_enter": ["a red circular DO NOT ENTER sign with a white bar", "a WRONG WAY sign"],
    "one_way": ["a black and white ONE WAY arrow sign"],
    "no_turn": ["a NO LEFT TURN sign, an arrow crossed by a red circle and slash", "a NO U-TURN sign",
                "a NO RIGHT TURN sign"],
    "parking": ["a NO PARKING sign, a letter P crossed by a red circle",
                "a white parking restriction sign with times of day", "a tow-away no stopping sign",
                "a blue disabled parking sign with a wheelchair symbol",
                "a white sign with a red circle and slash over a letter P and parking hours below",
                "a street sweeping no parking sign with days and times"],
    "bike": ["a BIKE LANE sign with a bicycle symbol", "a bike route sign", "a bicycles may use full lane sign",
             "a share the road sign with a bicycle"],
    "pedestrian": ["a yellow pedestrian crossing warning sign with a walking person", "a school crossing sign",
                   "a pedestrian signal instruction sign", "a fluorescent yellow-green school zone sign with children"],
    "warning": ["a yellow diamond-shaped road warning sign", "a yellow curve ahead sign", "a signal ahead warning sign",
                "a red and white object marker on a pole", "a yellow and black striped object marker"],
    "street_name": ["a street name sign", "a blue street name sign with white letters", "a green street name sign"],
    "guide": ["a green highway guide sign with directions and arrows", "a freeway entrance sign",
              "a destination guide sign to a university, a stadium or a district", "an interstate route shield sign"],
    "lane_use": ["a lane use sign with turn arrows", "a RIGHT LANE MUST TURN RIGHT sign", "a bus lane sign",
                 "a turn only sign"],
    "bus_stop": ["a bus stop sign on a pole", "a Metro bus stop sign with route numbers"],
    # Vistas has no cone or barricade class: the segmenter takes their red and white stripes for a sign face
    "roadwork": ["an orange traffic cone", "a red and white striped traffic cone",
                 "an orange and white striped construction barricade", "an orange ROAD WORK AHEAD sign",
                 "a SIDEWALK CLOSED sign", "a construction arrow board trailer"],
    "not_a_sign": ["a store sign", "an advertisement", "a billboard", "a shop window", "a building wall", "a tree"],
}
SIGN_CATEGORIES = [c for c in SIGN_TYPES if c != "not_a_sign"]

# (pattern on the upper-cased text, category); the first match wins, so specific phrases come before generic words
# "Pico BI" and "BLVO" are how Florence-2 reads "Pico Bl" and "Blvd" (a lone "BI" is the start of BIKE LANE)
STREET_SUFFIX = (r"\b(ST|AVE?|BLVD|BLVO|BL|DR|RD|PL|WAY|PKWY|PARKWAY|CT|LN|TER|CIR|SQ)\b\.?|[A-Z]{3,}\s+BI\b"
                 r"|\b\d{3,5}\s?[NSEW]\b")
DIRECTION = r"\b\d{1,3}\s?(NORTH|SOUTH|EAST|WEST)\b|^(NORTH|SOUTH|EAST|WEST)$"  # "10 EAST": a route marker
TEXT_RULES: list[tuple[str, str]] = [
    (r"BUS STOP|\bMETRO\b", "bus_stop"),
    (r"\bSTOP\b", "stop"),
    (r"\bYIELD\b", "yield"),
    (r"SPEED|LIMIT|\bSPE[A-Z]*\s+LIM", "speed_limit"),  # "END SCHOO ZONE SPE LIMI"
    (r"DO NOT ENTER|WRONG WAY", "do_not_enter"),
    (r"ONE WAY", "one_way"),
    (r"NO (LEFT|RIGHT|U)[ -]?TURN|NO TURN", "no_turn"),
    (r"ROAD WOR|SIDEWALK CLOSED|\bDETOUR\b|CONSTRUCTION|PUBLIC WORKS|WORK ZONE|(ROAD|LANE) CLOSED(?!.*\d\s?[AP]M)",
     "roadwork"),
    (r"CLOSED AHEAD", "warning"),  # a closure at set hours ("7AM TO 4PM SCHOOL DAYS"), not road work
    (r"BIKE|BICYCL|SHARE THE ROAD|FULL LANE", "bike"),  # BICYCL: "WATCH FOR BICYCLISTS"
    (r"PARKWAY", "street_name"),
    # "PARKHIG" is how a parking sign 108 px tall came out; times of day mark parking restrictions ("8AM-10AM")
    (r"PARKING|NO PARK|\bPARK[A-Z]{2,}|\bTOW\b|STOPPING|LOADING|\d\s?[AP]M|\bHOURS?\b|\bHR\b", "parking"),
    (r"PED|CROS|XING|SCHOO|WALK", "pedestrian"),
    # not AHEAD: the plaque under a school or pedestrian crossing sign, which CLIP types from the sign above it
    (r"\bMERGE|LANE ENDS|CLEARANCE|\bWARNING\b|\bCURVE\b|\bSLOW\b|DEAD END|NO OUTLET|\bBUMP\b|\bDIP\b", "warning"),
    (r"\bONLY\b|\bMUST\b|KEEP (RIGHT|LEFT)|LANE", "lane_use"),
    (r"FWY|FREEWAY|EXIT|\bTO\b|\bI-?\d|\bF[OA]Y\b|UNIVERSI|COLLEGE|MUSEUM|STADIUM|ARENA|CENTER|HOSPITAL|DISTRICT",
     "guide"),  # "FOY": Florence-2 on "Fwy"
    (DIRECTION, "guide"),
    (STREET_SUFFIX, "street_name"),
]
SPEED = re.compile(r"\b(1[05]|[2-6][05])\b")  # posted US limits on city streets


# OCR on small signs misspells a letter or two ("SPELD LIMIT", "DIKE LANE", "RIKELANC", "PATKINL"): words within one
# edit of a keyword (two, keeping the first letter, for those in FUZZY_WORDS_2), and runs of letters within two edits of
# a key phrase, are read as the keyword.
FUZZY_WORDS = ["SPEED", "LIMIT", "PARKING", "CROSSING", "SIGNAL", "BICYCLE", "SCHOOL", "MERGE"]
FUZZY_WORDS_2 = {"PARKING", "CROSSING"}  # not BICYCLE (RECYCLE); the first letter keeps WARNING, BANKING out
FUZZY_PHRASES = {"BIKELANE": "BIKE LANE", "BIKEROUTE": "BIKE ROUTE", "ONEWAY": "ONE WAY", "DONOTENTER": "DO NOT ENTER"}


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def normalise_text(text: str) -> str:
    """Upper-case words with OCR slips on the keywords corrected (FUZZY_WORDS, FUZZY_PHRASES)."""
    words = re.findall(r"[A-Z0-9.\-]+", text.upper())
    fixed = [next((k for k in FUZZY_WORDS if edit_distance(w, k) <= (2 if k in FUZZY_WORDS_2 and w[0] == k[0] else 1)),
                  w) if len(w) >= 5 else w for w in words]
    out = " ".join(fixed)
    letters = "".join(ch for ch in text.upper() if ch.isalpha())
    for key, phrase in FUZZY_PHRASES.items():
        if phrase not in out and any(edit_distance(letters[i:i + len(key)], key) <= 2
                                     for i in range(max(1, len(letters) - len(key) + 1))):
            out += " " + phrase
    return out


def text_category(text: str) -> str | None:
    upper = normalise_text(text)
    for pattern, category in TEXT_RULES:
        if re.search(pattern, upper):
            return category
    return None


def speed_limit(text: str) -> int | None:
    match = SPEED.search(text.upper().replace("O", "0"))
    return int(match.group(1)) if match else None


def contrast(crop: np.ndarray) -> float:
    """Grey-level standard deviation of a BGR crop (MIN_CONTRAST)."""
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) if crop.size else 0.0


def sign_crop(image: np.ndarray, box: tuple[float, float, float, float], pad: float = CROP_PAD) -> np.ndarray:
    x1, y1, x2, y2 = box
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    return image[int(max(0, y1 - ph)): int(min(image.shape[0], y2 + ph)),
                 int(max(0, x1 - pw)): int(min(image.shape[1], x2 + pw))].copy()


def decide(probs: np.ndarray, text: str, stop_confirmed: bool, height: float = float("inf"),
           aspect: float = 1.0, contrast: float = float("inf")) -> tuple[str, float]:
    """Category and its confidence from the averaged CLIP probabilities (SIGN_TYPES order), the read text, the
    stop-sign check, and the height (px), width / height and contrast (grey-level standard deviation) of the sign's
    largest view. Text that matches a rule wins. CLIP judges only views it can see: a sign never seen at least
    MIN_TYPE_H tall, seen only edge-on (MIN_ASPECT), or too dark or washed out (MIN_CONTRAST: signs at night) is
    neither typed nor rejected from its appearance - the segmenter's "traffic sign" stands, as "other_sign" (counted,
    not typed). Otherwise "not a sign" needs a confident CLIP call, and a type CLIP is unsure of also leaves
    "other_sign". Stop and yield are never decided by CLIP alone (TEXT_ONLY)."""
    names = list(SIGN_TYPES)
    by_text = text_category(text) if text else None
    if stop_confirmed or by_text == "stop":
        return "stop", 1.0
    if by_text is not None:
        return by_text, 1.0
    if height < MIN_TYPE_H or aspect < MIN_ASPECT or contrast < MIN_CONTRAST:
        return "other_sign", 0.0
    if probs[names.index("not_a_sign")] >= NOT_A_SIGN_MIN:
        return "not_a_sign", float(probs[names.index("not_a_sign")])
    signs = np.array([probs[names.index(c)] for c in SIGN_CATEGORIES])
    for category in TEXT_ONLY:
        signs[SIGN_CATEGORIES.index(category)] = 0
    best = int(signs.argmax())
    category, conf = SIGN_CATEGORIES[best], float(signs[best] / max(signs.sum(), 1e-6))
    if conf < SURE_TYPE_CONF.get(category, MIN_TYPE_CONF):
        return "other_sign", conf
    if category == "speed_limit" and re.search(r"\d", text) and speed_limit(text) is None:
        return "other_sign", conf  # a number that is no posted limit: a fuel-price board, an address
    return category, conf


class SignTyper:
    """CLIP sign-type probabilities for crops."""

    def __init__(self, device: str = "cpu", fp16: bool = True, clip_model: tuple[str, str] = CLIP_MODEL,
                 probe: dict | None = None) -> None:
        import open_clip
        import torch

        self.probe = probe
        self.torch, self.device = torch, device
        precision = "fp16" if fp16 and is_cuda(device) else "fp32"
        self.dtype = torch.float16 if precision == "fp16" else torch.float32  # the first parameter stays fp32
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(clip_model[0], pretrained=clip_model[1],
                                                                              device=device, precision=precision)
        self.clip.eval()
        tokenizer = open_clip.get_tokenizer(clip_model[0])
        with torch.inference_mode():
            rows = []
            for prompts in SIGN_TYPES.values():
                tokens = tokenizer([t.format(p) for p in prompts for t in TEMPLATES]).to(device)
                emb = self.clip.encode_text(tokens).float()
                emb = (emb / emb.norm(dim=-1, keepdim=True)).mean(0)
                rows.append(emb / emb.norm())
            self.text = torch.stack(rows)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """(n, d) unit-length CLIP image features of BGR crops."""
        from PIL import Image

        torch = self.torch
        if not crops:
            return np.zeros((0, self.text.shape[1]), np.float32)
        batch = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops])
        with torch.inference_mode():
            img = self.clip.encode_image(batch.to(self.device, self.dtype)).float()
            return (img / img.norm(dim=-1, keepdim=True)).cpu().numpy()

    def probs(self, feats: np.ndarray) -> np.ndarray:
        """(n, len(SIGN_TYPES)) probabilities for CLIP image features (embed): zero-shot, blended with the
        reviewed-sign probe if any."""
        feats = np.asarray(feats, np.float32)
        if not len(feats):
            return np.zeros((0, len(SIGN_TYPES)))
        logits = 100 * feats @ self.text.cpu().numpy().T
        zero_shot = np.exp(logits - logits.max(axis=1, keepdims=True))
        zero_shot /= zero_shot.sum(axis=1, keepdims=True)
        return blend_probe(zero_shot, feats, self.probe)

    def type_probs(self, crops: list[np.ndarray]) -> np.ndarray:
        """probs() of BGR crops."""
        return self.probs(self.embed(crops))


class SignOCR:
    """Florence-2 text of a crop."""

    def __init__(self, device: str = "cpu", fp16: bool = True, model: str = OCR_MODEL) -> None:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.torch, self.device = torch, device
        self.dtype = torch.float16 if fp16 and is_cuda(device) else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Florence2ForConditionalGeneration.from_pretrained(model, dtype=self.dtype).to(device).eval()

    def read_text(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return ""
        task = "<OCR>"
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        inputs = self.processor(text=task, images=rgb, return_tensors="pt").to(self.device, self.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=32, num_beams=3, do_sample=False)
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        text = self.processor.post_process_generation(raw, task=task, image_size=(rgb.shape[1], rgb.shape[0]))[task]
        return " ".join(str(text).split())


def embed_batch(device: str) -> int:
    """Crops per CLIP image batch for this device (EMBED_BATCH, or SMALL_GPU_BATCH on a GPU under SMALL_GPU_BYTES)."""
    if not is_cuda(device):
        return EMBED_BATCH
    import torch

    index = int(device.split(":")[1]) if ":" in str(device) else 0 if not str(device).isdigit() else int(device)
    small = torch.cuda.get_device_properties(index).total_memory < SMALL_GPU_BYTES
    return SMALL_GPU_BATCH if small else EMBED_BATCH


def free_gpu(device: str) -> None:
    import gc

    gc.collect()
    if is_cuda(device):
        import torch

        torch.cuda.empty_cache()


def blend_probe(zero_shot: np.ndarray, feats: np.ndarray, probe: dict | None, weight: float | None = None) -> np.ndarray:
    """Zero-shot probabilities mixed with the probe's (on the types it was trained on) with the probe's chosen weight
    (or `weight`)."""
    if probe is None:
        return zero_shot
    names = list(SIGN_TYPES)
    learned = np.zeros_like(zero_shot)
    p = probe["model"].predict_proba(feats)
    for j, category in enumerate(probe["model"].classes_):
        learned[:, names.index(category)] = p[:, j]
    w = probe.get("weight", 0.5) if weight is None else weight
    return (1 - w) * zero_shot + w * learned


def load_probe(path: Path = PROBE_PATH) -> dict | None:
    return pickle.loads(path.read_bytes()) if path.exists() else None


def typing_fingerprint(probe_path: Path = PROBE_PATH) -> str:
    """Changes whenever typing the same signs can come out differently: CLIP model, prompts, text rules, thresholds or
    the reviewed-sign probe. A finished pass records it (infrastructure.json, `sign_typing`), and bikesafe.run types a
    ride's signs again - from the text and CLIP features already saved - when it differs."""
    import hashlib

    settings = (CLIP_MODEL, TEMPLATES, SIGN_TYPES, TEXT_RULES, FUZZY_WORDS, sorted(FUZZY_WORDS_2), FUZZY_PHRASES,
                SPEED.pattern, MIN_TYPE_H, MIN_TYPE_CONF, SURE_TYPE_CONF, sorted(TEXT_ONLY), MIN_ASPECT, MIN_CONTRAST,
                VIEWS_PER_SIGN, NOT_A_SIGN_MIN)
    digest = hashlib.sha1(repr(settings).encode())
    if probe_path.exists():
        digest.update(probe_path.read_bytes())
    return digest.hexdigest()[:12]


def load_features(path: Path, crops_path: Path) -> dict[int, np.ndarray]:
    """CLIP image features of a ride's saved sign crops (crop row -> vector), if they were computed with the current
    CLIP model after the crops were saved; otherwise empty."""
    if not path.exists() or (crops_path.exists() and path.stat().st_mtime < crops_path.stat().st_mtime):
        return {}
    with np.load(path, allow_pickle=False) as data:
        if str(data["model"]) != "/".join(CLIP_MODEL):
            return {}
        return dict(zip(data["rows"].astype(int).tolist(), data["features"]))


def save_features(path: Path, features: dict[int, np.ndarray]) -> None:
    rows = np.array(sorted(features), np.int64)
    np.savez(path, rows=rows, features=np.stack([features[r] for r in rows]).astype(np.float16),
             model=np.array("/".join(CLIP_MODEL)))


def train_probe(labels, crops: list[np.ndarray], typer, groups) -> tuple[dict | None, dict]:
    """Logistic-regression probe on CLIP features of reviewed signs, and its leave-one-ride-out accuracy against
    zero-shot on the same signs. `labels` are the reviewed types (other_sign = "type unknown" is left out)."""
    from sklearn.linear_model import LogisticRegression

    labels, groups = np.asarray(labels, dtype=object), np.asarray(groups, dtype=object)
    keep = np.isin(labels, list(SIGN_TYPES))
    labels, groups = labels[keep], groups[keep]
    crops = [c for c, k in zip(crops, keep) if k]
    report: dict = {"n_labels": int(len(labels)), "types": {str(k): int(v) for k, v in
                                                           zip(*np.unique(labels, return_counts=True))}}
    if len(labels) < MIN_PROBE_LABELS or len(set(labels)) < 2:
        return None, report | {"trained": False}
    feats = typer.embed(crops)
    make = lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")  # noqa: E731
    names = np.array(list(SIGN_TYPES))
    logits = 100 * feats @ typer.text.cpu().numpy().T
    zero_probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    zero_probs /= zero_probs.sum(axis=1, keepdims=True)
    zero = names[zero_probs.argmax(axis=1)]
    held = {w: zero.copy() for w in PROBE_WEIGHTS}
    validated = np.zeros(len(labels), bool)
    for g in np.unique(groups):
        test = groups == g
        if len(set(labels[~test])) < 2:
            continue
        probe = {"model": make().fit(feats[~test], labels[~test])}
        validated |= test
        for w in PROBE_WEIGHTS:
            held[w][test] = names[blend_probe(zero_probs[test], feats[test], probe, w).argmax(axis=1)]
    if not validated.any():  # reviewed signs from one ride only: nothing to validate the probe on
        return None, report | {"trained": False, "reason": "reviewed signs from fewer than two rides"}
    accuracy = {w: float((held[w][validated] == labels[validated]).mean()) for w in PROBE_WEIGHTS}
    best = max(PROBE_WEIGHTS, key=lambda w: (accuracy[w], -w))  # ties: the smaller weight, so no gain means zero-shot
    if best == 0.0:
        return None, report | {"trained": False, "reason": "the probe did not type held-out rides better",
                               "zero_shot_accuracy": round(accuracy[0.0], 3),
                               "accuracy_by_weight": {str(w): round(a, 3) for w, a in accuracy.items()}}
    report |= {"trained": True, "zero_shot_accuracy": round(accuracy[0.0], 3), "weight": best,
               "with_probe_accuracy_lovo": round(accuracy[best], 3),
               "accuracy_by_weight": {str(w): round(a, 3) for w, a in accuracy.items()}}
    return {"model": make().fit(feats, labels), "n_labels": int(len(labels)), "weight": best}, report


def read_signs(objects, crops: dict[int, np.ndarray], device: str = "cpu", fp16: bool = True,
               clip_model: tuple[str, str] = CLIP_MODEL, ocr_model: str | None = OCR_MODEL,
               typer=None, ocr=None, texts: dict[int, str] | None = None,
               features: dict[int, np.ndarray] | None = None, on_features=None) -> dict[int, dict]:
    """Category, text, posted speed limit and type confidence per sign object id.

    `objects` holds the sign detections (index = key into `crops`, with object_id, y1, y2, stop_confirmed); each
    object is typed from its VIEWS_PER_SIGN largest views and read on the largest one. CLIP runs over every sign
    first and is released before Florence-2 is loaded, so the two never share memory (`typer` / `ocr` replace them).
    `texts` (object id -> text) skips the reading, to type signs again with a new probe or prompts. `features` (crop
    key -> CLIP image features) are reused where present and filled in for the other views, so typing the signs again
    needs only CLIP's text side; `on_features(features)` is called once they are complete, before the slower reading,
    so a reading that fails does not lose them.
    """
    views = {}
    for object_id, group in objects.groupby("object_id"):
        ranked = group.assign(h=group.y2 - group.y1).sort_values("h", ascending=False)
        ranked = ranked[ranked.index.isin(list(crops))].head(VIEWS_PER_SIGN)
        if len(ranked):
            top = ranked.iloc[0]
            views[int(object_id)] = (list(ranked.index), float(top.h), bool(group.stop_confirmed.any()),
                                     float((top.x2 - top.x1) / max(top.h, 1.0)), contrast(crops[ranked.index[0]]))
    if not views:
        return {}
    typer = typer or SignTyper(device, fp16, clip_model, probe=load_probe())
    features = {} if features is None else features
    missing = sorted({i for idx, *_ in views.values() for i in idx} - set(features))
    step = embed_batch(device)
    for start in range(0, len(missing), step):
        batch = missing[start:start + step]
        features.update(zip(batch, typer.embed([crops[i] for i in batch])))
    if missing:
        print(f"CLIP features of {len(missing)} sign views ({len(views)} signs)", flush=True)
        if on_features is not None:
            on_features(features)
    probs = {oid: typer.probs(np.stack([features[i] for i in idx])).mean(axis=0) for oid, (idx, *_) in views.items()}
    del typer
    free_gpu(device)
    to_read = {oid: idx[0] for oid, (idx, h, *_) in views.items() if h >= MIN_OCR_H}
    if texts is not None:
        to_read = {}
    texts = dict(texts or {})
    if to_read and (ocr is not None or ocr_model):
        ocr = ocr or SignOCR(device, fp16, str(ocr_model))
        texts = {}
        for n, (oid, i) in enumerate(to_read.items(), 1):
            texts[oid] = ocr.read_text(crops[i])
            if n % OCR_REPORT == 0 or n == len(to_read):
                print(f"read {n} of {len(to_read)} signs", flush=True)
        del ocr
        free_gpu(device)
    out = {}
    for oid, (_, height, stop_confirmed, aspect, view_contrast) in views.items():
        text = texts.get(oid, "")
        category, conf = decide(probs[oid], text, stop_confirmed, height, aspect, view_contrast)
        out[oid] = {"category": category, "sign_text": text, "type_conf": round(conf, 3),
                    "speed_limit": speed_limit(text) if category == "speed_limit" else None}
    return out
