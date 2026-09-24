"""Riding-context and vehicle-appearance scores from the CLIP embeddings stored by the perception pass.

Scene (per second): separated_path / bike_lane / shared_road, zero-shot from prompts or from a linear probe
trained on labelled frames (bikesafe.train). Vehicle crops: viewpoint (rear/front/side) and emergency-vehicle
scores, averaged per track.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from bikesafe.common import ROOT, SCENES

SCENE_PROMPTS = {
    "separated_path": [
        "a photo taken while riding a bicycle on a bike path separated from car traffic",
        "a photo of a paved walkway on a university campus",
        "a photo of a pedestrian plaza with no cars",
        "a photo taken while riding a bicycle on a sidewalk",
    ],
    "bike_lane": [
        "a photo taken while riding a bicycle in a painted bike lane next to car traffic",
        "a photo of a green painted bike lane on a city street",
        "a photo of a bicycle lane with a white line separating it from cars",
    ],
    "shared_road": [
        "a photo taken while riding a bicycle in a traffic lane shared with cars",
        "a photo of a city street with no bike lane taken from the driver's position",
        "a photo of an intersection taken from the middle of the road",
    ],
}
VIEW_PROMPTS = {
    "view_rear": ["a photo of the back of a car", "the rear of a vehicle with tail lights"],
    "view_front": ["a photo of the front of a car", "the front of a vehicle with headlights and grille"],
    "view_side": ["a photo of the side of a car", "a side view of a parked car"],
}
VEHICLE_TYPE_PROMPTS = {
    "police_car": ["a police car", "a police patrol car with light bar"],
    "ambulance": ["an ambulance", "a paramedic ambulance van"],
    "fire_truck": ["a fire truck", "a fire engine"],
    "passenger_car": ["a parked sedan", "a passenger car"],
    "suv": ["a sport utility vehicle", "a crossover SUV"],
    "pickup": ["a pickup truck"],
    "van": ["a delivery van", "a cargo van"],
    "bus": ["a city bus"],
}
EMERGENCY_TYPES = ["police_car", "ambulance", "fire_truck"]
TEXT_CACHE = ROOT / "models" / "clip_prompt_embeddings.npz"


@lru_cache(maxsize=1)
def prompt_embeddings(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> dict[str, np.ndarray]:
    """Mean normalised text embedding per prompt group, cached to disk so analysis needs no text model."""
    groups = {**SCENE_PROMPTS, **VIEW_PROMPTS, **VEHICLE_TYPE_PROMPTS}
    if TEXT_CACHE.exists():
        cached = dict(np.load(TEXT_CACHE))
        if set(cached) == set(groups):
            return cached
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    out = {}
    with torch.inference_mode():
        for name, prompts in groups.items():
            feats = torch.nn.functional.normalize(model.encode_text(tokenizer(prompts)).float(), dim=-1)
            mean = feats.mean(0)
            out[name] = (mean / mean.norm()).numpy()
    TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TEXT_CACHE, **out)
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def zero_shot(embeddings: np.ndarray, names: list[str], temperature: float = 100.0) -> np.ndarray:
    text = prompt_embeddings()
    weights = np.stack([text[n] for n in names])
    return softmax(temperature * embeddings.astype(np.float32) @ weights.T)


def scene_features(clip_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """(frames, features) where features = [full-frame embedding, road-ahead embedding]."""
    z = np.load(clip_npz)
    return z["frame"], np.hstack([z["full"], z["road"]]).astype(np.float32)


def scene_probabilities(clip_npz: Path, fps_source: float, probe=None, smooth_s: float = 5.0) -> pd.DataFrame:
    """Per-second scene probabilities, from a trained probe if given, otherwise zero-shot on both views."""
    frames, feats = scene_features(clip_npz)
    if probe is not None:
        probs = probe.predict_proba(feats)
        classes = list(probe.classes_)
        probs = probs[:, [classes.index(s) for s in SCENES]]
    else:
        half = feats.shape[1] // 2
        probs = 0.5 * (zero_shot(feats[:, :half], SCENES) + zero_shot(feats[:, half:], SCENES))
    df = pd.DataFrame(probs, columns=[f"p_{s}" for s in SCENES])
    df.insert(0, "frame", frames)
    df.insert(1, "time_s", frames / fps_source)
    window = max(1, int(round(smooth_s)))
    cols = [f"p_{s}" for s in SCENES]
    df[cols] = df[cols].rolling(window, center=True, min_periods=1).mean()
    df["scene"] = np.array(SCENES)[df[cols].to_numpy().argmax(axis=1)]
    return df


def crop_scores(clip_npz: Path) -> pd.DataFrame:
    """Per-track mean viewpoint and emergency-vehicle probabilities from the throttled crop embeddings."""
    z = np.load(clip_npz)
    if len(z["crop"]) == 0:
        return pd.DataFrame(columns=["track_id", "view_rear", "view_front", "view_side", "emergency", "n_crops"])
    emb = z["crop"].astype(np.float32)
    views = zero_shot(emb, list(VIEW_PROMPTS))
    types = zero_shot(emb, list(VEHICLE_TYPE_PROMPTS))
    emergency = types[:, [list(VEHICLE_TYPE_PROMPTS).index(t) for t in EMERGENCY_TYPES]].sum(axis=1)
    df = pd.DataFrame(views, columns=list(VIEW_PROMPTS))
    df["emergency"] = emergency
    df["track_id"] = z["crop_track"]
    agg = df.groupby("track_id").agg(
        view_rear=("view_rear", "mean"), view_front=("view_front", "mean"), view_side=("view_side", "mean"),
        emergency=("emergency", "max"), n_crops=("emergency", "size"),
    )
    return agg.reset_index()
