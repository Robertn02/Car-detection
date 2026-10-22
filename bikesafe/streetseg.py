"""Street-scene segmentation with an open model trained on street-level imagery: Mask2Former (Swin-L) on Mapillary
Vistas, `facebook/mask2former-swin-large-mapillary-vistas-semantic` (downloaded from Hugging Face on first use).

Mapillary Vistas is the standard dataset for this camera's point of view: 25,000 street-level photos from six
continents, many taken from bikes and on foot, labelled with 65 classes. Three of them are what bikesafe.infrastructure
needs and what the open-vocabulary detector did badly:

* **Traffic Sign (Front)** - signs facing the camera, kept apart from Traffic Sign (Back), Billboard and Banner. On the
  ride frames it found speed-limit, street-name, no-turn, no-parking and bike-lane signs that YOLOE either missed or
  could only call "other sign", and it labels shop signs and billboards as such instead of as traffic signs.
* **Lane Marking - General** and **Lane Marking - Crosswalk** - painted lines, pixel by pixel, without the cracks,
  kerbs, shadow edges and sunlit patches a brightness filter also picks up, and with crosswalk bars kept apart.
* **Bike Lane** - the surface of a painted bike lane.

The model runs on a frame resized to SEG_HEIGHT rows (multiple of 32), in half precision on CUDA.
"""

from __future__ import annotations

import cv2
import numpy as np

from bikesafe.common import is_cuda

VISTAS_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
SEG_HEIGHT = 768  # rows the frame is resized to; 1088 finds smaller (more distant) signs at ~2x the cost

# Mapillary Vistas v1.2 class ids (model.config.id2label)
CURB, BIKE_LANE, CROSSWALK_PLAIN, PARKING, ROAD, SIDEWALK = 2, 7, 8, 10, 13, 15
CROSSWALK_MARKING, LANE_MARKING = 23, 24
BILLBOARD, TRAFFIC_LIGHT, SIGN_BACK, SIGN_FRONT = 35, 48, 49, 50
MIN_SIGN_AREA = 12  # px at segmentation resolution; smaller blobs are too far away to read or count


class StreetSegmenter:
    """Per-pixel Vistas class (uint8) and its probability for a BGR frame, at segmentation resolution."""

    def __init__(self, device: str = "cpu", fp16: bool = True, height: int = SEG_HEIGHT, model: str = VISTAS_MODEL) -> None:
        import torch
        from transformers import Mask2FormerForUniversalSegmentation
        from transformers.utils import logging as hf_logging

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if fp16 and is_cuda(device) else torch.float32
        self.height = int(round(height / 32)) * 32
        # The load report lists the Swin backbone's final layer norm as missing: Mask2Former reads the backbone through
        # its per-stage norms and never uses that layer, so the report is noise.
        hf_logging.set_verbosity_error()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model, dtype=self.dtype).to(device).eval()
        self.names = dict(self.model.config.id2label)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def size_for(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """(height, width) the frame is resized to: SEG_HEIGHT rows, width keeping the aspect, multiples of 32."""
        width = int(round(shape[1] * self.height / shape[0] / 32)) * 32
        return self.height, width

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        height, width = self.size_for(image.shape)
        rgb = np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = ((x - self.mean) / self.std).to(self.dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=x)
            # semantic map as in Mask2Former: class probabilities of each query weighted by its mask
            classes = out.class_queries_logits.float().softmax(-1)[..., :-1]
            masks = out.masks_queries_logits.float().sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", classes, masks)
            seg = torch.nn.functional.interpolate(seg, size=(height, width), mode="bilinear", align_corners=False)[0]
            prob, label = seg.max(0)
            prob = prob / seg.sum(0).clamp_min(1e-6)
        return label.byte().cpu().numpy(), prob.half().cpu().numpy()


CURB_CUT, PEDESTRIAN_AREA, SERVICE_LANE, TERRAIN, MANHOLE, CATCH_BASIN, POTHOLE = 9, 11, 14, 29, 41, 36, 43
VEHICLES = (52, 54, 55, 56, 57, 59, 60, 61, 62)  # bicycle, bus, car, caravan, motorcycle, other vehicle, trailer, truck
ROADWAY = (ROAD, BIKE_LANE, CROSSWALK_PLAIN, SERVICE_LANE, CROSSWALK_MARKING, LANE_MARKING, MANHOLE, CATCH_BASIN, POTHOLE)
KERBSIDE = (CURB, CURB_CUT, SIDEWALK, PEDESTRIAN_AREA, PARKING, TERRAIN)
CONTEXT_SCAN_M = 4.0  # how far beside a vehicle the kerb is looked for
CONTEXT_COLUMNS = ["frame", "track_id", "kerb_gap_m", "outer_vehicle", "inner_road_m", "under_offroad"]


def vehicle_context(labels: np.ndarray, boxes: np.ndarray, track_ids, u_foe: float, horizon: float,
                    lateral_height_m: float) -> list[dict]:
    """Where each vehicle stands, from the full-resolution label map: what a parked car has and a car queued in a
    traffic lane does not.

    Along the rows at the vehicle's wheels, scanning outward (away from the rider's path) from the box edge:
    `kerb_gap_m` is the distance to the first kerb / sidewalk / parking / verge pixel (a parked car is within half a
    metre of the kerb; NaN if none within CONTEXT_SCAN_M or the scan leaves the frame), and `outer_vehicle` whether
    another vehicle comes first (a car in a traffic lane next to a parked row). Scanning inward, `inner_road_m` is how
    much roadway lies between the vehicle and the rider's side. `under_offroad` is the share of the strip under the
    wheels that is not roadway (driveways, lots, sidewalk). Metres use the flat-road lateral scale at the wheel row.
    """
    height, width = labels.shape[:2]
    out = []
    for (x1, y1, x2, y2), tid in zip(np.asarray(boxes, float).reshape(-1, 4), track_ids):
        rows_below = y2 - horizon
        if rows_below < 6 or y2 >= height - 2 or y2 - y1 < 12:
            continue
        m_per_px = lateral_height_m / rows_below
        reach = int(CONTEXT_SCAN_M / m_per_px)
        rows = [int(y2) - k for k in (1, 3, 6) if int(y2) - k > y1]
        outward = 1 if (x1 + x2) / 2 > u_foe else -1
        edge_out, edge_in = (int(x2), int(x1)) if outward > 0 else (int(x1), int(x2))
        gaps, blocked, inner = [], 0, []
        for r in rows:
            line = labels[r]
            seq = line[edge_out: min(width, edge_out + reach)] if outward > 0 else line[max(0, edge_out - reach): edge_out][::-1]
            kerb = np.flatnonzero(np.isin(seq, KERBSIDE))
            car = np.flatnonzero(np.isin(seq, VEHICLES))
            if len(car) and (not len(kerb) or car[0] < kerb[0]) and car[0] * m_per_px < 1.5:
                blocked += 1
            elif len(kerb):
                gaps.append(kerb[0] * m_per_px)
            inside = line[max(0, edge_in - reach): edge_in][::-1] if outward > 0 else line[edge_in: min(width, edge_in + reach)]
            not_road = np.flatnonzero(~np.isin(inside, ROADWAY))
            inner.append((not_road[0] if len(not_road) else len(inside)) * m_per_px)
        strip = labels[int(y2): min(height, int(y2 + 0.08 * (y2 - y1)) + 2), max(0, int(x1)): int(x2)]
        out.append({"track_id": int(tid), "kerb_gap_m": float(np.median(gaps)) if gaps else np.nan,
                    "outer_vehicle": blocked / max(len(rows), 1), "inner_road_m": float(np.median(inner)) if inner else np.nan,
                    "under_offroad": float((~np.isin(strip, ROADWAY)).mean()) if strip.size else np.nan})
    return out


def full_resolution(labels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Label map resized to the frame (nearest neighbour, labels stay labels)."""
    return cv2.resize(labels, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def sign_boxes(labels: np.ndarray, prob: np.ndarray, shape: tuple[int, ...], cls: int = SIGN_FRONT,
               min_area: int = MIN_SIGN_AREA) -> list[dict]:
    """One box (frame pixels) per connected blob of `cls`, with the blob's mean class probability as confidence.

    A sign assembly on one pole (a street name above a stop sign, a parking plate below a bike-lane sign) can come out
    as one blob; the reader then types it by the sign that dominates the crop.
    """
    n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == cls).astype(np.uint8), connectivity=8)
    sy, sx = shape[0] / labels.shape[0], shape[1] / labels.shape[1]
    found = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue
        found.append({"x1": float(x * sx), "y1": float(y * sy), "x2": float((x + w) * sx), "y2": float((y + h) * sy),
                      "conf": float(prob[comp == k].astype(np.float32).mean()), "seg_area": int(area)})
    return found
