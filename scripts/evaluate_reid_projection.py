"""Evaluate a trained ReID projection and calibrate a cosine-similarity threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, weights_only=True)
    rows = pd.read_csv(args.manifest)
    rows = rows[(rows["status"] == "reviewed") & (rows["split"] == args.split)].copy()
    if rows.empty:
        raise SystemExit(f"No reviewed rows for split {args.split}")
    root = args.manifest.parent
    paths = sorted(set(rows["anchor"]) | set(rows["positive"]) | set(rows["negative"]))
    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval()
    projection = nn.Linear(checkpoint["feature_size"], checkpoint["embedding_size"], bias=False)
    projection.load_state_dict(checkpoint["projection"])
    projection.eval()

    embeddings = {}
    with torch.inference_mode():
        for path in paths:
            image = transform(Image.open(root / path).convert("RGB")).unsqueeze(0)
            embeddings[path] = F.normalize(projection(F.normalize(backbone(image), dim=1)), dim=1)[0]

    scored = []
    for _, row in rows.iterrows():
        anchor = embeddings[row["anchor"]]
        positive_score = float(torch.dot(anchor, embeddings[row["positive"]]))
        negative_score = float(torch.dot(anchor, embeddings[row["negative"]]))
        scored += [
            {"anchor": row["anchor"], "candidate": row["positive"], "same_vehicle": 1, "cosine_similarity": positive_score},
            {"anchor": row["anchor"], "candidate": row["negative"], "same_vehicle": 0, "cosine_similarity": negative_score},
        ]
    scores = pd.DataFrame(scored)
    thresholds = np.linspace(-1, 1, 4001)
    labels = scores["same_vehicle"].to_numpy(dtype=bool)
    values = scores["cosine_similarity"].to_numpy()
    accuracies = np.array([((values >= threshold) == labels).mean() for threshold in thresholds])
    best_threshold = float(thresholds[int(accuracies.argmax())])
    predictions = values >= best_threshold
    scores["predicted_same"] = predictions.astype(int)
    scores.to_csv(args.output_dir / "validation_pair_scores.csv", index=False)

    positive = scores.loc[scores["same_vehicle"] == 1, "cosine_similarity"]
    negative = scores.loc[scores["same_vehicle"] == 0, "cosine_similarity"]
    metrics = {
        "split": args.split,
        "triplets": len(rows),
        "pairs": len(scores),
        "calibrated_threshold": round(best_threshold, 4),
        "pair_accuracy_on_calibration_split": round(float(accuracies.max()), 4),
        "false_positives": int(((predictions == 1) & (labels == 0)).sum()),
        "false_negatives": int(((predictions == 0) & (labels == 1)).sum()),
        "mean_positive_similarity": round(float(positive.mean()), 4),
        "mean_negative_similarity": round(float(negative.mean()), 4),
        "triplet_ordering_accuracy": round(float((positive.to_numpy() > negative.to_numpy()).mean()), 4),
        "caveat": "Threshold accuracy is measured on the same split used to calibrate the threshold; validate it on a new sequence before deployment.",
    }
    (args.output_dir / "reid_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "REID_THRESHOLD_EVALUATION.md").write_text(
        f"""# ReID threshold evaluation

The best seed-555 triplet projection was evaluated on {metrics['triplets']} approved
validation triplets ({metrics['pairs']} same/different pairs).

| Measure | Result |
|---|---:|
| Triplet ordering accuracy | {100 * metrics['triplet_ordering_accuracy']:.1f}% |
| Calibrated cosine threshold | {metrics['calibrated_threshold']:.4f} |
| Pair accuracy on calibration split | {100 * metrics['pair_accuracy_on_calibration_split']:.1f}% |
| False positive matches | {metrics['false_positives']} |
| False negative matches | {metrics['false_negatives']} |
| Mean same-vehicle similarity | {metrics['mean_positive_similarity']:.3f} |
| Mean different-vehicle similarity | {metrics['mean_negative_similarity']:.3f} |

The projection ranks the correct vehicle ahead of a sampled negative reliably, but a single
hard threshold still produces too many false matches. Use the model to rank proposed track
fragment links for review; do not automatically merge identities from this threshold.
""",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(values), max(values), 28)
    ax.hist(negative, bins=bins, alpha=0.65, label="Different vehicle", color="#d46a43")
    ax.hist(positive, bins=bins, alpha=0.65, label="Same vehicle", color="#267ba8")
    ax.axvline(best_threshold, color="#222222", linestyle="--", label=f"Calibrated threshold {best_threshold:.3f}")
    ax.set(title="ReID cosine similarity on approved validation pairs", xlabel="Cosine similarity", ylabel="Pairs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "similarity_distribution.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
