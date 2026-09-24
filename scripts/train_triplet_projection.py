"""Train a small triplet-loss projection on frozen ImageNet features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms


def triplet_accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:
    positive_distance = torch.linalg.vector_norm(anchor - positive, dim=1)
    negative_distance = torch.linalg.vector_norm(anchor - negative, dim=1)
    return float((positive_distance < negative_distance).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=555)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rows = pd.read_csv(args.manifest)
    rows = rows[rows["status"] == "reviewed"].copy()
    if len(rows) < 25 or not {"train", "validation"}.issubset(set(rows["split"])):
        raise SystemExit("Reviewed train and validation triplets are required")

    root = args.manifest.parent
    unique_paths = sorted(set(rows["anchor"]) | set(rows["positive"]) | set(rows["negative"]))
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    feature_size = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone.eval()

    features = {}
    with torch.inference_mode():
        for start in range(0, len(unique_paths), args.batch_size):
            paths = unique_paths[start : start + args.batch_size]
            batch = torch.stack([transform(Image.open(root / path).convert("RGB")) for path in paths])
            encoded = F.normalize(backbone(batch), dim=1)
            features.update({path: vector for path, vector in zip(paths, encoded)})
            print(f"Encoded {min(start + len(paths), len(unique_paths))}/{len(unique_paths)} crops", flush=True)

    def tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(torch.stack([features[path] for path in frame[column]]) for column in ["anchor", "positive", "negative"])

    train = tensors(rows[rows["split"] == "train"])
    validation = tensors(rows[rows["split"] == "validation"])
    baseline_accuracy = triplet_accuracy(*validation)

    projection = nn.Linear(feature_size, args.embedding_size, bias=False)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        projection.train()
        for anchor, positive, negative in loader:
            a = F.normalize(projection(anchor), dim=1)
            p = F.normalize(projection(positive), dim=1)
            n = F.normalize(projection(negative), dim=1)
            loss = loss_fn(a, p, n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        projection.eval()
        with torch.inference_mode():
            va, vp, vn = (F.normalize(projection(tensor), dim=1) for tensor in validation)
            validation_loss = float(loss_fn(va, vp, vn))
            validation_accuracy = triplet_accuracy(va, vp, vn)
        history.append({"epoch": epoch, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy})
        print(f"epoch {epoch:03d} val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.3f}", flush=True)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save({
                "projection": projection.state_dict(), "backbone": "resnet18_imagenet1k_v1",
                "feature_size": feature_size, "embedding_size": args.embedding_size,
                "image_size": args.image_size, "margin": args.margin,
            }, args.output_dir / "best_reid_projection.pt")
        elif epoch - best_epoch >= args.patience:
            break

    checkpoint = torch.load(args.output_dir / "best_reid_projection.pt", weights_only=True)
    projection.load_state_dict(checkpoint["projection"])
    projection.eval()
    with torch.inference_mode():
        va, vp, vn = (F.normalize(projection(tensor), dim=1) for tensor in validation)
        final_accuracy = triplet_accuracy(va, vp, vn)
    metrics = {
        "reviewed_triplets": len(rows),
        "train_triplets": int((rows["split"] == "train").sum()),
        "validation_triplets": int((rows["split"] == "validation").sum()),
        "unique_crops": len(unique_paths),
        "baseline_validation_accuracy": round(baseline_accuracy, 4),
        "trained_validation_accuracy": round(final_accuracy, 4),
        "best_validation_loss": round(best_loss, 6),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "backbone_frozen": True,
        "seed": args.seed,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
