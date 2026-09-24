"""Train a compact ReID encoder, but only from human-reviewed triplets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class Triplets(Dataset):
    def __init__(self, rows: pd.DataFrame, root: Path, training: bool) -> None:
        self.rows = rows.reset_index(drop=True)
        self.root = root
        ops = [transforms.Resize((224, 224))]
        if training:
            ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1, 0.03)]
        ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        images = []
        for key in ["anchor", "positive", "negative"]:
            image = Image.open(self.root / row[key]).convert("RGB")
            images.append(self.transform(image))
        return tuple(images)


class Encoder(nn.Module):
    def __init__(self, embedding_size: int = 128) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(features, embedding_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(image)), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()

    rows = pd.read_csv(args.manifest)
    if "status" not in rows or not (rows["status"] == "reviewed").any():
        raise SystemExit("No human-reviewed triplets. Review identities and set status=reviewed before training.")
    rows = rows[rows["status"] == "reviewed"].copy()
    train_rows = rows[rows["split"] == "train"]
    val_rows = rows[rows["split"] == "validation"]
    if len(train_rows) < 20 or len(val_rows) < 5:
        raise SystemExit("Need at least 20 reviewed train and 5 reviewed validation triplets")

    root = args.manifest.parent
    loaders = {
        "train": DataLoader(Triplets(train_rows, root, True), batch_size=args.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(Triplets(val_rows, root, False), batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        record = {"epoch": epoch}
        for phase in ["train", "validation"]:
            model.train(phase == "train")
            total_loss = 0.0
            count = 0
            for anchor, positive, negative in loaders[phase]:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    loss = loss_fn(model(anchor), model(positive), model(negative))
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total_loss += float(loss) * len(anchor)
                count += len(anchor)
            record[f"{phase}_loss"] = total_loss / max(1, count)
        history.append(record)
        print(record, flush=True)
        if record["validation_loss"] < best_val:
            best_val = record["validation_loss"]
            torch.save({"model": model.state_dict(), "embedding_size": 128}, args.output_dir / "best_reid_encoder.pt")

    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
