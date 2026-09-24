"""Freeze user-approved review sets without overwriting the original pseudo-labels."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def finalize_detection(root: Path) -> dict:
    manifest_path = root / "review_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["status"] = "reviewed"
    manifest["split"] = ["validation" if index % 5 == 0 else "train" for index in range(len(manifest))]

    counts = {"train": 0, "validation": 0}
    for _, row in manifest.iterrows():
        split = row["split"]
        image_source = root / Path(row["image"])
        pseudo_source = root / Path(row["pseudo_label"])
        image_dest = root / "images" / ("val" if split == "validation" else "train") / image_source.name
        label_dest = root / "labels" / ("val" if split == "validation" else "train") / f"{image_source.stem}.txt"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_dest)
        reviewed_lines = []
        if pseudo_source.exists():
            for line in pseudo_source.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    reviewed_lines.append(" ".join(parts[:5]))
        label_dest.write_text("\n".join(reviewed_lines) + ("\n" if reviewed_lines else ""), encoding="utf-8")
        counts[split] += 1

    manifest.to_csv(manifest_path, index=False)
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: car\n  1: motorcycle\n  2: bus\n  3: truck\n",
        encoding="utf-8",
    )
    return {"items": len(manifest), **counts}


def finalize_tracking(root: Path) -> dict:
    source = root / "pseudo_tracks_for_review.csv"
    data = pd.read_csv(source)
    data["review_status"] = "reviewed"
    destination = root / "approved_tracks.csv"
    data.to_csv(destination, index=False)
    return {
        "rows": len(data),
        "frames": int(data["frame"].nunique()),
        "track_ids": int(data.loc[data["track_id"] >= 0, "track_id"].nunique()),
    }


def finalize_triplets(root: Path) -> dict:
    path = root / "triplets.csv"
    data = pd.read_csv(path)
    data["status"] = "reviewed"
    data.to_csv(path, index=False)
    return {
        "triplets": len(data),
        "train": int((data["split"] == "train").sum()),
        "validation": int((data["split"] == "validation").sum()),
        "anchor_track_ids": int(data["anchor_track_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detection_review", type=Path)
    parser.add_argument("tracking_review", type=Path)
    parser.add_argument("triplet_review", type=Path)
    args = parser.parse_args()

    record = {
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_source": "User stated: human review approved",
        "detection": finalize_detection(args.detection_review),
        "tracking": finalize_tracking(args.tracking_review),
        "triplets": finalize_triplets(args.triplet_review),
        "caveat": "Approval freezes the reviewed references. Tracking metrics are not computed against an identical copied prediction file.",
    }
    for root in [args.detection_review, args.tracking_review, args.triplet_review]:
        (root / "APPROVAL_RECORD.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
