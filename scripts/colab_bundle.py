"""Run the sign / lane / kerb pass (bikesafe.infrastructure) on a Colab GPU for rides this machine is too slow for.

The pass is the one heavy GPU stage: about 6.5 s a sample on a GeForce MX250, 0.35 s on a Colab L4 (a 20-minute ride
in about 22 minutes instead of 5 hours). This prepares the run and brings the results back:

    python scripts/colab_bundle.py VID_20260223_090217_00_003 VID_20260209_183334_00_002
    python scripts/colab_bundle.py --fetch "G:/My Drive/bikesafe"     # afterwards

The first form writes an upload folder (--out): the code, the three small model files the pass loads, each ride's
perception and ego-motion inputs, videos_manifest.json (the byte size of each video, which the notebook waits for -
Google Drive lists a file before its content has arrived) and scripts/colab_infrastructure.ipynb. It also hard-links
the videos into --videos-out, which takes no disk space (it has to be on the same drive as the videos).

Then: put the upload folder in Google Drive as My Drive/bikesafe (copy it into the Google Drive for desktop mount, or
drag it onto drive.google.com); add the --videos-out folder in Google Drive for desktop (Preferences > My Computer >
Add folder > Sync with Google Drive), which uploads the videos straight from the originals; open
colab_infrastructure.ipynb in Colab, pick a GPU runtime and run all. Each ride starts as soon as its video is up.

--fetch copies each finished ride's results from the Drive folder into work/analysis/<ride>/, the sign crops before
their CLIP features so the saved features stay valid; bikesafe.run then carries on from there.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ["infrastructure_vocab.npz", "yolo11m.pt", "yoloe-26l-seg.pt"]
PERCEPTION_FILES = ["meta.json", "detections.parquet"]
ANALYSIS_FILES = ["egomotion_raw.parquet"]
# what a finished pass writes, in the order to copy it back (crops before their features, the coverage file last)
RESULTS = ["lanes.parquet", "vehicle_context.parquet", "sign_crops.parquet", "infrastructure.parquet", "sign_clip.npz",
           "infrastructure.json"]


def bundle(rides: list[str], out: Path, videos: Path, videos_out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "bikesafe", out / "code" / "bikesafe", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "requirements.txt", out / "code")
    (out / "code" / "models").mkdir(parents=True)
    for name in MODEL_FILES:
        shutil.copy2(ROOT / "models" / name, out / "code" / "models")
    for ride in rides:
        for folder, names in (("perception", PERCEPTION_FILES), ("analysis", ANALYSIS_FILES)):
            target = out / "work" / folder / ride
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(ROOT / "work" / folder / ride / name, target)
    sizes = {ride: (videos / f"{ride}.mp4").stat().st_size for ride in rides}
    (out / "videos_manifest.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    (out / "videos").mkdir()
    shutil.copy2(ROOT / "scripts" / "colab_infrastructure.ipynb", out)
    print(f"upload folder: {out} (as My Drive/bikesafe)")
    if videos.drive.upper() != videos_out.drive.upper():  # the videos are not on this disk: already in Google Drive
        print(f"videos: the notebook reads them from My Drive/bikesafe/videos ({videos} here)")
        return
    videos_out.mkdir(parents=True, exist_ok=True)
    for ride in rides:
        link = videos_out / f"{ride}.mp4"
        if not link.exists():
            os.link(videos / f"{ride}.mp4", link)
    print(f"videos to back up with Google Drive for desktop: {videos_out} ({sum(sizes.values()) / 1e9:.1f} GB)")


def fetch(drive: Path, analysis: Path) -> None:
    for coverage in sorted((drive / "work" / "analysis").glob("*/infrastructure.json")):
        ride = coverage.parent.name
        if not json.loads(coverage.read_text(encoding="utf-8")).get("complete"):
            print(f"{ride}: not finished on Colab, skipped")
            continue
        copied = []
        # only results newer than the local copy and different from it: copying an unchanged lane file would make the
        # track features look stale, and an older one would undo work done here since (signs typed again)
        for name in RESULTS:
            source, target = coverage.parent / name, analysis / ride / name
            if not source.exists() or (target.exists() and (target.stat().st_mtime >= source.stat().st_mtime
                                                            or target.read_bytes() == source.read_bytes())):
                continue
            shutil.copy(source, target)
            copied.append(name)
            time.sleep(1)  # distinct modification times: the features must be newer than the crops
        print(f"{ride}: {', '.join(copied) if copied else 'already up to date'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rides", nargs="*", help="Ride names (video file stems)")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "colab" / "bikesafe")
    parser.add_argument("--videos", type=Path, default=Path(os.environ.get("BIKESAFE_VIDEOS", ROOT / "videos")))
    parser.add_argument("--videos-out", type=Path, default=ROOT.parent / "bikesafe_videos",
                        help="Folder of hard links to the videos, for Google Drive for desktop to back up")
    parser.add_argument("--fetch", type=Path, default=None, help="The bikesafe folder on Drive to copy results from")
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    args = parser.parse_args()
    if args.fetch is not None:
        fetch(args.fetch, args.analysis)
    elif args.rides:
        bundle(args.rides, args.out, args.videos, args.videos_out)
    else:
        parser.error("give the rides to bundle, or --fetch")


if __name__ == "__main__":
    main()
