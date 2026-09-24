"""One-command corpus run for a GPU workstation.

    python -m bikesafe.run D:/rides --out-root D:/rides_out [--target-fps 12] [--render-minutes 1]

Stages (each resumable): GPU perception -> CPU ego-motion -> depth/lane pass -> geometry/track features -> typology models + exposure outputs ->
optional overlay clip per video. Uses the trained models in models/typology; without them it falls back to the
rule baseline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bikesafe.common import ROOT, list_videos, read_json


def run(args: list[str]) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--out-root", type=Path, default=ROOT / "work")
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--gmc", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame rate for the depth and lane pass")
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "typology")
    parser.add_argument("--render-minutes", type=float, default=0.0, help="Overlay clip length from each video's midpoint")
    args = parser.parse_args()

    perception = args.out_root / "perception"
    analysis = args.out_root / "analysis"
    corpus = args.out_root / "corpus"
    videos = [args.videos] if args.videos.is_file() else list_videos(args.videos)
    run(["bikesafe.perceive", *map(str, videos), "--out", str(perception), "--target-fps", str(args.target_fps), "--gmc", args.gmc])
    video_dir = str(args.videos if args.videos.is_dir() else args.videos.parent)
    run(["bikesafe.egomotion", *[str(perception / v.stem) for v in videos], "--analysis", str(analysis),
         "--videos", video_dir])
    run(["bikesafe.scene3d", *[str(perception / v.stem) for v in videos], "--analysis", str(analysis),
         "--videos", video_dir, "--sample-fps", str(args.sample_fps)])
    pending = [perception / v.stem for v in videos if not (analysis / v.stem / "tracks.parquet").exists()]
    if pending:
        run(["bikesafe.tracks", *map(str, pending), "--out", str(analysis)])
    run(["bikesafe.exposure", "--analysis", str(analysis), "--perception", str(perception), "--models", str(args.models),
         "--out", str(corpus)])
    if args.render_minutes > 0:
        for v in videos:
            meta = read_json(perception / v.stem / "meta.json")
            start = max(0.0, meta["source_frames"] / meta["source_fps"] / 2 - 30 * args.render_minutes)
            run(["bikesafe.render", v.stem, "--videos", str(v.parent), "--analysis", str(analysis), "--corpus", str(corpus),
                 "--start-s", f"{start:.0f}", "--duration-s", f"{60 * args.render_minutes:.0f}",
                 "--out", str(corpus / v.stem / "typology_overlay.mp4")])


if __name__ == "__main__":
    main()
