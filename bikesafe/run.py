"""One-command corpus run for a GPU workstation.

    python -m bikesafe.run D:/rides --out-root D:/rides_out [--target-fps 12] [--render-minutes 1] [--jobs 4]

Stages (each resumable): GPU perception -> CPU ego-motion -> depth/lane pass -> lights, signs and bike lanes ->
geometry/track features -> typology models + exposure outputs -> optional overlay clip per video. Uses the trained
models in models/typology; without them it falls back to the rule baseline.

Faster than running the stages by hand:
  * CPU-only stages (ego-motion, track features) run for several rides at once (--jobs, default: CPU cores / 2).
  * --hw-decode decodes video on the GPU's video engine in every stage (NVDEC etc.), which matters for 4K/360 exports.
  * FP16 inference is the default on CUDA for the detector, CLIP and the depth model (--fp32 to switch it off).
Track features written by an older pipeline version are recomputed automatically; everything else is kept.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bikesafe.common import HW_DECODE_ENV, ROOT, list_videos, read_json

TIMINGS: list[tuple[str, float]] = []


def run(args: list[str], env: dict | None = None, label: str | None = None) -> None:
    print(">", " ".join(args), flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT, env=env)
    TIMINGS.append((label or f"{args[0]} {Path(args[1]).name}", time.perf_counter() - started))


def run_parallel(commands: list[list[str]], jobs: int, env: dict | None) -> None:
    """Independent per-ride commands, `jobs` at a time (each is its own process, so CPU stages scale with cores)."""
    if jobs <= 1 or len(commands) <= 1:
        for command in commands:
            run(command, env)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in [pool.submit(run, command, env) for command in commands]:
            future.result()


def tracks_outdated(analysis_dir: Path) -> bool:
    """True when track features are missing or were written before the current feature version (or before the
    lane-paint features they can now include)."""
    from bikesafe.tracks import FEATURES_VERSION

    calib_path = analysis_dir / "calibration.json"
    if not (analysis_dir / "tracks.parquet").exists() or not calib_path.exists():
        return True
    calib = read_json(calib_path)
    if calib.get("features_version", 1) < FEATURES_VERSION:
        return True
    return (analysis_dir / "lanes.parquet").exists() and not calib.get("lane_features", False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--out-root", type=Path, default=ROOT / "work")
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--gmc", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame rate for the depth, lane and sign passes")
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "typology")
    parser.add_argument("--render-minutes", type=float, default=0.0, help="Overlay clip length from each video's midpoint")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Rides processed at once in the CPU-only stages")
    parser.add_argument("--hw-decode", action="store_true", help="Decode video with the GPU's video engine when available")
    parser.add_argument("--fp32", action="store_true", help="Full precision on the GPU (FP16 is the default on CUDA)")
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip the depth pass (its lane counts are superseded by the lane-paint features)")
    parser.add_argument("--skip-infrastructure", action="store_true", help="Skip lights, signs and bike lanes")
    parser.add_argument("--infra-model", default=None, help="YOLOE-26 weights for lights, signs and stencils")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.hw_decode:
        env[HW_DECODE_ENV] = "1"
    precision = ["--fp32"] if args.fp32 else []
    perception = args.out_root / "perception"
    analysis = args.out_root / "analysis"
    corpus = args.out_root / "corpus"
    videos = [args.videos] if args.videos.is_file() else list_videos(args.videos)
    video_dir = str(args.videos if args.videos.is_dir() else args.videos.parent)
    perception_dirs = [str(perception / v.stem) for v in videos]
    started = time.perf_counter()

    rides = f"({len(videos)} rides)"
    run(["bikesafe.perceive", *map(str, videos), "--out", str(perception), "--target-fps", str(args.target_fps),
         "--gmc", args.gmc, *precision], env, f"bikesafe.perceive {rides}")
    run_parallel([["bikesafe.egomotion", p, "--analysis", str(analysis), "--videos", video_dir] for p in perception_dirs],
                 args.jobs, env)
    if not args.skip_depth:
        run(["bikesafe.scene3d", *perception_dirs, "--analysis", str(analysis), "--videos", video_dir,
             "--sample-fps", str(args.sample_fps), *precision], env, f"bikesafe.scene3d {rides}")
    if not args.skip_infrastructure:
        model = ["--model", args.infra_model] if args.infra_model else []
        run(["bikesafe.infrastructure", *perception_dirs, "--analysis", str(analysis), "--videos", video_dir,
             "--sample-fps", str(args.sample_fps), *model], env, f"bikesafe.infrastructure {rides}")
    pending = [perception / v.stem for v in videos if tracks_outdated(analysis / v.stem)]
    run_parallel([["bikesafe.tracks", str(p), "--out", str(analysis)] for p in pending], args.jobs, env)
    run(["bikesafe.exposure", "--analysis", str(analysis), "--perception", str(perception), "--models", str(args.models),
         "--out", str(corpus)], env, f"bikesafe.exposure {rides}")
    if args.render_minutes > 0:
        for v in videos:
            meta = read_json(perception / v.stem / "meta.json")
            start = max(0.0, meta["source_frames"] / meta["source_fps"] / 2 - 30 * args.render_minutes)
            run(["bikesafe.render", v.stem, "--videos", str(v.parent), "--analysis", str(analysis), "--corpus", str(corpus),
                 "--start-s", f"{start:.0f}", "--duration-s", f"{60 * args.render_minutes:.0f}",
                 "--out", str(corpus / v.stem / "typology_overlay.mp4")], env)

    print(f"\nstage timings (total {(time.perf_counter() - started) / 60:.1f} min wall clock):")
    for name, seconds in TIMINGS:
        print(f"  {seconds / 60:7.1f} min  {name}")


if __name__ == "__main__":
    main()
