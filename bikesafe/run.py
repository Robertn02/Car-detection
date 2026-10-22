"""One command for the whole corpus: every stage, resumable, then retraining and the per-ride outputs.

    python -m bikesafe.run videos [--out-root work] [--render-minutes 1] [--skip-plates]

Stages: GPU perception -> CPU ego-motion -> depth/lane pass -> lights, signs (typed and read) and bike lanes ->
licence plates read as text -> geometry/track features and vehicle ids -> retrain the typology models on
data/typology (labels follow their vehicles across perception re-runs, bikesafe.labels) -> signs typed again where the
typing settings or the reviewed-sign probe changed -> exposure outputs in results/corpus -> optional overlay clips in
<out-root>/overlays (plates boxed and read out).

Finished work is skipped: perception, ego-motion and depth are kept once written, infrastructure and plates are redone
only where they predate the current outputs (or a pass stopped part-way, which resumes), and track features are
recomputed when written by an older pipeline version or before a file they are computed from. The plan printed at the
start says what will run for each ride. CPU-only stages run for several rides at once (--jobs; by default half the
cores, fewer when memory is short). Half precision is used on CUDA; --fp32 turns it off (worth it on GPUs without fast FP16, such as the GeForce 10 series). --hw-decode decodes on the GPU's video
engine, which helps with 4K/360 exports but was slower than CPU decoding for 1080p on a laptop GPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bikesafe.common import HW_DECODE_ENV, ROOT, VIDEOS, list_videos, read_json

TIMINGS: list[tuple[str, float]] = []
# in the ride's analysis dir, the files the track features are computed from (sign types do not enter them, so typing
# the signs again does not make the features stale)
TRACK_INPUTS = ("egomotion_raw.parquet", "scene3d.parquet", "lanes.parquet", "vehicle_context.parquet")
JOB_MEMORY = 1.5 * 2**30  # bytes of free memory per CPU-stage job


def default_jobs() -> int:
    """Rides processed at once in the CPU-only stages: half the cores, and no more than the free memory holds (a
    track-feature job peaks near 1 GB; too many at once failed with OpenBLAS allocation errors on an 8 GB laptop)."""
    cores = max(1, (os.cpu_count() or 2) // 2)
    try:
        import psutil
    except ImportError:
        return cores
    return max(1, min(cores, int(psutil.virtual_memory().available / JOB_MEMORY)))


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
    """True when track features are missing, were written before the current feature version (or before the
    lane-paint and kerb-context features they can now include), or are older than a file they are computed from (a
    re-run of the ego-motion, depth or infrastructure pass)."""
    from bikesafe.tracks import FEATURES_VERSION

    calib_path, tracks_path = analysis_dir / "calibration.json", analysis_dir / "tracks.parquet"
    if not tracks_path.exists() or not calib_path.exists():
        return True
    inputs = [analysis_dir / name for name in TRACK_INPUTS]
    if any(path.exists() and path.stat().st_mtime > tracks_path.stat().st_mtime for path in inputs):
        return True
    calib = read_json(calib_path)
    if calib.get("features_version", 1) < FEATURES_VERSION:
        return True
    if (analysis_dir / "lanes.parquet").exists() and not calib.get("lane_features", False):
        return True
    return (analysis_dir / "vehicle_context.parquet").exists() and not calib.get("context_features", False)


def columns_of(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names) if path.exists() else set()


def infrastructure_outdated(analysis_dir: Path) -> bool:
    """True when the infrastructure pass has not covered the whole ride (missing, a test window, or stopped part-way:
    it resumes from its checkpoint), or ran before signs were segmented and read and lane paint came from the
    segmenter."""
    from bikesafe.infrastructure import pass_complete

    lanes, objects = columns_of(analysis_dir / "lanes.parquet"), columns_of(analysis_dir / "infrastructure.parquet")
    return not pass_complete(analysis_dir) or "crosswalk_share" not in lanes or "sign_text" not in objects


def signs_outdated(analysis_dir: Path) -> bool:
    """True when a finished pass read the signs with other sign-typing settings or another reviewed-sign probe than
    the current ones (bikesafe.signs.typing_fingerprint): they are typed again from the saved text and CLIP features."""
    from bikesafe.infrastructure import COVERAGE, pass_complete
    from bikesafe.signs import typing_fingerprint

    if not pass_complete(analysis_dir):
        return False
    coverage = read_json(analysis_dir / COVERAGE)
    return bool(coverage.get("signs_read")) and coverage.get("sign_typing") != typing_fingerprint()


def plates_outdated(analysis_dir: Path) -> bool:
    """True when plates have not been read, were stored as hashes by the earlier identity audit, or were read before
    each plate's best view was kept (bikesafe.review shows it)."""
    return not {"plate_text", "plate_frame"} <= columns_of(analysis_dir / "plates.parquet")


def plan(videos: list[Path], perception: Path, analysis: Path, args: argparse.Namespace) -> list[tuple[str, str]]:
    """What still has to run for each ride, checked from the files each stage writes."""
    rows = []
    for v in videos:
        meta_path, out = perception / v.stem / "meta.json", analysis / v.stem
        todo = []
        if not (meta_path.exists() and read_json(meta_path).get("complete")):
            todo.append("perception")
        if not (out / "egomotion_raw.parquet").exists():
            todo.append("ego-motion")
        if not args.skip_depth and not (out / "scene3d.parquet").exists():
            todo.append("depth")
        if not args.skip_infrastructure and infrastructure_outdated(out):
            todo.append("infrastructure")
        elif not args.skip_infrastructure and signs_outdated(out):
            todo.append("sign typing")
        if not args.skip_plates and plates_outdated(out):
            todo.append("plates")
        if set(todo) - {"plates", "sign typing"} or tracks_outdated(out):  # plates and sign types do not enter them
            todo.append("track features")
        rows.append((v.stem, ", ".join(todo) or "up to date"))
    return rows


def download_models() -> None:
    """Fetch every model the stages load, so a job can run where there is no internet (e.g. CARC compute nodes)."""
    from bikesafe.infrastructure import DEFAULT_MODEL, InfrastructureDetector
    from bikesafe.perceive import ClipEncoder
    from bikesafe.plates import PlateReader
    from bikesafe.scene3d import DepthModel
    from bikesafe.signs import SignOCR, SignTyper
    from bikesafe.streetseg import StreetSegmenter

    InfrastructureDetector(DEFAULT_MODEL, "cpu")
    StreetSegmenter("cpu", fp16=False)
    SignTyper("cpu", fp16=False)
    SignOCR("cpu", fp16=False)
    DepthModel("cpu", fp16=False)
    ClipEncoder("ViT-B-32", "laion2b_s34b_b79k", "cpu", fp16=False)
    PlateReader()
    print("all models downloaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", type=Path, nargs="?", default=VIDEOS,
                        help="A ride video or a folder of them (default: videos/, or BIKESAFE_VIDEOS)")
    parser.add_argument("--out-root", type=Path, default=ROOT / "work")
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--gmc", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame rate for the depth, lane and sign passes")
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "typology")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results" / "corpus", help="Per-ride output tables")
    parser.add_argument("--no-train", action="store_true", help="Keep the current typology models instead of retraining")
    parser.add_argument("--render-minutes", type=float, default=0.0, help="Overlay clip length from each video's midpoint")
    parser.add_argument("--jobs", type=int, default=default_jobs(),
                        help="Rides processed at once in the CPU-only stages")
    parser.add_argument("--hw-decode", action="store_true", help="Decode video with the GPU's video engine when available")
    parser.add_argument("--fp32", action="store_true", help="Full precision on the GPU (FP16 is the default on CUDA)")
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip the depth pass (its lane counts are superseded by the lane-paint features)")
    parser.add_argument("--skip-infrastructure", action="store_true", help="Skip lights, signs and bike lanes")
    parser.add_argument("--infra-model", default=None, help="YOLOE-26 weights for lights and stencils")
    parser.add_argument("--seg-height", type=int, default=768,
                        help="Rows the street-scene segmenter sees (1088 finds smaller signs at ~2x the cost)")
    parser.add_argument("--download-models", action="store_true",
                        help="Only fetch the models (run once where there is internet access), then exit")
    parser.add_argument("--skip-plates", action="store_true", help="Do not read licence plates")
    args = parser.parse_args()
    if args.download_models:
        download_models()
        return

    env = dict(os.environ)
    if args.hw_decode:
        env[HW_DECODE_ENV] = "1"
    precision = ["--fp32"] if args.fp32 else []
    perception = args.out_root / "perception"
    analysis = args.out_root / "analysis"
    corpus = args.corpus
    videos = [args.videos] if args.videos.is_file() else list_videos(args.videos)
    if not videos:
        raise SystemExit(f"No videos in {args.videos}")
    video_dir = str(args.videos if args.videos.is_dir() else args.videos.parent)
    perception_dirs = [str(perception / v.stem) for v in videos]
    print("plan:")
    for stem, todo in plan(videos, perception, analysis, args):
        print(f"  {stem}: {todo}")
    started = time.perf_counter()

    rides = f"({len(videos)} rides)"
    run(["bikesafe.perceive", *map(str, videos), "--out", str(perception), "--target-fps", str(args.target_fps),
         "--gmc", args.gmc, *precision], env, f"bikesafe.perceive {rides}")
    run_parallel([["bikesafe.egomotion", p, "--analysis", str(analysis), "--videos", video_dir] for p in perception_dirs],
                 args.jobs, env)
    if not args.skip_depth:
        run(["bikesafe.scene3d", *perception_dirs, "--analysis", str(analysis), "--videos", video_dir,
             "--sample-fps", str(args.sample_fps), *precision], env, f"bikesafe.scene3d {rides}")
    redo = [str(perception / v.stem) for v in videos if infrastructure_outdated(analysis / v.stem)]
    if not args.skip_infrastructure and redo:
        model = ["--model", args.infra_model] if args.infra_model else []
        run(["bikesafe.infrastructure", *redo, "--analysis", str(analysis), "--videos", video_dir, "--overwrite",
             "--sample-fps", str(args.sample_fps), "--seg-height", str(args.seg_height), *model, *precision], env,
            f"bikesafe.infrastructure ({len(redo)} rides)")
    redo = [str(perception / v.stem) for v in videos if plates_outdated(analysis / v.stem)]
    if not args.skip_plates and redo:
        run(["bikesafe.plates", *redo, "--analysis", str(analysis), "--videos", video_dir, "--overwrite"], env,
            f"bikesafe.plates ({len(redo)} rides)")
    pending = [perception / v.stem for v in videos if tracks_outdated(analysis / v.stem)]
    run_parallel([["bikesafe.tracks", str(p), "--out", str(analysis), "--videos", video_dir] for p in pending],
                 args.jobs, env)
    if not args.no_train:
        run(["bikesafe.train", "--analysis", str(analysis), "--perception", str(perception), "--models", str(args.models)],
            env, "bikesafe.train")
    redo = [str(perception / v.stem) for v in videos if signs_outdated(analysis / v.stem)]
    if not args.skip_infrastructure and redo:  # after training: a new sign probe applies too
        run(["bikesafe.infrastructure", *redo, "--analysis", str(analysis), "--videos", video_dir, "--read-only",
             "--keep-text", *precision], env, f"sign typing ({len(redo)} rides)")
    run(["bikesafe.exposure", "--analysis", str(analysis), "--perception", str(perception), "--models", str(args.models),
         "--out", str(corpus)], env, f"bikesafe.exposure {rides}")
    if args.render_minutes > 0:
        for v in videos:
            meta = read_json(perception / v.stem / "meta.json")
            start = max(0.0, meta["source_frames"] / meta["source_fps"] / 2 - 30 * args.render_minutes)
            run(["bikesafe.render", v.stem, "--videos", str(v.parent), "--analysis", str(analysis), "--corpus", str(corpus),
                 "--start-s", f"{start:.0f}", "--duration-s", f"{60 * args.render_minutes:.0f}",
                 "--out", str(args.out_root / "overlays" / f"{v.stem}.mp4")], env)

    print(f"\nstage timings (total {(time.perf_counter() - started) / 60:.1f} min wall clock):")
    for name, seconds in TIMINGS:
        print(f"  {seconds / 60:7.1f} min  {name}")
    print(f"per-ride outputs: {corpus}")


if __name__ == "__main__":
    main()
