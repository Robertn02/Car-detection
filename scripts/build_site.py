"""Build the project's web page: the overlay demos and the review page, which play in any browser with nothing to
install. GitHub Pages serves it from the gh-pages branch, at https://robertn02.github.io/Car-detection/.

    python scripts/build_site.py                    # into work/site, a worktree of the gh-pages branch
    python -m http.server -d work/site 8000         # look at it: http://127.0.0.1:8000
    git -C work/site add -A
    git -C work/site commit -m "Update the site"
    git -C work/site push origin gh-pages

The clips (a few hundred MB) live on that branch only, so main stays quick to clone (the Colab notebook clones it).
The demos are rendered at the videos' own size and frame rate as H.264 at CRF 23, capped at 10 Mbit/s so they
stream; the review page is `bikesafe.review --export` (decisions made on it are downloaded from its Progress tab and
added with `bikesafe.review --import`). Needs the ride videos (BIKESAFE_VIDEOS) and the finished analysis
(work/analysis, results/corpus). Posters for the README are written to demos/ as well.
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bikesafe.common import RELATION_COLOURS_BGR, RELATION_TEXT, RELATIONS  # noqa: E402

REPO = "https://github.com/Robertn02/Car-detection"
COLAB = "https://colab.research.google.com/github/Robertn02/Car-detection/blob/main/scripts/colab_try_your_video.ipynb"
# file name, ride, start (s), title
DEMOS = [("overlay_006_bikelane_440s", "VID_20260224_162848_00_006", 440, "Painted bike lane"),
         ("overlay_005_following_280s", "VID_20260224_091119_00_005", 280, "Following a car from a red light"),
         ("overlay_006_parked_700s", "VID_20260224_162848_00_006", 700, "Parked rows, then a red light")]
DURATION_S = 60
CRF, MAX_MBPS = 23, 10
POSTER_WIDTH = 1280


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def worktree(out: Path) -> None:
    """Make `out` a worktree of the gh-pages branch: the published branch if there is one, else a new orphan branch
    (none of main's history or files)."""
    if (out / ".git").exists():
        return
    if git("rev-parse", "--verify", "--quiet", "refs/heads/gh-pages").returncode and \
            not git("ls-remote", "--exit-code", "--heads", "origin", "gh-pages").returncode:
        git("fetch", "origin", "gh-pages:gh-pages")
    exists = not git("rev-parse", "--verify", "--quiet", "refs/heads/gh-pages").returncode
    done = git("worktree", "add", str(out), "gh-pages") if exists else git("worktree", "add", "--orphan", "-b",
                                                                          "gh-pages", str(out))
    if done.returncode:
        raise SystemExit(done.stderr)


def poster(clip: Path, *targets: Path) -> None:
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read {clip}")
    image = cv2.resize(image, (POSTER_WIDTH, round(image.shape[0] * POSTER_WIDTH / image.shape[1])),
                       interpolation=cv2.INTER_AREA)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def render(out: Path, again: bool) -> None:
    for name, ride, start, _ in DEMOS:
        clip = out / "demos" / f"{name}.mp4"
        if clip.exists() and not again:
            continue
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "bikesafe.render", ride, "--start-s", str(start), "--duration-s",
                        str(DURATION_S), "--crf", str(CRF), "--max-mbps", str(MAX_MBPS), "--out", str(clip)],
                       cwd=ROOT, check=True)
        poster(clip, clip.with_suffix(".jpg"), ROOT / "demos" / f"{name}.jpg")


def hex_colour(bgr: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*bgr[::-1])


def index_page() -> str:
    legend = "".join(f'<span class="key"><i style="background:{hex_colour(RELATION_COLOURS_BGR[r])}"></i>'
                     f'{html.escape(RELATION_TEXT[r])}</span>' for r in RELATIONS)
    demos = "".join(f'''
<section class="demo" id="{name}">
  <h2>{html.escape(title)}</h2>
  <video controls playsinline preload="metadata" poster="demos/{name}.jpg" src="demos/{name}.mp4"></video>
  <p class="muted">Ride {ride[-1]}, from {start // 60}:{start % 60:02d} &middot; {DURATION_S} s at the camera's full size and
  frame rate &middot; <a href="demos/{name}.mp4" download>download</a></p>
</section>''' for name, ride, start, title in DEMOS)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike route vehicle typology</title>
<style>
:root {{ --bg:#f6f6f4; --panel:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#deded9; --accent:#2b6cb0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --panel:#1f1f1d; --ink:#ececea; --muted:#9a9a94;
  --line:#33332f; --accent:#63a4e8; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:16px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--ink); }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:28px; margin:0 0 6px; }} h2 {{ font-size:19px; margin:0 0 10px; }}
a {{ color:var(--accent); }} .muted {{ color:var(--muted); font-size:14px; }}
.lead {{ max-width:760px; margin:0 0 14px; }}
.keys {{ display:flex; flex-wrap:wrap; gap:6px 16px; margin:0 0 24px; font-size:14px; }}
.key i {{ display:inline-block; width:14px; height:14px; border-radius:3px; margin-right:6px; vertical-align:-2px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin:0 0 28px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.card h2 {{ font-size:17px; }} .card p {{ margin:0 0 10px; }}
.button {{ display:inline-block; padding:7px 14px; border-radius:6px; background:var(--accent); color:#fff;
  text-decoration:none; font-weight:600; }}
.demo {{ margin:0 0 32px; }}
video {{ width:100%; aspect-ratio:16/9; background:#000; border-radius:8px; display:block; }}
</style></head><body><main>
<h1>Bike route vehicle typology</h1>
<p class="lead">From a handlebar camera, every vehicle around the rider and how it relates to the rider, with the
riding context, traffic signals, signs (typed and read) and licence plates. Box colours:</p>
<div class="keys">{legend}</div>
<div class="cards">
  <div class="card"><h2>Review page</h2><p>The model's answers - vehicles, signs, plates - most likely errors first.
  Confirm or correct with one key; decisions stay in your browser until you download them.</p>
  <a class="button" href="review/">Open the review page</a></div>
  <div class="card"><h2>Your own video</h2><p>Runs in Google Colab on a free GPU: nothing to install. Results go to
  your Google Drive.</p><a class="button" href="{COLAB}">Open in Colab</a></div>
  <div class="card"><h2>Code and results</h2><p>Per-ride tables, accuracy and limitations, and the pipeline.</p>
  <a class="button" href="{REPO}">GitHub</a></div>
</div>{demos}
</main></body></html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "site")
    parser.add_argument("--per-queue", type=int, default=20, help="Items per queue on the review page")
    parser.add_argument("--render-again", action="store_true", help="Render the demos even if they are there")
    args = parser.parse_args()
    worktree(args.out)
    render(args.out, args.render_again)
    subprocess.run([sys.executable, "-m", "bikesafe.review", "--export", str(args.out / "review"), "--per-queue",
                    str(args.per_queue)], cwd=ROOT, check=True)
    (args.out / "index.html").write_text(index_page(), encoding="utf-8")
    (args.out / ".nojekyll").write_text("", encoding="utf-8")  # serve the files as they are
    print(f"site in {args.out}")


if __name__ == "__main__":
    main()
