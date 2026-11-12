"""Build the project's web page: the overlay demos and the review page, which play in any browser with nothing to
install. GitHub Pages serves it from the gh-pages branch, at https://robertn02.github.io/Car-detection/.

    python scripts/build_site.py                    # into work/site, a worktree of the gh-pages branch
    python scripts/build_site.py --serve 8000       # look at it: http://127.0.0.1:8000
    git -C work/site add -A
    git -C work/site commit -m "Update the site"
    git -C work/site push origin gh-pages

The clips (a few hundred MB) live on that branch only, so main stays quick to clone (the Colab notebook clones it).
The demos (2 minutes each) are rendered at the videos' own size and frame rate as H.264 at CRF 23, capped at
5.5 Mbit/s so they stream and stay under GitHub's 100 MB a file. The page itself is scripts/site/ (template,
stylesheet and script) filled with the tables' numbers - it reads as a short research note whose figures are the
demo player, the tables and the map; the review page is `bikesafe.review --export` (decisions made
on it are downloaded and added with `bikesafe.review --import`). Needs the ride videos (BIKESAFE_VIDEOS) and the finished analysis
(work/analysis, results/corpus). Posters for the README are written to demos/ as well.

Rides with a GPS track (bikesafe.gps, results/corpus/<video>/gps_1s.csv) get a map beside their demo, whose dot follows
the video, and make up the exposure map (map/). After running bikesafe.gps, --pages-only writes just the pages.

--serve serves the built site locally. Unlike `python -m http.server`, it answers byte-range requests, as GitHub
Pages does: without them a browser cannot jump to a moment in a clip and plays it from the start instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bikesafe.common import RELATION_COLOURS_BGR, RELATION_TEXT, RELATIONS  # noqa: E402
from bikesafe.exposure import RELIABLE_Z_M  # noqa: E402
from bikesafe.review import CAUSES  # noqa: E402

REPO = "https://github.com/Robertn02/Car-detection"
SITE_URL = "https://robertn02.github.io/Car-detection"  # where GitHub Pages serves the built site
COLAB = "https://colab.research.google.com/github/Robertn02/Car-detection/blob/main/scripts/colab_try_your_video.ipynb"
# file name, ride, start (s), title, what is in it. Windows are outside the rides' privacy zones (bikesafe.review.
# private_seconds) and were picked from the events: each shows what its text says.
DEMOS = [("overlay_006_overtakes_200s", "VID_20260224_162848_00_006", 200, "Bike lane beside traffic",
          "Afternoon traffic passes on the left while the rider keeps to a painted bike lane next to parked cars. "
          "Three cars overtake, the closest at 1.7 m."),
         ("overlay_005_closepass_712s", "VID_20260224_091119_00_005", 712, "Two close passes",
          "Two waits at red lights, then two vehicles pass within 1.5 m. Just before the first, a car pulls out "
          "across the bike lane; the software has it as parked."),
         ("overlay_001_greenlane_590s", "VID_20260209_090112_00_001", 590, "Onto a green bike lane",
          "A shared street, then a green-painted bike lane beside parked camper vans. One car passes at 1.3 m "
          "near the end."),
         ("overlay_002_night_340s", "VID_20260209_183334_00_002", 340, "After dark",
          "The same pipeline at night, working from headlights and street lighting. Three overtakes, two of them "
          "close."),
         ("overlay_005_following_260s", "VID_20260224_091119_00_005", 260, "Behind a car",
          "Riding in the lane behind other vehicles. Each is boxed in red with its distance."),
         ("overlay_006_parked_680s", "VID_20260224_162848_00_006", 680, "Red lights and parked cars",
          "Two waits at red lights on a street lined with parked cars. Cross traffic at the junctions is boxed "
          "in blue.")]
DURATION_S = 120
# two minutes at the camera's size and frame rate: capped at 5.5 Mbit/s so a clip stays under GitHub's 100 MB a file
CRF, MAX_MBPS = 23, 5.5
# Who to credit under the page's title, e.g. ["A. Rider", "B. Student"]; the line is left out while this is empty
AUTHORS: list[str] = []
# The page's worked example of what the labels miss: a demo, ride seconds of two stills from it, and what each shows
CONTEXT_CASE = {"demo": "overlay_005_closepass_712s", "from_s": 802,
                "stills": [(805.0, "A car pulls out across the bike lane. The software has it as parked, 3 m ahead."),
                           (810.0, "The rider has moved left around it, and a pickup passes at 1.2 m.")]}
CONTEXT = ROOT / "data" / "typology" / "event_context.csv"  # what led to a pass, as riders said on the review page
NIGHT_RIDES = ("002", "004")
POSTER_WIDTH = 1280
CORPUS = ROOT / "results" / "corpus"
LEAFLET = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4"
# OpenStreetMap's own tiles need no key and have street names; the dark maps are the same tiles through a CSS filter
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_CREDIT = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
# Where tile.openstreetmap.org cannot be reached (a network whose DNS does not resolve it), the maps fall back to the
# Humanitarian OSM style served by OpenStreetMap France: the same data, no key needed
TILE_FALLBACK = {"url": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
                 "credit": TILE_CREDIT + ', tiles <a href="https://www.hotosm.org/">HOT</a> hosted by '
                                         '<a href="https://openstreetmap.fr/">OSM France</a>'}
DARK_TILES = "invert(1) grayscale(1) brightness(0.75) contrast(1.1)"
HEAT_RAMP = ["#2c1a6e", "#b3206e", "#f05a28", "#fbb33b", "#fffbe6"]  # least to most, on the map and its legend
# heatmap layers: per-second counts from timeline_1s.csv (value = column or sum of columns)
HEAT = {"traffic": ("Moving traffic around the rider", ["n_ego_lane", "n_adjacent_same", "n_oncoming"]),
        "ego": ("Vehicles in the rider's lane", ["n_ego_lane"]),
        "door": ("Parked cars in the door zone", ["door_zone_parked"]),
        "parked": ("Parked cars alongside", ["n_parked"])}
EVENTS = {"close_pass_under_1p5m": "Close pass (under 1.5 m)", "oncoming_close_pass": "Oncoming close pass",
          "vehicle_close_in_my_lane": "Vehicle close in my lane", "overtaken_by_vehicle": "Overtaken"}
# riding contexts, the same colours on the map and the page's charts
SCENES = {"separated_path": ("Separated path", "#2aa39a"), "bike_lane": ("Bike lane", "#35a565"),
          "shared_road": ("Shared road", "#9aa4a9")}
CONTEXT_COLOURS = SCENES


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
    for name, ride, start, *_ in DEMOS:
        clip = out / "demos" / f"{name}.mp4"
        if clip.exists() and not again:
            continue
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "bikesafe.render", ride, "--start-s", str(start), "--duration-s",
                        str(DURATION_S), "--crf", str(CRF), "--max-mbps", str(MAX_MBPS), "--out", str(clip)],
                       cwd=ROOT, check=True)
        poster(clip, clip.with_suffix(".jpg"), ROOT / "demos" / f"{name}.jpg")


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def context_stills(out: Path) -> None:
    """The stills of CONTEXT_CASE, cut from its demo clip (nothing to do before the clip is rendered)."""
    start = next(start for name, _, start, *_ in DEMOS if name == CONTEXT_CASE["demo"])
    clip = out / "demos" / f"{CONTEXT_CASE['demo']}.mp4"
    if not clip.exists():
        return
    cap = cv2.VideoCapture(str(clip))
    for i, (t, _) in enumerate(CONTEXT_CASE["stills"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t - start) * cap.get(cv2.CAP_PROP_FPS))))
        ok, image = cap.read()
        if ok:
            image = cv2.resize(image, (POSTER_WIDTH, round(image.shape[0] * POSTER_WIDTH / image.shape[1])),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "demos" / f"context_{i + 1}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()


def hex_colour(bgr: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*bgr[::-1])


def ride_gps(corpus: Path = CORPUS) -> dict[str, pd.DataFrame]:
    """timeline_1s.csv with the position in each second (bikesafe.gps; blank in the privacy zones), for the rides
    filmed with GPS."""
    rides = {}
    for path in sorted(corpus.glob("*/gps_1s.csv")):
        timeline = pd.read_csv(path.parent / "timeline_1s.csv")
        rides[path.parent.name] = timeline.merge(pd.read_csv(path)[["second", "lat", "lon"]], on="second", how="left")
    return rides


def polylines(t: pd.DataFrame, every: int = 2) -> list[list[list[float]]]:
    """The positions as lines, every `every` seconds, broken where there is no position."""
    pieces, current = [], []
    for lat, lon in t[["lat", "lon"]].to_numpy(float)[::every]:
        if np.isfinite(lat):
            current.append([round(float(lat), 5), round(float(lon), 5)])
        elif current:
            pieces, current = pieces + [current], []
    return [piece for piece in pieces + [current] if len(piece) > 1]


def demo_gps(rides: dict[str, pd.DataFrame]) -> dict:
    """Per demo filmed with GPS: its ride's route, and the position in each second of the clip (null where unknown)."""
    out = {}
    for name, ride, start, *_ in DEMOS:
        if ride not in rides:
            continue
        t = rides[ride].set_index("second")
        clip = [[round(float(t.at[s, "lat"]), 6), round(float(t.at[s, "lon"]), 6)]
                if s in t.index and np.isfinite(t.at[s, "lat"]) else None for s in range(start, start + DURATION_S + 1)]
        if any(clip):
            out[name] = {"route": polylines(rides[ride]), "clip": clip}
    return out


def exposure_data(rides: dict[str, pd.DataFrame]) -> dict:
    """What the exposure map draws: each second outside the privacy zones as [lat, lon, ride, riding context, heat
    values...], the events that have a position, and each ride's route."""
    points, events, routes = [], [], []
    for r, (ride, t) in enumerate(rides.items()):
        placed = t[t.lat.notna()]
        scene = placed.scene.map({k: i for i, k in enumerate(SCENES)}).fillna(-1).astype(int).to_numpy()
        heat = np.column_stack([placed.reindex(columns=cols).fillna(0).sum(axis=1).to_numpy(float)
                                for _, cols in HEAT.values()])
        for (lat, lon), sc, values in zip(placed[["lat", "lon"]].to_numpy(float), scene, heat):
            points.append([round(float(lat), 5), round(float(lon), 5), r, int(sc),
                           *[round(float(v), 2) for v in values]])
        start = pd.Timestamp(t.timestamp_local.iloc[0])
        routes.append({"label": f"Ride {ride[-1]} - {start:%a %b %d, %H:%M}", "lines": polylines(t)})
        path = CORPUS / ride / "events.csv"
        if not path.exists():
            continue
        where = t.set_index("second")
        for e in pd.read_csv(path).to_dict("records"):
            second = int(e["t_start"])
            if e["event"] in EVENTS and second in where.index and np.isfinite(where.at[second, "lat"]):
                clearance = e.get("clearance_m", np.nan)
                events.append([round(float(where.at[second, "lat"]), 5), round(float(where.at[second, "lon"]), 5),
                               e["event"], r, str(where.at[second, "timestamp_local"])[11:19],
                               round(float(clearance), 1) if pd.notna(clearance) else None])
    return {"points": points, "events": events, "routes": routes,
            "heat": [{"key": k, "label": label} for k, (label, _) in HEAT.items()],
            "event_names": EVENTS, "scenes": [[label, colour] for label, colour in SCENES.values()]}


KINDS = {"close_pass": ("Close pass, under 1.5 m", "#c8322f"), "overtake": ("Overtaken", "#ff9600"),
         "in_lane": ("Vehicle close ahead in the lane", "#eb2828"), "oncoming": ("Oncoming vehicle passes close", "#c83cc8"),
         "red_wait": ("Waiting at a red light", "#8f1f1c"), "stop_sign": ("Stop sign", "#5a646b")}
RELATION_MEANING = {"ego_lane": "Ahead of the rider, in the lane the rider is using.",
                    "adjacent_same": "Going the rider's way in another lane. Overtaking vehicles are here.",
                    "oncoming": "Coming towards the rider on the same street.",
                    "parked": "Stopped at the curb, in a parking lane or in a lot. Waiting in traffic is not parked.",
                    "cross_side": "On a cross street or a driveway, or beyond a median."}
SITE_FILES = ROOT / "scripts" / "site"  # the page's template, stylesheet and script


def beside_parked(timeline: pd.DataFrame, t: float) -> bool:
    """Whether the rider was within door range of a parked car as a pass began (the second before to two after)."""
    if "door_zone_parked" not in timeline:
        return False
    around = timeline[timeline.second.between(int(t) - 1, int(t) + 2)]
    return bool((around.door_zone_parked.fillna(0) > 0).any())


def demo_moments(ride: str, start: int, corpus: Path = CORPUS, duration: int = DURATION_S,
                 context: Path = CONTEXT) -> list[dict]:
    """What happens in a demo clip, from the ride's events.csv, in time order: seconds into the clip, the kind
    (KINDS), in words, and how long for a wait. A close pass is said once, as the overtake it is, with what the
    tables know of its setting (parked cars within door range on the other side) and, as `why`, what a rider said
    led to it on the review page (`context`) - which the pipeline does not see."""
    path = corpus / ride / "events.csv"
    if not path.exists():
        return []
    ev = pd.read_csv(path)
    ev = ev[(ev.t_start >= start) & (ev.t_start < start + duration)].sort_values("t_start")
    close = set(ev.loc[ev.event == "close_pass_under_1p5m", "vehicle_id"].dropna())
    timeline = corpus / ride / "timeline_1s.csv"
    timeline = pd.read_csv(timeline) if timeline.exists() else pd.DataFrame(columns=["second"])
    said = pd.read_csv(context) if context.exists() else pd.DataFrame(columns=["video", "track_id", "cause", "note"])
    said = said[(said.video == ride) & ~said.cause.isin(["none", "unclear"])].set_index("track_id")
    words = {key: text for key, text, _ in CAUSES}
    out = []
    for e in ev.itertuples():
        at = round(float(e.t_start) - start, 1)
        if e.event == "overtaken_by_vehicle":
            kind = "close_pass" if e.vehicle_id in close else "overtake"
            text = f"{'Close pass' if kind == 'close_pass' else 'Overtaken'}, {float(e.clearance_m):.1f} m on the {e.side}"
            moment = {"t": at, "kind": kind, "text": text + (", beside parked cars" if beside_parked(timeline, e.t_start) else "")}
            if e.track_id in said.index:
                row = said.loc[e.track_id]
                note = row.note if isinstance(row.note, str) and row.note.strip() else ""
                moment["why"] = note or f"The rider says: {words.get(row.cause, row.cause)}."
            out.append(moment)
        elif e.event == "red_light_wait":
            out.append({"t": at, "kind": "red_wait", "text": f"Red light, {float(e.duration_s):.0f} s",
                        "span": round(min(float(e.duration_s), start + duration - float(e.t_start)), 1)})
        elif e.event == "vehicle_close_in_my_lane":
            out.append({"t": at, "kind": "in_lane", "text": f"Vehicle {float(e.z_min_m):.0f} m ahead in the lane"})
        elif e.event == "oncoming_close_pass":
            out.append({"t": at, "kind": "oncoming", "text": "Oncoming vehicle passes close"})
        elif e.event == "stop_sign":
            out.append({"t": at, "kind": "stop_sign", "text": "Stop sign"})
    return out


def demo_summary(ride: str, start: int, moments: list[dict], corpus: Path = CORPUS, duration: int = DURATION_S) -> str:
    """One line for a demo's thumbnail: the ride, night, the riding context it is mostly in, and what happens."""
    line = pd.read_csv(corpus / ride / "timeline_1s.csv")
    window = line[(line.second >= start) & (line.second < start + duration)]
    parts = [f"Ride {int(ride[-3:])}"]
    if ride.endswith(NIGHT_RIDES):
        parts.append("night")
    shares = window.scene.value_counts(normalize=True)
    parts += [label.lower() for scene, (label, _) in SCENES.items() if shares.get(scene, 0) >= 0.3]
    kinds = [m["kind"] for m in moments]
    plural = lambda n, word, words: f"{n} {word if n == 1 else words}"  # noqa: E731
    overtakes, close = kinds.count("overtake") + kinds.count("close_pass"), kinds.count("close_pass")
    if overtakes:
        parts.append(plural(overtakes, "overtake", "overtakes") + (f" ({close} close)" if close else ""))
    if kinds.count("red_wait"):
        parts.append(plural(kinds.count("red_wait"), "red light", "red lights"))
    return ", ".join(parts)


def ride_rows(corpus: Path = CORPUS) -> list[dict]:
    """The rides as the page's first table: when, minutes, vehicles counted, overtakes, those under 1.5 m, waits at
    red lights and the share of the time beside parked cars; the last row adds them up."""
    s = pd.read_csv(corpus / "corpus_summary.csv")
    rows = []
    for r in s.itertuples():
        when = pd.to_datetime(r.video[4:19], format="%Y%m%d_%H%M%S", errors="coerce")
        part = "" if pd.isna(when) else ("evening, dark" if r.video.endswith(NIGHT_RIDES) else
                                         "morning" if when.hour < 12 else "afternoon")
        rows.append({"ride": f"Ride {int(r.video[-3:])}", "when": "" if pd.isna(when) else f"{when.day} {when:%b}, {part}",
                     "minutes": round(float(r.minutes)), "vehicles": int(r.vehicles_counted),
                     "overtakes": int(r.events_overtaken_by_vehicle), "close": int(r.events_close_pass_under_1p5m),
                     "red_waits": int(r.events_red_light_wait), "door": float(r.door_zone_seconds) / 60 / float(r.minutes)})
    total = {"ride": "All rides", "when": "", "minutes": round(float(s.minutes.sum())),
             "vehicles": int(s.vehicles_counted.sum()), "overtakes": int(s.events_overtaken_by_vehicle.sum()),
             "close": int(s.events_close_pass_under_1p5m.sum()), "red_waits": int(s.events_red_light_wait.sum()),
             "door": float(s.door_zone_seconds.sum()) / 60 / float(s.minutes.sum())}
    return rows + [total]


def accuracy(typology: Path = ROOT / "results" / "typology") -> dict:
    """How often the deployed model was right on rides it was not trained on, among labelled vehicles within the
    range relations are checked in: overall, by closest approach and per relation ({} without an evaluation)."""
    predictions = typology / "relation_lovo_predictions.csv"
    if not predictions.exists():
        return {}
    summary = json.loads((typology / "evaluation_summary.json").read_text())
    p = pd.read_csv(predictions)
    limit = summary.get("near_z_m", RELIABLE_Z_M)
    near = p[p.z_min <= limit]
    right = near[f"pred_{summary['selected_model']}"] == near.label
    edges = [e for e in (0, 8, 16) if e < limit] + [limit]
    by_distance = [{"range": f"{a:.0f} to {b:.0f} m", "n": int(((near.z_min > a) & (near.z_min <= b)).sum()),
                    "right": float(right[(near.z_min > a) & (near.z_min <= b)].mean())} for a, b in zip(edges, edges[1:])]
    by_relation = {r: {"n": int((near.label == r).sum()), "right": float(right[near.label == r].mean())}
                   for r in RELATIONS if (near.label == r).any()}
    return {"right": float(right.mean()), "n": int(len(near)), "limit_m": float(limit), "trained_on": int(len(p)),
            "by_distance": by_distance, "by_relation": by_relation}


def facts(corpus: Path = CORPUS) -> dict[str, str]:
    """The numbers the page's sentences quote ({{name}} in scripts/site/index.html), from the tables."""
    rides = ride_rows(corpus)
    total, s = rides[-1], pd.read_csv(corpus / "corpus_summary.csv")
    count = len(rides) - 1
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    out = {"rides": words[count] if count < len(words) else str(count), "minutes": str(total["minutes"]),
           "vehicles": f"{total['vehicles']:,}",
           "parked_share": f"{s.vehicles_parked.sum() / s.vehicles_counted.sum():.0%}",
           "overtakes": str(total["overtakes"]), "close": str(total["close"]), "red_waits": str(total["red_waits"]),
           "door_share": f"{total['door']:.0%}", "range_m": f"{RELIABLE_Z_M:.0f}",
           "updated": f"{pd.Timestamp.now():%d %B %Y}".lstrip("0"),
           "byline": (" and ".join([", ".join(AUTHORS[:-1]), AUTHORS[-1]] if len(AUTHORS) > 1 else AUTHORS) + ". ")
           if AUTHORS else ""}
    a = accuracy()
    if a:
        best = max(a["by_relation"], key=lambda r: a["by_relation"][r]["right"])
        worst = min(a["by_relation"], key=lambda r: a["by_relation"][r]["right"])
        out |= {"acc": f"{a['right']:.0%}", "acc_n": str(a["n"]), "trained_on": str(a["trained_on"]),
                "acc_best": f"{a['by_relation'][best]['right']:.0%}", "acc_best_name": RELATION_TEXT[best],
                "acc_worst": f"{a['by_relation'][worst]['right']:.0%}", "acc_worst_name": RELATION_TEXT[worst],
                "acc_worst_n": str(a["by_relation"][worst]["n"])}
    squeezed = 0  # close passes that began with parked cars within door range on the rider's other side
    for events in sorted(corpus.glob("*/events.csv")):
        timeline = events.with_name("timeline_1s.csv")
        if timeline.exists():
            line, ev = pd.read_csv(timeline), pd.read_csv(events)
            squeezed += sum(beside_parked(line, t) for t in ev.t_start[ev.event == "close_pass_under_1p5m"])
    out["squeezed"] = str(squeezed)
    calibration = corpus / "metric_calibration.json"
    checks = ([abs(e["now"]["gps_over_parked_closing"] - 1) for e in json.loads(calibration.read_text())["rides"].values()
               if e.get("now")] if calibration.exists() else [])
    out["gps_within"] = f"{max(checks):.0%}" if checks else "a few percent"
    labels = ROOT / "data" / "typology" / "track_labels.csv"
    if labels.exists():  # checked by a named person: not the AI-assisted passes, nor the review page's default name
        who = pd.read_csv(labels).labeler.astype(str)
        out["human_labels"] = str(int((~who.str.startswith("claude") & (who != "reviewer")).sum()))
    return out


def page_data(gps: dict, corpus: Path = CORPUS) -> dict:
    """Everything the project page draws (scripts/site/site.js reads it as SITE)."""
    demos = []
    for name, ride, start, title, look in DEMOS:
        moments = demo_moments(ride, start, corpus)
        demos.append({"name": name, "title": title, "look": look, "ride": int(ride[-3:]), "start": start,
                      "duration": DURATION_S, "video": f"demos/{name}.mp4", "poster": f"demos/{name}.jpg",
                      "summary": demo_summary(ride, start, moments, corpus), "moments": moments, "gps": gps.get(name)})
    by_scene = pd.read_csv(ROOT / "results" / "typology" / "exposure_by_scene_overall.csv").set_index("scene")
    exposure = [{"scene": s, "label": label, "colour": colour, "minutes": float(by_scene.at[s, "minutes"]),
                 "per_min": {r: float(by_scene.at[s, r]) for r in RELATIONS}}
                for s, (label, colour) in CONTEXT_COLOURS.items() if s in by_scene.index]
    case_start = next(start for name, _, start, *_ in DEMOS if name == CONTEXT_CASE["demo"])
    context = {"demo": CONTEXT_CASE["demo"], "at": CONTEXT_CASE["from_s"] - case_start,
               "stills": [{"src": f"demos/context_{i + 1}.jpg", "time": clock(t), "text": text}
                          for i, (t, text) in enumerate(CONTEXT_CASE["stills"])]}
    return {"relations": [{"key": r, "label": RELATION_TEXT[r], "meaning": RELATION_MEANING[r],
                           "colour": hex_colour(RELATION_COLOURS_BGR[r])} for r in RELATIONS],
            "kinds": {k: {"label": label, "colour": colour} for k, (label, colour) in KINDS.items()},
            "demos": demos, "rides": ride_rows(corpus), "exposure": exposure, "accuracy": accuracy(),
            "context": context, "tiles": {"url": TILE_URL, "credit": TILE_CREDIT, "fallback": TILE_FALLBACK}}


def index_page(gps: dict) -> str:
    """The project page: scripts/site/index.html with the page's data and the numbers its sentences quote."""
    template = (SITE_FILES / "index.html").read_text(encoding="utf-8")
    for name, value in facts().items():
        template = template.replace("{{" + name + "}}", value)
    for name in ("site.css", "site.js"):  # a new version of either is fetched, not taken from a browser's cache
        stamp = hashlib.md5((SITE_FILES / name).read_bytes()).hexdigest()[:8]
        template = template.replace(f"assets/{name}", f"assets/{name}?v={stamp}")
    data = json.dumps(page_data(gps), separators=(",", ":")).replace("</", "<\\/")
    return (template.replace("__REPO__", REPO).replace("__COLAB__", COLAB).replace("__LEAFLET__", LEAFLET)
            .replace("__SITE__", SITE_URL).replace("<!--DATA-->", f"<script>const SITE = {data};</script>"))


# The exposure map: heat of per-second counts (or the riding context) as coloured lines over all rides, and the events.
# The heat is drawn like Strava's: thin lines, no blur. Each second's stretch of road is coloured by the counts summed
# over every second of the rides shown within about a cell of it, so overlapping rides and waits come out brighter, and
# the colours are the same at every zoom.
EXPOSURE_MAP_JS = """
const $ = s => document.querySelector(s);
// the map's tiles; where tile.openstreetmap.org cannot be reached (some networks do not resolve it, and take seconds
// to say so) - no tile in TILE_WAIT_MS, or more failing than loading (a few may come from the browser's cache) - the
// same map data from a second server
const TILE_WAIT_MS = 3000;
function baseTiles(m, tiles) {
  const layer = L.tileLayer(tiles.url, {attribution: tiles.credit, maxZoom: 19}).addTo(m);
  let loads = 0, errors = 0, swapped = false;
  const swap = () => {
    if (swapped || !tiles.fallback) return;
    swapped = true;
    m.attributionControl.removeAttribution(tiles.credit).addAttribution(tiles.fallback.credit);
    layer.options.attribution = tiles.fallback.credit;
    layer.setUrl(tiles.fallback.url);
  };
  layer.on("tileload", () => { loads++; });
  layer.on("tileerror", () => { if (++errors > loads) swap(); });
  setTimeout(() => { if (!loads) swap(); }, TILE_WAIT_MS);
  return layer;
}
const map = L.map("map", {preferCanvas: true});
baseTiles(map, TILES);
const EVENT_COLOURS = {close_pass_under_1p5m: "#ff4d4f", oncoming_close_pass: "#d36bff",
  vehicle_close_in_my_lane: "#ffd60a", overtaken_by_vehicle: "#f2f2f2"};  // apart from the riding-context colours
const CELL_M = 20, LEVELS = 32;
const ramp = f => {  // the colour a fraction f of the way along HEAT_RAMP
  const x = f * (HEAT_RAMP.length - 1), i = Math.min(HEAT_RAMP.length - 2, Math.floor(x)), t = x - i;
  const rgb = h => [1, 3, 5].map(o => parseInt(h.slice(o, o + 2), 16)), a = rgb(HEAT_RAMP[i]), b = rgb(HEAT_RAMP[i + 1]);
  return `rgb(${a.map((v, c) => Math.round(v + (b[c] - v) * t)).join(",")})`;
};
const joined = (p, last) => last && p[2] === last[2] && map.distance([p[0], p[1]], [last[0], last[1]]) <= 60;
fetch("exposure.json").then(r => r.json()).then(data => {
  const layers = [...data.heat.map(h => [h.key, h.label]), ["context", "Riding context"]];
  $("#layers").innerHTML = layers.map(([k, label], i) =>
    `<label><input type="radio" name="layer" value="${k}" ${i ? "" : "checked"}> ${label}</label>`).join("");
  data.routes.forEach((r, i) => $("#ride").insertAdjacentHTML("beforeend", `<option value="${i}">${r.label}</option>`));
  L.layerGroup(data.routes.flatMap(r => r.lines.map(line => L.polyline(line, {color: "#5a5a55", weight: 2, opacity: 0.6})))).addTo(map);
  const ridePoints = () => { const ride = +$("#ride").value; return data.points.filter(p => ride < 0 || p[2] === ride); };
  map.fitBounds(L.latLngBounds(data.points.map(p => [p[0], p[1]])).pad(0.05));
  let shown = L.layerGroup().addTo(map);
  const events = L.layerGroup().addTo(map);
  const draw = () => {
    const layer = document.querySelector("input[name=layer]:checked").value, ride = +$("#ride").value, pts = ridePoints();
    shown.remove();
    events.clearLayers();
    if (layer === "context") {  // runs of seconds in one riding context, as coloured lines
      shown = L.layerGroup();
      let run = [], last = null;
      const flush = () => { if (run.length > 1) L.polyline(run, {color: data.scenes[last[3]][1], weight: 4, opacity: 0.9}).addTo(shown); run = []; };
      for (const p of pts) {
        const gap = last && !joined(p, last);
        if (last && (gap || p[3] !== last[3])) { const prev = last; flush(); if (!gap) run.push([prev[0], prev[1]]); }
        if (p[3] >= 0) run.push([p[0], p[1]]);
        last = p;
      }
      flush();
      $("#legend").innerHTML = data.scenes.map(([label, colour]) =>
        `<div><span class="swatch" style="background:${colour}"></span>${label}</div>`).join("");
    } else {
      const k = 4 + data.heat.findIndex(h => h.key === layer);
      const dLat = CELL_M / 111320, dLon = dLat / Math.cos(pts[0][0] * Math.PI / 180), sums = new Map();
      const cell = p => [Math.round(p[0] / dLat), Math.round(p[1] / dLon)];
      for (const p of pts) { const key = cell(p).join(); sums.set(key, (sums.get(key) || 0) + p[k]); }
      const near = pts.map(p => {  // the counts in this cell and the 8 round it
        const [i, j] = cell(p);
        let sum = 0;
        for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) sum += sums.get(`${i + a},${j + b}`) || 0;
        return sum;
      });
      const lit = near.filter(v => v > 0).sort((a, b) => a - b), top = lit.length ? lit[Math.floor(0.98 * (lit.length - 1))] : 1;
      shown = L.layerGroup();
      let run = [], level = -1;
      const flush = () => {  // stretches with none are left to the grey route
        if (run.length > 1 && level >= 0) L.polyline(run, {color: ramp(level / (LEVELS - 1)), weight: 3, opacity: 1}).addTo(shown);
      };
      pts.forEach((p, n) => {  // runs of seconds at one level; a second's stretch starts at the second before
        const l = near[n] > 0 ? Math.min(LEVELS - 1, Math.floor(Math.sqrt(near[n] / top) * LEVELS)) : -1;
        const linked = joined(p, pts[n - 1]);
        if (!linked || l !== level) { flush(); run = linked ? [[pts[n - 1][0], pts[n - 1][1]]] : []; level = l; }
        run.push([p[0], p[1]]);
      });
      flush();
      const seconds = pts.filter(p => p[k] > 0).length;
      $("#legend").innerHTML = `<div class="ramp"></div><p class="muted">Brighter: more vehicles, for longer, near that
        spot - a wait at a red light adds up. ${seconds} of ${pts.length} seconds had any.</p>`;
    }
    shown.addTo(map);
    if ($("#events").checked) {
      for (const e of data.events.filter(e => ride < 0 || e[3] === ride)) {
        L.circleMarker([e[0], e[1]], {radius: 5, color: "#111", weight: 1, fillColor: EVENT_COLOURS[e[2]] || "#fff", fillOpacity: 0.95})
          .bindPopup(`<b>${data.event_names[e[2]]}</b>${e[5] !== null ? ` - ${e[5]} m` : ""}<br>${data.routes[e[3]].label}, ${e[4]}`)
          .addTo(events);
      }
      $("#legend").insertAdjacentHTML("beforeend", Object.entries(data.event_names).map(([k, label]) =>
        `<div class="muted"><span class="swatch dot" style="background:${EVENT_COLOURS[k]}"></span>${label}</div>`).join(""));
    }
  };
  document.querySelectorAll("input[name=layer]").forEach(r => r.onchange = draw);
  $("#events").onchange = draw;
  $("#ride").onchange = () => { draw(); const pts = ridePoints(); if (pts.length) map.fitBounds(L.latLngBounds(pts.map(p => [p[0], p[1]])).pad(0.05)); };
  draw();
});"""


def map_page() -> str:
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exposure map</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,400..700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{LEAFLET}/leaflet.css">
<style>
:root {{ --panel:rgba(17,19,21,0.93); --ink:#e7eaec; --muted:#98a2a8; --line:#2f353a; --accent:#6cc698; }}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; height:100%; background:#0e0e0d; color:var(--ink); font:15px/1.45 "Archivo", "Helvetica Neue", Arial, sans-serif; }}
.home {{ display:inline-block; color:var(--muted); font-size:.85rem; font-weight:500; text-decoration:none; margin-bottom:12px; }}
.home:hover {{ color:var(--ink); text-decoration:underline; }}
#map {{ position:absolute; inset:0; background:#0e0e0d; }}
#map .leaflet-tile-pane {{ filter:{DARK_TILES}; }}
.panel {{ position:absolute; z-index:1000; top:12px; left:12px; width:310px; max-width:calc(100% - 24px);
  max-height:calc(100% - 24px); overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:4px;
  padding:16px 18px; }}
.panel h1 {{ font:650 22px/1.1 "Archivo", sans-serif; font-stretch:90%; margin:0 0 6px; }} .panel p {{ margin:6px 0; }}
.muted {{ color:var(--muted); font-size:13px; }} a {{ color:var(--accent); }}
.panel label {{ display:block; padding:3px 0; cursor:pointer; }}
select {{ font:inherit; color:inherit; background:#1f1f1d; border:1px solid var(--line); border-radius:6px; padding:4px 8px;
  width:100%; }}
.swatch {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.swatch.dot {{ border-radius:50%; }}
.ramp {{ height:10px; border-radius:5px; margin:6px 0 2px;
  background:linear-gradient(90deg, {", ".join(HEAT_RAMP)}); }}
@media (max-width:600px) {{ .panel {{ top:auto; bottom:12px; max-height:45%; }} }}
</style></head><body>
<div id="map"></div>
<div class="panel">
  <a class="home" href="../">Bike route vehicle typology</a>
  <h1>Exposure map</h1>
  <p class="muted">Every second of the five rides with GPS, placed on the street and colored by what the camera saw
  there. The first and last 400 to 800 m of each ride are left off.</p>
  <div id="layers"></div>
  <p><select id="ride" aria-label="Ride"><option value="-1">All rides</option></select></p>
  <label><input type="checkbox" id="events" checked> Close passes and overtakes</label>
  <div id="legend"></div>
  <p class="muted"><a href="../#demos">Back to the clips</a></p>
</div>
<script src="{LEAFLET}/leaflet.js"></script>
<script>
const TILES = {json.dumps({"url": TILE_URL, "credit": TILE_CREDIT, "fallback": TILE_FALLBACK})};
const HEAT_RAMP = {json.dumps(HEAT_RAMP)};{EXPOSURE_MAP_JS}
</script></body></html>
'''


def write_pages(out: Path) -> None:
    rides = ride_gps()
    (out / "assets").mkdir(exist_ok=True)
    for name in ("site.css", "site.js"):
        shutil.copyfile(SITE_FILES / name, out / "assets" / name)
    (out / "index.html").write_text(index_page(demo_gps(rides)), encoding="utf-8")
    context_stills(out)
    if rides:
        (out / "map").mkdir(exist_ok=True)
        (out / "map" / "exposure.json").write_text(json.dumps(exposure_data(rides), separators=(",", ":")),
                                                    encoding="utf-8")
        (out / "map" / "index.html").write_text(map_page(), encoding="utf-8")
    print(f"pages written: {len(rides)} rides with GPS")


class RangeHandler(SimpleHTTPRequestHandler):
    """A static file server that answers `Range: bytes=a-b` with 206 and that part of the file, so video seeks."""

    def send_head(self):  # type: ignore[override]
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        path = Path(self.translate_path(self.path))
        if not match or not path.is_file() or not any(match.groups()):
            return super().send_head()
        size = path.stat().st_size
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(int(last), size - 1) if first and last else size - 1
        if start >= size or start > end:
            self.send_error(416, "Requested range not satisfiable")
            return None
        f = path.open("rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile) -> None:  # type: ignore[override]
        left = getattr(self, "range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(left, 1 << 16))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):  # the browser cancelled the request: it seeked on
                break
            left -= len(chunk)
        self.range_left = None

    def end_headers(self) -> None:
        if getattr(self, "range_left", None) is None:
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def serve(directory: Path, port: int) -> None:
    """Serve the built site on http://127.0.0.1:<port>, answering byte ranges like GitHub Pages."""
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(RangeHandler, directory=str(directory)))
    print(f"serving {directory} on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "site")
    parser.add_argument("--per-queue", type=int, default=20, help="Items per queue on the review page")
    parser.add_argument("--render-again", action="store_true", help="Render the demos even if they are there")
    parser.add_argument("--pages-only", action="store_true",
                        help="Only write the pages and the map data (e.g. after bikesafe.gps); keep clips and review")
    parser.add_argument("--serve", type=int, metavar="PORT", help="Serve the built site locally instead of building it")
    args = parser.parse_args()
    if args.serve:
        serve(args.out, args.serve)
        return
    worktree(args.out)
    if not args.pages_only:
        render(args.out, args.render_again)
        subprocess.run([sys.executable, "-m", "bikesafe.review", "--export", str(args.out / "review"), "--per-queue",
                        str(args.per_queue)], cwd=ROOT, check=True)
    write_pages(args.out)
    (args.out / ".nojekyll").write_text("", encoding="utf-8")  # serve the files as they are
    print(f"site in {args.out}")


if __name__ == "__main__":
    main()
