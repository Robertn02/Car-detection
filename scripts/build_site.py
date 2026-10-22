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

Rides with a GPS track (bikesafe.gps, results/corpus/<video>/gps_1s.csv) get a map beside their demo, whose dot follows
the video, and make up the exposure map (map/). After running bikesafe.gps, --pages-only writes just the pages.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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
CORPUS = ROOT / "results" / "corpus"
LEAFLET = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4"
# OpenStreetMap's own tiles need no key and have street names; the dark maps are the same tiles through a CSS filter
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_CREDIT = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
DARK_TILES = "invert(1) grayscale(1) brightness(0.75) contrast(1.1)"
HEAT_RAMP = ["#2c1a6e", "#b3206e", "#f05a28", "#fbb33b", "#fffbe6"]  # least to most, on the map and its legend
# heatmap layers: per-second counts from timeline_1s.csv (value = column or sum of columns)
HEAT = {"traffic": ("Moving traffic around the rider", ["n_ego_lane", "n_adjacent_same", "n_oncoming"]),
        "ego": ("Vehicles in the rider's lane", ["n_ego_lane"]),
        "door": ("Parked cars in the door zone", ["door_zone_parked"]),
        "parked": ("Parked cars alongside", ["n_parked"])}
EVENTS = {"close_pass_under_1p5m": "Close pass (under 1.5 m)", "oncoming_close_pass": "Oncoming close pass",
          "vehicle_close_in_my_lane": "Vehicle close in my lane", "overtaken_by_vehicle": "Overtaken"}
SCENES = {"separated_path": ("Separated path", "#3ecf8e"), "bike_lane": ("Bike lane", "#4aa3ff"),
          "shared_road": ("Shared road", "#ff7a45")}


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
    for name, ride, start, _ in DEMOS:
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


STYLE = """:root { --bg:#f6f6f4; --panel:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#deded9; --accent:#2b6cb0; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --panel:#1f1f1d; --ink:#ececea; --muted:#9a9a94;
  --line:#33332f; --accent:#63a4e8; } }
* { box-sizing:border-box; }
body { margin:0; font:16px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
a { color:var(--accent); } .muted { color:var(--muted); font-size:14px; }
.button { display:inline-block; padding:7px 14px; border-radius:6px; background:var(--accent); color:#fff;
  text-decoration:none; font-weight:600; }"""

# The demo maps: a dot that follows the video along the clip's stretch of the route (blue), the rest of the ride grey.
DEMO_MAP_JS = """
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const accent = dark ? "#63a4e8" : "#2b6cb0";
for (const [name, g] of Object.entries(GPS)) {
  const video = document.querySelector(`#${name} video`);
  const map = L.map(`map-${name}`, {scrollWheelZoom: false});
  L.tileLayer(TILE_URL, {attribution: TILE_CREDIT, maxZoom: 19}).addTo(map);
  for (const line of g.route) L.polyline(line, {color: dark ? "#8a8a84" : "#9a9a94", weight: 3, opacity: 0.7}).addTo(map);
  const clip = L.polyline(g.clip.filter(Boolean), {color: accent, weight: 6}).addTo(map);
  map.fitBounds(clip.getBounds().pad(0.8));
  const dot = L.circleMarker(g.clip.find(Boolean), {radius: 8, color: "#fff", weight: 2, fillColor: accent, fillOpacity: 1}).addTo(map);
  const at = t => {  // t seconds into the clip: between the positions of the whole seconds around it
    const i = Math.max(0, Math.min(g.clip.length - 1, Math.floor(t))), a = g.clip[i], b = g.clip[i + 1] || a, f = t - i;
    return a && [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  };
  const follow = () => { const p = at(video.currentTime); if (p) { dot.setLatLng(p); if (!map.getBounds().pad(-0.15).contains(p)) map.panTo(p); } };
  let frame = 0;
  const loop = () => { follow(); frame = requestAnimationFrame(loop); };
  video.addEventListener("play", () => { cancelAnimationFrame(frame); loop(); });
  video.addEventListener("pause", () => cancelAnimationFrame(frame));
  video.addEventListener("seeked", follow);
  clip.on("click", e => {  // jump the video to the nearest second of the clip
    let best = 0, d = Infinity;
    g.clip.forEach((p, i) => { if (p) { const dd = (p[0] - e.latlng.lat) ** 2 + (p[1] - e.latlng.lng) ** 2; if (dd < d) { d = dd; best = i; } } });
    video.currentTime = best; follow();
  });
}"""


def index_page(gps: dict) -> str:
    legend = "".join(f'<span class="key"><i style="background:{hex_colour(RELATION_COLOURS_BGR[r])}"></i>'
                     f'{html.escape(RELATION_TEXT[r])}</span>' for r in RELATIONS)

    def demo(name: str, ride: str, start: int, title: str) -> str:
        where = (f'<div class="map" id="map-{name}"></div>' if name in gps else
                 '<div class="map nomap"><p class="muted">No GPS was recorded on this ride.</p></div>')
        hint = " &middot; the dot follows the video; click the blue line to jump there" if name in gps else ""
        return f'''
<section class="demo" id="{name}">
  <h2>{html.escape(title)}</h2>
  <div class="pair"><video controls playsinline preload="metadata" poster="demos/{name}.jpg" src="demos/{name}.mp4"></video>
  {where}</div>
  <p class="muted">Ride {ride[-1]}, from {start // 60}:{start % 60:02d} &middot; {DURATION_S} s at the camera's full size and
  frame rate{hint} &middot; <a href="demos/{name}.mp4" download>download</a></p>
</section>'''

    maps = "" if not gps else (
        f'<link rel="stylesheet" href="{LEAFLET}/leaflet.css"><script src="{LEAFLET}/leaflet.js"></script>\n'
        f"<script>\nconst GPS = {json.dumps(gps, separators=(',', ':'))};\nconst TILE_URL = {json.dumps(TILE_URL)};\n"
        f"const TILE_CREDIT = {json.dumps(TILE_CREDIT)};{DEMO_MAP_JS}\n</script>")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike route vehicle typology</title>
<style>
{STYLE}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:28px; margin:0 0 6px; }} h2 {{ font-size:19px; margin:0 0 10px; }}
.lead {{ max-width:760px; margin:0 0 14px; }}
.keys {{ display:flex; flex-wrap:wrap; gap:6px 16px; margin:0 0 24px; font-size:14px; }}
.key i {{ display:inline-block; width:14px; height:14px; border-radius:3px; margin-right:6px; vertical-align:-2px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:14px; margin:0 0 28px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.card h2 {{ font-size:17px; }} .card p {{ margin:0 0 10px; }}
.demo {{ margin:0 0 32px; }}
.pair {{ display:grid; grid-template-columns:minmax(0, 2fr) minmax(0, 1fr); gap:12px; }}
@media (max-width:800px) {{ .pair {{ grid-template-columns:1fr; }} .map {{ aspect-ratio:4/3; }} }}
video {{ width:100%; aspect-ratio:16/9; background:#000; border-radius:8px; display:block; }}
.map {{ width:100%; height:100%; min-height:240px; border-radius:8px; border:1px solid var(--line); background:var(--panel); }}
.nomap {{ display:flex; align-items:center; justify-content:center; padding:16px; text-align:center; }}
.map .leaflet-tile-pane {{ filter:grayscale(0.5); }}
@media (prefers-color-scheme: dark) {{ .map .leaflet-tile-pane {{ filter:{DARK_TILES}; }} }}
</style></head><body><main>
<h1>Bike route vehicle typology</h1>
<p class="lead">From a handlebar camera, every vehicle around the rider and how it relates to the rider, with the
riding context, traffic signals, signs (typed and read) and licence plates. Box colours:</p>
<div class="keys">{legend}</div>
<div class="cards">
  {"" if not gps else """<div class="card"><h2>Exposure map</h2><p>Where along the rides the rider met traffic, parked
  cars and close passes, from the detections and the GPS tracks.</p><a class="button" href="map/">Open the map</a></div>"""}
  <div class="card"><h2>Review page</h2><p>The model's answers - vehicles, signs, plates - most likely errors first.
  Confirm or correct with one key; decisions stay in your browser until you download them.</p>
  <a class="button" href="review/">Open the review page</a></div>
  <div class="card"><h2>Your own video</h2><p>Runs in Google Colab on a free GPU: nothing to install. Results go to
  your Google Drive.</p><a class="button" href="{COLAB}">Open in Colab</a></div>
  <div class="card"><h2>Code and results</h2><p>Per-ride tables, accuracy and limitations, and the pipeline.</p>
  <a class="button" href="{REPO}">GitHub</a></div>
</div>{"".join(demo(*d) for d in DEMOS)}
</main>{maps}</body></html>
'''


# The exposure map: heat of per-second counts (or the riding context) as coloured lines over all rides, and the events.
# The heat is drawn like Strava's: thin lines, no blur. Each second's stretch of road is coloured by the counts summed
# over every second of the rides shown within about a cell of it, so overlapping rides and waits come out brighter, and
# the colours are the same at every zoom.
EXPOSURE_MAP_JS = """
const $ = s => document.querySelector(s);
const map = L.map("map", {preferCanvas: true});
L.tileLayer(TILE_URL, {attribution: TILE_CREDIT, maxZoom: 19}).addTo(map);
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
<link rel="stylesheet" href="{LEAFLET}/leaflet.css">
<style>
:root {{ --panel:rgba(22,22,21,0.92); --ink:#ececea; --muted:#a3a39d; --line:#3a3a36; --accent:#63a4e8; }}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; height:100%; background:#0e0e0d; color:var(--ink); font:15px/1.45 system-ui, sans-serif; }}
#map {{ position:absolute; inset:0; background:#0e0e0d; }}
#map .leaflet-tile-pane {{ filter:{DARK_TILES}; }}
.panel {{ position:absolute; z-index:1000; top:12px; left:12px; width:310px; max-width:calc(100% - 24px);
  max-height:calc(100% - 24px); overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; }}
.panel h1 {{ font-size:18px; margin:0 0 4px; }} .panel p {{ margin:6px 0; }}
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
  <h1>Exposure map</h1>
  <p class="muted">Each second of the rides placed by its GPS track, with what the camera saw then. Each ride's
  start and end are left out, within 400 to 800 m of them.</p>
  <div id="layers"></div>
  <p><select id="ride" aria-label="Ride"><option value="-1">All rides</option></select></p>
  <label><input type="checkbox" id="events" checked> Close passes and overtakes</label>
  <div id="legend"></div>
  <p class="muted"><a href="../">Back to the demos</a></p>
</div>
<script src="{LEAFLET}/leaflet.js"></script>
<script>
const TILE_URL = {json.dumps(TILE_URL)};
const TILE_CREDIT = {json.dumps(TILE_CREDIT)};
const HEAT_RAMP = {json.dumps(HEAT_RAMP)};{EXPOSURE_MAP_JS}
</script></body></html>
'''


def write_pages(out: Path) -> None:
    rides = ride_gps()
    (out / "index.html").write_text(index_page(demo_gps(rides)), encoding="utf-8")
    if rides:
        (out / "map").mkdir(exist_ok=True)
        (out / "map" / "exposure.json").write_text(json.dumps(exposure_data(rides), separators=(",", ":")),
                                                    encoding="utf-8")
        (out / "map" / "index.html").write_text(map_page(), encoding="utf-8")
    print(f"pages written: {len(rides)} rides with GPS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "site")
    parser.add_argument("--per-queue", type=int, default=20, help="Items per queue on the review page")
    parser.add_argument("--render-again", action="store_true", help="Render the demos even if they are there")
    parser.add_argument("--pages-only", action="store_true",
                        help="Only write the pages and the map data (e.g. after bikesafe.gps); keep clips and review")
    args = parser.parse_args()
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
