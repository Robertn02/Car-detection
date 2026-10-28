"""Join the rides to their GPS tracks (a Strava export): where the rider was in each second of a video.

    python -m bikesafe.gps --strava ../Strava

--strava is a Strava export folder: activities.csv and activities/<id>.gpx. Each video is matched to the activity
recorded while it was filmed (Strava's dates are UTC, the video file names local time). The camera's clock and the
GPS's need not agree, so the offset between them is measured: the bike's speed from the video (ego_speed_mps of
timeline_1s.csv) against the GPS speed, at the lag within +-MAX_LAG_S where the two agree best. A ride whose speeds
agree less than MIN_SYNC_CORRELATION keeps the clocks as they are, flagged in gps_sync.json.

    python -m bikesafe.gps --calibrate

measures the camera's metric scale against the GPS (calibrate, below) into results/corpus/metric_calibration.json,
which bikesafe.tracks applies.

Writes results/corpus/<video>/gps_1s.csv (one row per second of video: lat, lon, GPS speed) and
results/corpus/gps_sync.json (activity, offset, agreement of the speeds). Positions and GPS speeds near where a GPS
track starts or ends (usually home or work) are left blank, like Strava's privacy zones: these tables and the project
page are public. Each zone's radius is drawn between PRIVACY_M and twice that from the track itself (see zone_radii).
The GPX files themselves stay out of the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from bikesafe.common import ROOT, read_json, write_json
from bikesafe.geometry import DEFAULT_FLOW_SCALE, DEFAULT_METRIC_SCALE

TIMEZONE = "America/Los_Angeles"  # the rides' local time, as in the video file names
PRIVACY_M = 400.0
MAX_LAG_S = 180
MIN_SYNC_CORRELATION = 0.5
MIN_OVERLAP_S = 120
SMOOTH_S = 5  # both speeds are averaged over this many seconds before they are compared
EARTH_M = 6_371_000.0
UNIX = pd.Timestamp("1970-01-01", tz="UTC")
METRIC_CALIBRATION = ROOT / "results" / "corpus" / "metric_calibration.json"
CALIBRATION_MIN_GPS_MPS = 2.0  # seconds the GPS says the rider moved at least this fast are compared
CALIBRATION_PRIOR_TRACKS = 30  # a ride's distance scale is pulled towards the pooled one as if by this many tracks


def read_gpx(path: Path) -> pd.DataFrame:
    """Track points (time UTC, lat, lon), in time order."""
    root = ET.parse(path).getroot()
    ns = root.tag[:root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    rows = [(p.findtext(f"{ns}time"), float(p.attrib["lat"]), float(p.attrib["lon"])) for p in root.iter(f"{ns}trkpt")]
    track = pd.DataFrame(rows, columns=["time", "lat", "lon"]).dropna()
    track["time"] = pd.to_datetime(track.time, utc=True)
    return track.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(a))


def per_second(track: pd.DataFrame) -> pd.DataFrame:
    """The track on a 1-second grid of whole UTC seconds (interpolated: a recording paused while the bike stood still
    has gaps), with the speed over each second."""
    epoch = ((track.time - UNIX) / pd.Timedelta(seconds=1)).to_numpy(float)
    grid = np.arange(np.ceil(epoch[0]), np.floor(epoch[-1]) + 1)
    lat, lon = np.interp(grid, epoch, track.lat), np.interp(grid, epoch, track.lon)
    speed = np.r_[np.nan, distance_m(lat[:-1], lon[:-1], lat[1:], lon[1:])]
    return pd.DataFrame({"epoch": grid.astype(np.int64), "lat": lat, "lon": lon, "speed_mps": speed})


def local_epoch(timestamps: pd.Series) -> np.ndarray:
    """Whole UTC seconds of local wall-clock timestamps."""
    times = pd.to_datetime(timestamps).dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return ((times - UNIX) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def sync_offset(video_epoch: np.ndarray, video_speed: np.ndarray, gps: pd.DataFrame) -> tuple[int, float, int]:
    """Seconds to add to the video's clock to read the GPS's, the correlation of the two speeds at that lag, and the
    seconds they overlap there."""
    smooth = lambda v: pd.Series(v, dtype=float).rolling(SMOOTH_S, center=True, min_periods=1).mean().to_numpy()  # noqa: E731
    v, g = smooth(video_speed), smooth(gps.speed_mps.to_numpy())
    start = int(gps.epoch.iloc[0])
    best = (0, -1.0, 0)
    for lag in range(-MAX_LAG_S, MAX_LAG_S + 1):
        idx = video_epoch + lag - start
        inside = (idx >= 0) & (idx < len(g))
        a, b = v[inside], g[idx[inside]]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < MIN_OVERLAP_S or a[ok].std() == 0 or b[ok].std() == 0:
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if r > best[1]:
            best = (lag, r, int(ok.sum()))
    return best


def zone_radii(track: pd.DataFrame) -> tuple[float, float]:
    """Radii of the privacy zones round the track's first and last points, each between PRIVACY_M and twice that. Were
    they all the same, the ends of the shown routes would lie on circles of known radius round home and work, whose
    centres a few rides give away; drawn from the track's own points, they are the same each run but cannot be worked
    out without the GPX."""
    digest = hashlib.sha256(track[["lat", "lon"]].to_numpy(float).tobytes()).digest()
    u = np.random.default_rng(int.from_bytes(digest[:8], "little")).random(2)
    return PRIVACY_M * (1 + float(u[0])), PRIVACY_M * (1 + float(u[1]))


def private(lat: np.ndarray, lon: np.ndarray, track: pd.DataFrame) -> np.ndarray:
    """Positions inside the privacy zones round the track's first and last points."""
    (first, last), (r_first, r_last) = (track.iloc[0], track.iloc[-1]), zone_radii(track)
    return (distance_m(lat, lon, first.lat, first.lon) < r_first) | (distance_m(lat, lon, last.lat, last.lon) < r_last)


def video_gps(timeline: pd.DataFrame, track: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per second of video: position and GPS speed (blank inside the privacy zones, or outside the track), and how the
    clocks were matched."""
    gps = per_second(track)
    epoch = local_epoch(timeline.timestamp_local)
    lag, r, overlap = sync_offset(epoch, timeline.ego_speed_mps.to_numpy(float), gps)
    synced = r >= MIN_SYNC_CORRELATION
    idx = epoch + (lag if synced else 0) - int(gps.epoch.iloc[0])
    inside = (idx >= 0) & (idx < len(gps))
    out = pd.DataFrame({"second": timeline.second, "timestamp_local": timeline.timestamp_local,
                        "lat": np.nan, "lon": np.nan, "gps_speed_mps": np.nan, "in_privacy_zone": False})
    hit = gps.iloc[idx[inside]]
    out.loc[inside, ["lat", "lon", "gps_speed_mps"]] = hit[["lat", "lon", "speed_mps"]].to_numpy()
    zone = np.zeros(len(out), bool)
    zone[inside] = private(hit.lat.to_numpy(), hit.lon.to_numpy(), track)
    out["in_privacy_zone"] = zone
    out.loc[zone, ["lat", "lon", "gps_speed_mps"]] = np.nan  # summed, the speeds would give the hidden path's length
    out[["lat", "lon"]] = out[["lat", "lon"]].round(6)
    out["gps_speed_mps"] = out.gps_speed_mps.round(2)
    sync = {"offset_s": lag if synced else 0, "speed_correlation": round(r, 3), "synced": synced,
            "seconds_overlapping": overlap, "seconds_with_position": int(out.lat.notna().sum()),
            "seconds_in_privacy_zone": int(zone.sum()), "seconds_outside_track": int((~inside).sum())}
    return out, sync


def load_activities(strava: Path) -> pd.DataFrame:
    """activities.csv with start and end in UTC (the export repeats some column names; the first of each is kept)."""
    a = pd.read_csv(strava / "activities.csv")
    a = a.loc[:, ~a.columns.duplicated()]
    a["start"] = pd.to_datetime(a["Activity Date"], format="%b %d, %Y, %I:%M:%S %p").dt.tz_localize("UTC")
    a["end"] = a.start + pd.to_timedelta(a["Elapsed Time"], unit="s")
    return a


def match_activity(activities: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """The activity overlapping the video the longest, if it covers at least half of it."""
    overlap = (np.minimum(activities.end, end) - np.maximum(activities.start, start)).dt.total_seconds()
    if not len(overlap) or overlap.max() < 0.5 * (end - start).total_seconds():
        return None
    return activities.loc[overlap.idxmax()]


# --------------------------------------------------------------------------------------------------------------------
# metric scale


def metric_scales(video: str, path: Path = METRIC_CALIBRATION) -> tuple[float, float]:
    """(metric_scale, flow_scale) of a ride for bikesafe.geometry: its own, measured against its GPS, or the rides'
    default (a ride without GPS, or a new video); the camera's measured defaults when nothing was calibrated."""
    if not path.exists():
        return DEFAULT_METRIC_SCALE, DEFAULT_FLOW_SCALE
    table = read_json(path)
    entry = table.get("rides", {}).get(video) or table["default"]
    return float(entry["metric_scale"]), float(entry["flow_scale"])


def parked_closing_ratios(kin: pd.DataFrame, tracks: pd.DataFrame, gps_speed: pd.Series) -> list[float]:
    """GPS speed / the closing speed of parked vehicles' distances, one per parked track the rider passed: a parked
    car stands still, so the distance to it shrinks at the rider's speed, which the GPS gives."""
    parked = tracks[(tracks.vehicle_relation == "parked") & (tracks.proba_parked > 0.8) & (tracks.n_det >= 12)].track_id
    ratios = []
    for _, k in kin[kin.track_id.isin(parked) & kin.geom_ok.astype(bool)].groupby("track_id"):
        k = k[k.z_m.between(3.0, 40.0)]
        if len(k) < 8 or np.ptp(k.time_s) < 1.0:
            continue
        slope = np.polyfit(k.time_s, k.z_m, 1)[0]
        seconds = np.arange(int(k.time_s.min()), int(k.time_s.max()) + 1)
        speed = gps_speed.reindex(seconds).mean()
        if np.isfinite(speed) and speed > CALIBRATION_MIN_GPS_MPS and slope < -0.5:
            ratios.append(float(speed / -slope))
    return ratios


def calibrate(corpus: Path, analysis: Path) -> dict:
    """The camera's metric scale, measured against the GPS on every ride that has a track.

    Distances ahead and speeds both come from focal length x camera height (bikesafe.geometry), and the focal length
    is a guess. Two independent measurements against the GPS fix them:
    * distance: parked cars approach at the rider's speed, so GPS speed / their closing speed (from their distance
      in successive frames) is the factor distances ahead are short by - metric_scale. A ride with few parked cars
      passed (at night) is pulled towards the pooled factor;
    * speed: GPS speed / the speed measured from the video (seconds above CALIBRATION_MIN_GPS_MPS); what it needs
      beyond metric_scale is the flow reading the forward rate low - flow_scale (about 1 by day, more at night).
    Both are measured on the uncalibrated scale (the current tables divided by the scales they were made with), so
    calibrating again does not compound. `now` says how far the current tables are from the GPS (1 = calibrated).
    A ride without GPS gets the pooled distance scale and the median flow scale (`default`)."""
    rides, pooled = {}, []
    for path in sorted(corpus.glob("*/gps_1s.csv")):
        video = path.parent.name
        calib = read_json(analysis / video / "calibration.json")
        made_with = float(calib.get("metric_scale", 1.0)), float(calib.get("flow_scale", 1.0))
        gps_speed = pd.read_csv(path).set_index("second").gps_speed_mps.rolling(3, center=True, min_periods=1).median()
        timeline = pd.read_csv(corpus / video / "timeline_1s.csv").set_index("second")
        both = pd.concat([timeline.ego_speed_mps, gps_speed], axis=1).dropna()
        both = both[(both.gps_speed_mps > CALIBRATION_MIN_GPS_MPS) & (both.ego_speed_mps > 0.3)]
        speed_now = float((both.gps_speed_mps / both.ego_speed_mps).median()) if len(both) else float("nan")
        kin = pd.read_parquet(analysis / video / "kinematics.parquet", columns=["track_id", "time_s", "z_m", "geom_ok"])
        tracks = pd.read_csv(corpus / video / "tracks_typed.csv",
                             usecols=["track_id", "vehicle_relation", "proba_parked", "n_det"])
        ratios = parked_closing_ratios(kin, tracks, gps_speed)
        parked_now = float(np.median(ratios)) if ratios else float("nan")
        raw = [r * made_with[0] for r in ratios]  # the same on the uncalibrated scale
        pooled += raw
        rides[video] = {"speed_ratio": speed_now * made_with[0] * made_with[1], "speed_seconds": int(len(both)),
                        "parked_ratio": float(np.median(raw)) if raw else float("nan"), "parked_tracks": len(raw),
                        "now": {"gps_over_video_speed": round(speed_now, 3), "gps_over_parked_closing": round(parked_now, 3)}}
    if not rides:
        return {}
    pooled_ratio = float(np.median(pooled)) if pooled else DEFAULT_METRIC_SCALE
    for entry in rides.values():
        n = entry["parked_tracks"]
        own = entry["parked_ratio"] if n else pooled_ratio
        metric = (n * own + CALIBRATION_PRIOR_TRACKS * pooled_ratio) / (n + CALIBRATION_PRIOR_TRACKS)
        entry["metric_scale"] = round(metric, 3)
        entry["flow_scale"] = round(entry["speed_ratio"] / metric, 3) if np.isfinite(entry["speed_ratio"]) else 1.0
        entry["speed_ratio"], entry["parked_ratio"] = round(entry["speed_ratio"], 3), round(entry["parked_ratio"], 3)
    flows = [e["flow_scale"] for e in rides.values()]
    return {"method": "bikesafe.gps.calibrate: distance from parked cars' closing speed, speed from GPS speed",
            "default": {"metric_scale": round(pooled_ratio, 3), "flow_scale": round(float(np.median(flows)), 3)},
            "rides": rides}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strava", type=Path, default=None, help="Strava export folder (activities.csv, activities/)")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results" / "corpus")
    parser.add_argument("--analysis", type=Path, default=ROOT / "work" / "analysis")
    parser.add_argument("--calibrate", action="store_true",
                        help="Measure the metric scale against the GPS into metric_calibration.json (calibrate)")
    args = parser.parse_args()
    if args.strava is not None:
        join_rides(args.strava, args.corpus)
    if args.calibrate:
        table = calibrate(args.corpus, args.analysis)
        write_json(args.corpus / METRIC_CALIBRATION.name, table)
        for video, entry in table.get("rides", {}).items():
            print(f"{video}: distances x{entry['metric_scale']:.2f} ({entry['parked_tracks']} parked tracks), "
                  f"speed x{entry['metric_scale'] * entry['flow_scale']:.2f}; now {entry['now']}")
        print(f"default (no GPS): {table.get('default')}")
    if args.strava is None and not args.calibrate:
        parser.error("give --strava to join the rides to their tracks, --calibrate to measure the scale, or both")


def join_rides(strava: Path, corpus: Path) -> None:
    """Each ride's gps_1s.csv and gps_sync.json from the Strava export."""
    activities = load_activities(strava)
    report = {}
    for path in sorted(corpus.glob("*/timeline_1s.csv")):
        video = path.parent.name
        timeline = pd.read_csv(path)
        epoch = local_epoch(timeline.timestamp_local)
        start, end = (pd.Timestamp(int(e), unit="s", tz="UTC") for e in (epoch[0], epoch[-1]))
        activity = match_activity(activities, start, end)
        if activity is None:
            report[video] = {"activity": None, "note": "no Strava activity recorded during the video"}
            print(f"{video}: no Strava activity")
            continue
        gpx = strava / str(activity.Filename)
        if not gpx.exists():
            report[video] = {"activity": int(activity["Activity ID"]), "note": f"{activity.Filename} not found"}
            print(f"{video}: {gpx} not found")
            continue
        table, sync = video_gps(timeline, read_gpx(gpx))
        table.to_csv(path.parent / "gps_1s.csv", index=False)
        report[video] = {"activity": int(activity["Activity ID"]), **sync}
        print(f"{video}: activity {report[video]['activity']}, clock offset {sync['offset_s']:+d} s "
              f"(speed correlation {sync['speed_correlation']:.2f}{'' if sync['synced'] else ', NOT synced'}), "
              f"{sync['seconds_with_position']} s placed, {sync['seconds_in_privacy_zone']} s in privacy zones")
    write_json(corpus / "gps_sync.json", report)


if __name__ == "__main__":
    main()
