"""bikesafe.gps on a synthetic ride: GPX reading, clock offset from the speeds, privacy zones, activity matching."""

import numpy as np
import pandas as pd

from bikesafe.gps import PRIVACY_M, load_activities, match_activity, read_gpx, video_gps, zone_radii

START = pd.Timestamp("2026-02-24T17:00:00", tz="UTC")  # 09:00 local (PST)


def ride(seconds: int = 900, seed: int = 0) -> pd.DataFrame:
    """A ride due north whose speed varies (stops at lights), one GPS point a second."""
    rng = np.random.default_rng(seed)
    speed = np.clip(np.convolve(rng.uniform(-2, 9, seconds), np.ones(15) / 15, mode="same"), 0, None)
    lat = 34.0 + np.cumsum(speed) / 111_195.0
    return pd.DataFrame({"time": START + pd.to_timedelta(np.arange(seconds), unit="s"), "lat": lat,
                         "lon": -118.3, "speed": speed})


def write_gpx(path, track: pd.DataFrame) -> None:
    points = "".join(f'<trkpt lat="{r.lat:.7f}" lon="{r.lon:.7f}"><time>{r.time:%Y-%m-%dT%H:%M:%SZ}</time></trkpt>'
                     for r in track.itertuples())
    path.write_text('<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">'
                    f"<trk><trkseg>{points}</trkseg></trk></gpx>")


def test_the_camera_clock_offset_is_found_from_the_speeds(tmp_path):
    track = ride()
    write_gpx(tmp_path / "ride.gpx", track)
    gpx = read_gpx(tmp_path / "ride.gpx")
    assert len(gpx) == len(track) and str(gpx.time.dt.tz) == "UTC"
    # the camera's clock runs 7 s behind the GPS's: its 09:02:00 is the GPS's 09:02:07
    seconds = np.arange(600)
    true_time = START + pd.Timedelta(seconds=120) + pd.to_timedelta(seconds, unit="s")
    timeline = pd.DataFrame({"second": seconds,
                             "timestamp_local": (true_time - pd.Timedelta(seconds=7)).tz_convert("America/Los_Angeles")
                             .tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
                             "ego_speed_mps": 0.8 * track.speed.to_numpy()[120 + seconds]})  # the video's scale is off
    table, sync = video_gps(timeline, gpx)
    assert sync["synced"] and sync["offset_s"] == 7 and sync["speed_correlation"] > 0.9
    placed = table.dropna(subset=["lat"])
    expected = track.set_index(track.time.dt.floor("s")).lat
    assert np.allclose(placed.lat, expected.loc[true_time[placed.index]].round(6), atol=1e-6)


def test_positions_near_where_the_track_starts_or_ends_are_blank(tmp_path):
    track = ride()
    write_gpx(tmp_path / "ride.gpx", track)
    timeline = pd.DataFrame({"second": np.arange(900),
                             "timestamp_local": (track.time.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
                                                 .dt.strftime("%Y-%m-%dT%H:%M:%S")),
                             "ego_speed_mps": track.speed})
    gpx = read_gpx(tmp_path / "ride.gpx")
    table, sync = video_gps(timeline, gpx)
    r_start, r_end = zone_radii(gpx)
    from_start = (track.lat - track.lat.iloc[0]) * 111_195.0
    from_end = (track.lat.iloc[-1] - track.lat) * 111_195.0
    hidden = (from_start < r_start - 5) | (from_end < r_end - 5)
    assert table.lat[hidden].isna().all() and table.gps_speed_mps[hidden].isna().all()
    assert table.in_privacy_zone[hidden].all()
    shown = (from_start > r_start + 5) & (from_end > r_end + 5)
    assert table.lat[shown].notna().all() and sync["seconds_in_privacy_zone"] >= hidden.sum()


def test_the_privacy_zones_differ_between_rides_but_not_between_runs():
    radii = [zone_radii(ride(seed=seed)) for seed in range(20)]
    assert all(PRIVACY_M <= r <= 2 * PRIVACY_M for pair in radii for r in pair)
    assert len({round(r) for pair in radii for r in pair}) > 30  # no common radius to fit circles with
    assert zone_radii(ride(seed=3)) == radii[3]


def test_a_video_is_matched_to_the_activity_recorded_while_it_was_filmed(tmp_path):
    (tmp_path / "activities.csv").write_text(
        "Activity ID,Activity Date,Elapsed Time,Filename,Elapsed Time\n"
        '1,"Feb 24, 2026, 4:58:00 PM",1200,activities/1.gpx,1200\n'
        '2,"Feb 24, 2026, 6:00:00 PM",1200,activities/2.gpx,1200\n')
    activities = load_activities(tmp_path)
    hit = match_activity(activities, START, START + pd.Timedelta(minutes=15))
    assert hit is not None and hit["Activity ID"] == 1
    assert match_activity(activities, START + pd.Timedelta(hours=3), START + pd.Timedelta(hours=3, minutes=10)) is None
