"""
Compare several YOLO prediction runs over the same clip.

The question this answers is the one that actually matters for this footage:
does a bigger model (or a bigger input size) recover the small, distant cars
that YOLO11n at the default 640 input misses?

Usage:
    python scripts/compare_runs.py <name>=<labels_dir> [<name>=<labels_dir> ...]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

FRAME_RE = re.compile(r"_(\d+)$")
WIDTH, HEIGHT = 1920, 1080


def load(labels_dir: Path) -> pd.DataFrame:
    rows = []
    for txt in labels_dir.glob("*.txt"):
        match = FRAME_RE.search(txt.stem)
        if not match:
            continue
        frame = int(match.group(1))
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            xc, yc, w, h = (float(p) for p in parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else float("nan")
            rows.append({"frame": frame, "w": w, "h": h, "conf": conf})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["area_px"] = (df["w"] * WIDTH) * (df["h"] * HEIGHT)
    df["bucket"] = pd.cut(
        df["area_px"],
        bins=[0, 32 * 32, 96 * 96, float("inf")],
        labels=["small", "medium", "large"],
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=labels_dir")
    ap.add_argument("--total-frames", type=int, default=None)
    args = ap.parse_args()

    frames = {}
    for spec in args.runs:
        if "=" not in spec:
            sys.exit(f"Expected name=labels_dir, got {spec!r}")
        name, path = spec.split("=", 1)
        df = load(Path(path))
        if df.empty:
            print(f"WARNING: no detections found for {name}")
            continue
        frames[name] = df

    if not frames:
        sys.exit("No runs loaded.")

    total_frames = args.total_frames or max(int(df["frame"].max()) for df in frames.values())
    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    header = f"{'run':<26}{'dets':>9}{'frames':>9}{'per-frame':>11}{'mean conf':>11}{'<0.50':>8}"
    print(header)
    print("-" * len(header))
    for name, df in frames.items():
        per_frame = len(df) / total_frames
        print(
            f"{name:<26}{len(df):>9,}{df['frame'].nunique():>9,}"
            f"{per_frame:>11.2f}{df['conf'].mean():>11.3f}"
            f"{(df['conf'] < 0.50).mean():>8.1%}"
        )

    print()
    print("=" * 78)
    print("BY BOX SIZE  (small = under 32x32 px, i.e. distant cars)")
    print("=" * 78)
    header2 = f"{'run':<26}{'small':>10}{'medium':>10}{'large':>10}{'small conf':>12}"
    print(header2)
    print("-" * len(header2))
    for name, df in frames.items():
        counts = df["bucket"].value_counts()
        small_conf = df.loc[df["bucket"] == "small", "conf"].mean()
        print(
            f"{name:<26}{counts.get('small', 0):>10,}{counts.get('medium', 0):>10,}"
            f"{counts.get('large', 0):>10,}{small_conf:>12.3f}"
        )

    baseline_name = list(frames)[0]
    baseline = frames[baseline_name]
    print()
    print("=" * 78)
    print(f"CHANGE vs BASELINE ({baseline_name})")
    print("=" * 78)
    for name, df in frames.items():
        if name == baseline_name:
            continue
        total_delta = (len(df) - len(baseline)) / len(baseline)
        bc = baseline["bucket"].value_counts()
        dc = df["bucket"].value_counts()
        print(f"\n{name}:")
        print(f"  total detections : {total_delta:+.1%}")
        for bucket in ["small", "medium", "large"]:
            b, d = bc.get(bucket, 0), dc.get(bucket, 0)
            if b:
                print(f"  {bucket:<17}: {d - b:+,} ({(d - b) / b:+.1%})")


if __name__ == "__main__":
    main()
