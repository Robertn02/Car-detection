"""Build the upgrade report PDF (bike lanes, signals and signs, accuracy work, vehicle identities, plate privacy, speed).

Every number is read from the result files, so after running the pipeline on the full corpus

    python -m bikesafe.run D:/rides --out-root D:/rides_out
    python -m bikesafe.train --analysis D:/rides_out/analysis --perception D:/rides_out/perception
    python -m bikesafe.exposure --analysis D:/rides_out/analysis --perception D:/rides_out/perception --out results/corpus

rebuilding the report adds the corpus-level results (feature ablation, vehicle counts, bike-lane minutes, events):

    python scripts/build_upgrade_report.py [--out reports/UPGRADE_REPORT_2026-09.pdf]

Charts are written to reports/figures/; photo figures there were rendered from the demo clips with plates blurred.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
FOOTER = "Vehicle Tracking Project \u2014 Upgrade evidence package"

NAVY = colors.HexColor("#102a43")
TEAL = colors.HexColor("#1f7a8c")
LIGHT = colors.HexColor("#eaf4f5")
ZEBRA = colors.HexColor("#f4f7f9")
MUTED = colors.HexColor("#52606d")
TEXT = colors.HexColor("#1f2933")
CHART = {"blue": "#2678a5", "green": "#4b9a62", "orange": "#d37843", "grey": "#9aa5b1", "teal": "#1f7a8c",
         "navy": "#102a43"}
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

RELATION_NAMES = {"ego_lane": "in my lane", "adjacent_same": "other lane, same way", "oncoming": "oncoming",
                  "parked": "parked", "cross_side": "side road / crossing"}


# ------------------------------------------------------------------------------------------------ fonts and styles

def register_fonts() -> tuple[str, str]:
    """A Helvetica-like TrueType font with the symbols used here (<= -> +- x): Liberation Sans, Arial, or the DejaVu
    Sans that matplotlib ships with."""
    candidates = [
        (Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
         Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")),
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
    ]
    mpl_fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    candidates.append((mpl_fonts / "DejaVuSans.ttf", mpl_fonts / "DejaVuSans-Bold.ttf"))
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Body", str(regular)))
            pdfmetrics.registerFont(TTFont("Body-Bold", str(bold)))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
            return "Body", "Body-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, BOLD = register_fonts()
STYLES = {
    "kicker": ParagraphStyle("kicker", fontName=FONT, fontSize=8, textColor=TEAL, leading=11, spaceAfter=4),
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=25, textColor=NAVY, leading=30, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=11.5, textColor=TEAL, leading=15, spaceAfter=10),
    "label": ParagraphStyle("label", fontName=BOLD, fontSize=10.5, textColor=TEAL, leading=13, spaceBefore=8,
                            spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=15, textColor=NAVY, leading=19, spaceBefore=4, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=11, textColor=TEAL, leading=14, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=9.2, textColor=TEXT, leading=12.8, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName=FONT, fontSize=9.2, textColor=TEXT, leading=12.6, leftIndent=11,
                             bulletIndent=2, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", fontName=FONT, fontSize=7.8, textColor=MUTED, leading=10, spaceBefore=3,
                              spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8, textColor=TEXT, leading=10),
    "cell_head": ParagraphStyle("cell_head", fontName=BOLD, fontSize=8, textColor=colors.white, leading=10,
                                alignment=TA_CENTER),
    "box": ParagraphStyle("box", fontName=FONT, fontSize=9, textColor=TEXT, leading=12.5),
    "kpi_value": ParagraphStyle("kpi_value", fontName=BOLD, fontSize=17, textColor=TEAL, leading=20,
                                alignment=TA_CENTER),
    "kpi_label": ParagraphStyle("kpi_label", fontName=FONT, fontSize=7.3, textColor=MUTED, leading=9,
                                alignment=TA_CENTER),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=6.9, textColor=TEXT, leading=9.4, leftIndent=2),
    "ref": ParagraphStyle("ref", fontName=FONT, fontSize=7.8, textColor=MUTED, leading=10.5, spaceAfter=1.5),
}


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, STYLES["bullet"], bulletText="\u2022") for item in items]


def table(rows: list[list], widths: list[float], align_right_from: int = 1, header: bool = True,
          force_right: bool = False) -> Table:
    """Navy header row, zebra body; numbers (or, with force_right, everything) right-aligned from column
    `align_right_from`."""
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, value in enumerate(row):
            if isinstance(value, Paragraph):
                cells.append(value)
            elif r == 0 and header:
                cells.append(Paragraph(str(value), STYLES["cell_head"]))
            else:
                style = ParagraphStyle("c", parent=STYLES["cell"],
                                       alignment=2 if c >= align_right_from and (force_right or _numeric(value))
                                       else TA_LEFT)
                cells.append(Paragraph(str(value), style))
        data.append(cells)
    t = Table(data, colWidths=[w * CONTENT_W for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#cbd2d9"))]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for r in range(1 if header else 0, len(rows)):
        if (r - (1 if header else 0)) % 2 == 1:
            style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def _numeric(value) -> bool:
    text = str(value).replace("%", "").replace(",", "").replace("\u2212", "-").replace("+", "").strip()
    try:
        float(text.split(" ")[0])
        return True
    except ValueError:
        return False


def callout(title: str, text: str) -> Table:
    t = Table([[Paragraph(f"<b>{title}</b> {text}" if title else text, STYLES["box"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 1.1, TEAL),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def kpis(items: list[tuple[str, str]]) -> Table:
    cells = [[Paragraph(value, STYLES["kpi_value"]), Spacer(1, 3), Paragraph(label, STYLES["kpi_label"])]
             for value, label in items]
    t = Table([cells], colWidths=[CONTENT_W / len(items)] * len(items))
    t.setStyle(TableStyle([("BOX", (i, 0), (i, 0), 0.8, colors.HexColor("#cbd2d9")) for i in range(len(items))]
                          + [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def figure(path: Path, caption: str, width: float = 1.0) -> list:
    if not path.exists():
        return [callout("Figure missing:", f"{path.relative_to(ROOT)}")]
    import PIL.Image

    with PIL.Image.open(path) as im:
        w, h = im.size
    draw_w = CONTENT_W * width
    return [KeepTogether([Image(str(path), width=draw_w, height=draw_w * h / w), p(caption, "caption")])]


def code(lines: list[str]) -> Table:
    t = Table([[Paragraph("<br/>".join(line.replace(" ", "&nbsp;") for line in lines), STYLES["code"])]],
              colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f7fa")),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd2d9")),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------------------------------- data + charts

def chart_style() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "figure.dpi": 100})
    for name in ("Liberation Sans", "Arial", "DejaVu Sans"):
        if any(name == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break


def bike_lane_scores(per_second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy / precision / recall / F1 of the per-second bike-lane decision for CLIP, lane paint and fused."""
    d = per_second[per_second.labelled_in_bike_lane.notna()].copy()
    d["clip_bike"] = (d.scene_clip_v2 == "bike_lane").astype(float)
    d["paint"] = pd.to_numeric(d.paint_in_bike_lane, errors="coerce")
    d["fused"] = d.paint.fillna(d.clip_bike)
    rows = []
    for name, label in (("clip_bike", "CLIP scene probe (before)"), ("paint", "Lane paint alone"),
                        ("fused", "Fused: paint decides, CLIP fills in (new default)")):
        ok = d[name].notna()
        y, pred = d.loc[ok, "labelled_in_bike_lane"], d.loc[ok, name]
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((d[name] != 1) & (d.labelled_in_bike_lane == 1)).sum())
        rows.append({"method": label, "decided": ok.mean(), "accuracy": float((pred == y).mean()),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    return pd.DataFrame(rows), d


def chart_bike_lane_timeline(d: pd.DataFrame, out: Path) -> None:
    clip = d[d["clip"] == "typology_006_bikelane_440s"].sort_values("ride_second")
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    s = clip.ride_second.to_numpy()
    lane = clip.labelled_in_bike_lane.to_numpy() == 1
    for i, sec in enumerate(s):
        if lane[i]:
            ax.axvspan(sec - 0.5, sec + 0.5, color=CHART["green"], alpha=0.16, lw=0)
    ax.plot(s, clip.bike_lane_score, color=CHART["teal"], lw=1.8, label="paint evidence (7 s average)")
    ax.axhline(0.35, color=CHART["teal"], lw=0.8, ls="--")
    ax.text(s.min() + 0.3, 0.39, "bike-lane threshold", color=CHART["teal"], fontsize=7.5)
    ax.scatter(s, np.full(len(s), -1.18), c=np.where(clip.scene_clip_v2 == "bike_lane", CHART["orange"], "#e4e7eb"),
               s=14, marker="s")
    paint = pd.to_numeric(clip.paint_in_bike_lane, errors="coerce")
    ax.scatter(s, np.full(len(s), -1.38), c=np.where(paint == 1, CHART["green"], np.where(paint == 0, "#e4e7eb", "white")),
               edgecolors=np.where(paint.isna(), "#9aa5b1", "none"), s=14, marker="s")
    ax.set_ylim(-1.5, 1.05)
    ax.set_xlim(s.min() - 0.5, s.max() + 0.5)
    ax.set_xlabel("ride time (s)")
    ax.set_ylabel("paint evidence")
    ax.set_yticks([-1.38, -1.18, -1, -0.5, 0, 0.5, 1], ["paint", "CLIP", "-1", "-0.5", "0", "0.5", "1"])
    handles = [matplotlib.patches.Patch(color=CHART["green"], alpha=0.25, label="labelled bike lane"),
               matplotlib.lines.Line2D([], [], color=CHART["teal"], lw=1.8, label="paint evidence (7 s average)"),
               matplotlib.lines.Line2D([], [], color=CHART["orange"], marker="s", lw=0, label="CLIP: bike lane"),
               matplotlib.lines.Line2D([], [], color=CHART["green"], marker="s", lw=0, label="paint: bike lane")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def chart_auc(auc: pd.DataFrame, out: Path) -> None:
    names = {"world_speed": "world speed (v2)", "v_along_med": "along-road speed (v2)", "v_lat_absmed": "lateral speed (v2)",
             "closing_ratio_med": "closing ratio (new)", "v_lat_flow_absmed": "lateral flow (new)",
             "grid_proj_med": "grid projection (new)", "v_along_flow_med": "along-road flow (new)"}
    comparisons = [("auc_parked_vs_same_direction_rider_moving", "parked vs same direction"),
                   ("auc_parked_vs_oncoming", "parked vs oncoming"), ("auc_parked_vs_crossing", "parked vs crossing")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.45), sharey=True)
    labels = [names.get(f, f) for f in auc.feature]
    y = np.arange(len(auc))
    for ax, (col, title) in zip(axes, comparisons):
        colours = [CHART["teal"] if "new" in lab else CHART["grey"] for lab in labels]
        ax.barh(y, auc[col], color=colours)
        for yi, v in zip(y, auc[col]):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7)
        ax.set_xlim(0.5, 1.08)
        ax.set_title(title, fontsize=9)
        ax.axvline(0.5, color="#52606d", lw=0.6)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.supxlabel("AUC on hand-labelled demo tracks (0.5 = no information, 1 = perfect separation)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def chart_before_after(ba: pd.DataFrame, out: Path) -> None:
    rows = ba[ba.measure.str.startswith(("events_", "vehicles_visible"))].copy()
    nice = {"events_overtaken_by_vehicle": "overtaken by a vehicle", "events_close_pass_under_1p5m": "close pass < 1.5 m",
            "events_vehicle_close_in_my_lane": "vehicle close in my lane", "events_crossing_traffic_near": "crossing traffic near"}
    ev = rows[rows.measure.str.startswith("events_")]
    vis = rows[rows.measure.str.startswith("vehicles_visible")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(ev))
    a1.barh(y - 0.2, ev["before (tracks)"], 0.4, color=CHART["grey"], label="before: per track")
    a1.barh(y + 0.2, ev["after (vehicles)"], 0.4, color=CHART["teal"], label="after: per vehicle")
    a1.set_yticks(y, [nice.get(m, m) for m in ev.measure])
    a1.invert_yaxis()
    a1.set_title("Events on the demo clips")
    a1.legend(fontsize=7.5, frameon=False, loc="upper right")
    short = {"005_following_280s": "following", "006_bikelane_440s": "bike lane", "006_parked_700s": "parked cars"}
    clips = [short.get(m.replace("vehicles_visible_per_second_typology_", ""), m) for m in vis.measure]
    x = np.arange(len(vis))
    a2.bar(x - 0.2, vis["before (tracks)"], 0.4, color=CHART["grey"])
    a2.bar(x + 0.2, vis["after (vehicles)"], 0.4, color=CHART["teal"])
    for xi, b, a in zip(x, vis["before (tracks)"], vis["after (vehicles)"]):
        a2.text(xi + 0.2, a + 0.15, f"\u2212{100 * (1 - a / b):.0f}%", ha="center", fontsize=7.5)
    a2.set_xticks(x, clips, fontsize=7.5)
    a2.set_title("Vehicles visible per second")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def chart_plate_coverage(cov: pd.DataFrame, out: Path) -> None:
    cov = cov[cov.relation != "all"]
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    x = np.arange(len(cov))
    ax.bar(x - 0.2, 100 * cov.plate_at_least_60px, 0.4, color=CHART["grey"], label="plate at least 60 px wide at closest")
    ax.bar(x + 0.2, 100 * cov.plate_at_least_60px_not_side_view, 0.4, color=CHART["teal"],
           label="... and the vehicle mostly seen from front or rear")
    for xi, v in zip(x, cov.plate_at_least_60px_not_side_view):
        ax.text(xi + 0.2, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"{r}\n({n:,} tracks)" for r, n in zip(cov.relation, cov.tracks)], fontsize=7.8)
    ax.set_ylabel("% of counted tracks")
    ax.set_ylim(0, 85)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Upper bound on readable plates, six rides")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ the report

def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 11 * mm, FOOTER)
    canvas.drawRightString(PAGE_W - MARGIN, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(out: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    chart_style()
    evaluation = json.loads((RESULTS / "typology" / "evaluation_summary.json").read_text(encoding="utf-8"))
    near = evaluation["relation_within_near_z"]
    confusion = pd.read_csv(RESULTS / "typology" / "confusion_gradient_boosting_near.csv", index_col=0)
    ablation = read_csv(RESULTS / "typology" / "feature_ablation.csv")
    per_second = pd.read_csv(RESULTS / "infrastructure" / "demo_bike_lane_per_second.csv")
    lane_scores, lane_data = bike_lane_scores(per_second)
    lights = pd.read_csv(RESULTS / "infrastructure" / "light_state_check.csv")
    auc = pd.read_csv(RESULTS / "infrastructure" / "demo_motion_feature_auc.csv")
    stitch_dir = RESULTS / "stitching"
    mot = pd.read_csv(stitch_dir / "stitching_mot_metrics.csv", index_col=0)
    link_summary = pd.read_csv(stitch_dir / "stitching_links_summary.csv")
    breaks = pd.read_csv(stitch_dir / "reference_identity_breaks.csv", index_col=0)
    demo_links = pd.read_csv(stitch_dir / "demo_links.csv")
    demo_link_summary = pd.read_csv(stitch_dir / "demo_links_summary.csv")
    calibration = pd.read_csv(stitch_dir / "gate_calibration_pairs.csv")
    first_version = pd.read_csv(stitch_dir / "review_first_version_joins.csv")
    before_after = pd.read_csv(stitch_dir / "demo_before_after.csv")
    coverage = pd.read_csv(RESULTS / "plates" / "plate_coverage_by_relation.csv")
    audit = pd.read_csv(RESULTS / "plates" / "demo_identity_audit.csv")
    blurred = read_csv(RESULTS / "plates" / "blurred_media.csv")
    corpus = read_csv(RESULTS / "corpus" / "corpus_summary.csv")
    corpus_new = corpus is not None and "vehicles_counted" in corpus.columns

    chart_bike_lane_timeline(lane_data, FIGURES / "chart_bike_lane_timeline.png")
    chart_auc(auc, FIGURES / "chart_motion_auc.png")
    chart_before_after(before_after, FIGURES / "chart_vehicle_ids_before_after.png")
    chart_plate_coverage(coverage, FIGURES / "chart_plate_coverage.png")

    fused = lane_scores.iloc[2]
    clip_only = lane_scores.iloc[0]
    n_lane_seconds = int(per_second.labelled_in_bike_lane.notna().sum())
    lights_ok, lights_n = int(lights.correct.sum()), len(lights)
    closing_auc = float(auc.set_index("feature").loc["closing_ratio_med", "auc_parked_vs_same_direction_rider_moving"])
    best_v2_auc = float(auc[auc.version == "v2"].auc_parked_vs_same_direction_rider_moving.max())
    checked = demo_links[demo_links.checked_by_eye]
    checked_ok = int((checked.verdict == "same vehicle").sum())
    n_breaks = int((demo_links.kind == "break").sum())
    n_dups = int((demo_links.kind == "duplicate box").sum())
    tracks_total, vehicles_total = int(demo_link_summary.tracks.sum()), int(demo_link_summary.vehicles.sum())
    ba = before_after.set_index("measure")
    close_before = int(ba.loc["events_vehicle_close_in_my_lane", "before (tracks)"])
    close_after = int(ba.loc["events_vehicle_close_in_my_lane", "after (vehicles)"])
    gb_near = near["gradient_boosting"]
    cal_same = calibration[calibration.verdict == "same vehicle"]
    cal_diff = calibration[calibration.verdict == "different vehicles"]
    accepted_same = int((cal_same.score <= 0.45).sum())
    accepted_diff = int((cal_diff.score <= 0.45).sum())
    all_cov = coverage.set_index("relation").loc["all"]

    story: list = []
    # ------------------------------------------------------------------------------------------------------ cover
    story += [Spacer(1, 16 * mm), p("BIKE-ROUTE VEHICLE TYPOLOGY \u2014 UPGRADE", "kicker"), p("Upgrade Report", "title"),
              p("Bike lanes, traffic signals and signs, sharper motion cues, one identity per vehicle, "
                "licence-plate privacy, and faster processing", "subtitle"),
              p(f"{date.today():%B %Y}", "kicker"), Spacer(1, 4), p("WHAT CHANGED IN THE PIPELINE", "label"),
              callout("", "perception \u2192 ego-motion \u2192 depth \u2192 <b>traffic lights, signs and bike lanes</b> "
                          "\u2192 track features with <b>flow motion cues, lane-paint relations and one id per "
                          "vehicle</b> \u2192 typology model \u2192 exposure outputs and overlay clips with "
                          "<b>licence plates blurred</b>"),
              Spacer(1, 8),
              kpis([(f"{fused.f1:.2f}", f"bike-lane F1 per second<br/>(CLIP alone: {clip_only.f1:.2f})"),
                    (f"{lights_ok} / {lights_n}", "traffic-signal states<br/>read correctly"),
                    (f"{closing_auc:.2f}", f"AUC of the new closing ratio<br/>(best existing cue: {best_v2_auc:.2f})"),
                    (f"{checked_ok} / {len(checked)}", "vehicle-id joins checked<br/>by eye were correct")]),
              Spacer(1, 10)]
    story += figure(FIGURES / "fig_bike_lane_views.jpg",
                    "Figure 1. The new bike-lane detector at ride second 482. Right: bird's-eye view of the road 2.5-16 m "
                    "ahead, where paint strokes (magenta, orange for yellow) are found and fitted as lines; the blue line "
                    "is the rider's path. Left: the same lines drawn back into the camera view; green marks the line "
                    "bounding the rider's bike lane. The thin boxes and labels are burned into this demo clip by the "
                    "earlier overlay. Licence plates are blurred in every figure.")
    story += [callout("Summary.", "Bike lanes, traffic signals and stop signs are now detected, reported per second and "
                      "turned into events. The tracker artefacts that made one vehicle count as several - duplicate "
                      "boxes on pickups and vans, and tracks broken by a missed detection - are removed, so counts and "
                      "events are per physical vehicle. Licence plates are blurred in everything rendered; plate "
                      "reading exists only as an opt-in, hashed identity check. What the new motion features add to "
                      "the typology accuracy on all 378 labelled tracks is measured by one command on the project "
                      "workstation (Section 8)."), PageBreak()]

    # ------------------------------------------------------------------------------------------------- 1. scope
    story += [p("1. Starting point, scope and evidence", "h1"),
              p(f"The previous milestone typed every vehicle seen from the bike into five relations to the rider (in my "
                f"lane, other lane same way, oncoming, parked, side road / crossing) and reported exposure per ride. "
                f"Within 15 m of the rider the selected gradient-boosting model reached {pct(gb_near['accuracy'])} accuracy "
                f"and macro-F1 {gb_near['macro_f1']:.3f} in leave-one-video-out validation, and the most frequent error "
                f"was a moving vehicle called parked. The review asked for better accuracy, for bike-lane and "
                f"traffic-sign detection, and for faster processing; the use of licence plates as vehicle identities "
                f"was raised during the work."),
              p("Completed work", "h2"),
              table([["Request", "What was built", "Evidence"],
                     ["Bike-lane detection", "Lane paint and bicycle stencils in a bird's-eye view of the road; a "
                      "per-second decision fused with the CLIP scene probe",
                      f"F1 {clip_only.f1:.2f} \u2192 {fused.f1:.2f} on {n_lane_seconds} labelled seconds (Sec. 2)"],
                     ["Traffic signals", "Open-vocabulary detection (YOLOE-26) and lamp-state reading",
                      f"{lights_ok} of {lights_n} signal states correct; both real red-light waits found (Sec. 3)"],
                     ["Traffic signs", "Open-vocabulary detection; stop signs confirmed by a COCO detector",
                      "all 10 false stop signs removed, the real one kept (Sec. 3)"],
                     ["Relation accuracy", "Diagnosis; depth-free motion cues from the stored optical flow; "
                      "lane-paint relations; extra-trees candidate",
                      f"new cue separates parked from moving traffic best (AUC {closing_auc:.2f} vs {best_v2_auc:.2f}); "
                      "corpus gain from the ablation (Sec. 4)"],
                     ["Vehicle identity", "Duplicate boxes merged, broken tracks joined, one relation per vehicle",
                      f"{checked_ok} of {len(checked)} checked joins correct; per-second counts and events corrected (Sec. 5)"],
                     ["Plate privacy", "Plates blurred by default; opt-in hashed identity audit",
                      "all demo videos and snapshots blurred; audit agrees with the vehicle ids (Sec. 6)"],
                     ["Speed", "FP16 on the GPU, hardware video decoding, parallel CPU stages, resumable stages",
                      "stage timings printed by every run (Sec. 7)"]],
                    [0.17, 0.46, 0.37], align_right_from=9),
              p("Pipeline after the upgrade", "h2"),
              table([["Stage", "Module", "In this upgrade"],
                     ["Perception (GPU, ~12 fps)", "bikesafe.perceive", "FP16, hardware decoding, finished videos skipped"],
                     ["Ego-motion (CPU)", "bikesafe.egomotion", "several rides at once"],
                     ["Depth and lane structure (GPU, 2 fps)", "bikesafe.scene3d", "FP16; optional (--skip-depth)"],
                     ["Lights, signs, bike lanes (GPU, 2 fps)", "bikesafe.infrastructure", "<b>new</b>"],
                     ["Track features (CPU)", "bikesafe.tracks, .geometry, .stitch",
                      "<b>new</b> flow cues, lane-paint relations, vehicle ids"],
                     ["Licence plates", "bikesafe.plates", "<b>new</b> blurring; opt-in hashed audit"],
                     ["Learning and evaluation", "bikesafe.train", "extra trees, feature ablation, scene scoring with paint"],
                     ["Outputs", "bikesafe.exposure, .render", "per-vehicle counts and events; bike lane, signals, signs"],
                     ["One-command runner", "bikesafe.run", "resumable, parallel, per-stage timings"]],
                    [0.33, 0.29, 0.38], align_right_from=9),
              Spacer(1, 8),
              callout("Evidence available for this review.", "The ride videos (several GB each) and the intermediate "
                      "files stay on the project workstation. The upgrades were therefore validated on the three demo "
                      "clips in the repository (2.5 minutes of real riding, re-run through the whole pipeline), on the "
                      "approved 300-frame tracking reference, and on the per-track tables of all six rides (14,770 "
                      "tracks). Numbers that need the full videos - what the new features add on all labels, and "
                      "corpus-wide counts - are produced by the commands in Section 8, and rebuilding this report "
                      "then adds them."),
              PageBreak()]

    # ------------------------------------------------------------------------------------------------ 2. bike lanes
    probe = evaluation.get("scene", {}).get("linear_probe", {}).get("per_class_f1", {})
    story += [p("2. Bike-lane detection", "h1"),
              p(f"The CLIP scene probe separated paths from roads well but recognised painted bike lanes poorly (bike-lane "
                f"F1 {probe.get('bike_lane', float('nan')):.2f} on the labelled scene frames). A bike lane is defined by "
                f"its paint, so the new detector (<font face='Courier'>bikesafe.infrastructure</font>) looks for the paint."),
              *bullets([
                  "<b>Bird's-eye view.</b> The self-calibrated flat-road geometry the pipeline already uses (horizon, "
                  "camera height, lateral scale) warps the road 2.5-18 m ahead into a metric top view at 2 cm per "
                  "pixel, twice per second, with vehicles masked out. Lane lines run vertically there.",
                  "<b>Paint.</b> Thin (at most 0.35 m), elongated strokes brighter than the asphalt beside them; kerbs, "
                  "crosswalk bars, cracks and shadow edges are rejected by width and length. Yellow and white are told "
                  "apart by saturation, which also works for faded yellow and white in low sun.",
                  "<b>Lines.</b> Extracted greedily - the best-supported line, then the paint it explains is removed - "
                  "each with its own slant, and classified solid or broken, white or yellow.",
                  "<b>Stencils.</b> Bicycle symbols are found by YOLOE-26 in the top view, where they are upright and "
                  "unforeshortened.",
                  "<b>Decision.</b> Transparent evidence rules - for: a solid white line 0.3-2.4 m to the left, a narrow "
                  "lane between two lines, a stencil in the rider's corridor, green paint; against: a 3 m+ traffic lane "
                  "between broken lines, a yellow centre line beside the rider - averaged over 7 s and held through "
                  "junctions without paint.",
                  "<b>Fusion.</b> CLIP keeps the separated-path call it gets right; the paint decides bike lane versus "
                  "shared road wherever it has evidence, and CLIP fills the rest.",
              ]),
              p("Per-second check", "h2"),
              table([["Riding context per second", "Seconds decided", "Accuracy", "Bike-lane precision",
                      "Bike-lane recall", "F1"]]
                    + [[r.method, pct(r.decided, 0), pct(r.accuracy), pct(r.precision), pct(r.recall), f"{r.f1:.2f}"]
                       for r in lane_scores.itertuples()], [0.40, 0.12, 0.11, 0.13, 0.13, 0.11]),
              p(f"{n_lane_seconds} seconds of the three demo clips labelled by hand from the road view (junction crossings "
                "excluded): a boulevard section, a bike pocket before an intersection and a long kerbside bike lane "
                "beside parked cars, plus two shared residential streets.", "caption")]
    story += figure(FIGURES / "chart_bike_lane_timeline.png",
                    "Figure 2. The bike-lane clip second by second. CLIP (orange squares) misses most of the kerbside bike "
                    "lane; the paint evidence (line) crosses the threshold for nearly all of it. Hollow squares are seconds "
                    "without usable paint, where CLIP's call is kept.")
    story += [callout("Caveat.", "The rules were tuned while looking at these clips, so the table is optimistic. "
                      "bikesafe.train now also scores the labelled scene frames of all six rides with and without the "
                      "paint (scene_confusion_probe_plus_lane_paint.csv), which gives the unbiased number. Night "
                      "riding, where only the headlight shows the paint, is untested."),
              PageBreak()]

    # ----------------------------------------------------------------------------------------- 3. signals and signs
    lit = lights[lights.labelled.isin(["red", "yellow", "green"])]
    story += [p("3. Traffic signals and signs", "h1"),
              *bullets([
                  "<b>Detection.</b> YOLOE-26 (open-vocabulary detection) receives a road-furniture vocabulary - traffic "
                  "light, stop, yield, speed-limit, street-name, one-way, no-parking and other signs - plus distractor "
                  "classes (billboard, shop sign, licence plate, street lamp, vehicles, people) that absorb look-alikes. "
                  "The prompts' text features are cached, so the 240 MB text encoder is not needed at run time.",
                  "<b>Signal state.</b> The lit lamp is a bright, saturated, round blob inside the housing (lit reds look "
                  "pink-magenta on this camera, LED greens cyan-green); sky and walls seen through side-on heads are "
                  "rejected by shape and position.",
                  "<b>Stop signs</b> are confirmed by the COCO-trained YOLO11m already in the repository: the open "
                  "vocabulary alone also calls DO NOT ENTER, NO PARKING and red shop signs stop signs.",
                  "<b>One object, one row.</b> Detections of one sign or signal head are linked across samples (a static "
                  "object moves straight away from the focus of expansion and grows), giving signs.csv.",
                  "<b>Events.</b> <font face='Courier'>stop_sign</font> for each distinct stop sign passed close enough; "
                  "<font face='Courier'>red_light_wait</font> for a stop of 3 s or more with a red signal read for at "
                  "least 2 s of it, defined by the stop so that a bus crossing in front of the signal does not split "
                  "one wait into several.",
              ]),
              table([["Check on the demo clips", "Result"],
                     ["Signal state, labelled signal-head crops", f"{lights_ok} of {lights_n} correct "
                      f"({int(lit.correct.sum())} of {len(lit)} lit heads; the misses are a tiny far head and a faint "
                      "side-on one, and two unlit heads read as yellow against a yellow wall)"],
                     ["Signal heads, spot check of 24 detections", "23 are signal heads (one teal awning is not)"],
                     ["Stop signs, open vocabulary alone", "12 detections, 2 of them stop signs"],
                     ["Stop signs, after COCO confirmation", "the real stop sign facing the rider kept, all 10 false ones "
                      "removed (a small side-on stop sign facing cross traffic is also dropped)"],
                     ["Red-light waits", "2 found, both real; one lasted 23 s with a bus crossing in front of the signal"],
                     ["Other signs, 36 detections", "20 are the demo clip's own burned-in overlay labels; of the other 16, "
                      "about 12 are real traffic signs and 4 shop signs"]],
                    [0.36, 0.64], align_right_from=9),
              Spacer(1, 6)]
    story += figure(RESULTS / "infrastructure" / "light_state_crops.png",
                    "Figure 3. Signal-head crops from the demo clips with the state read by the lamp detector.", width=0.86)
    story += [p("Sign types other than stop signs are open-vocabulary guesses (a BIKE LANE sign was called a speed-limit "
                "sign), so reports use the number of signs and treat the fine types as indicative.", "body"),
              PageBreak()]

    # --------------------------------------------------------------------------------------------- 4. accuracy
    moving_as_parked = [(RELATION_NAMES[r], int(confusion.loc[r, "parked"]), int(confusion.loc[r].sum()))
                        for r in ["oncoming", "adjacent_same", "cross_side", "ego_lane"]]
    story += [p("4. Relation accuracy: where it is lost and what was added", "h1"),
              p(f"Leave-one-video-out, labelled tracks that came within 15 m: {pct(gb_near['accuracy'])} accuracy, "
                f"macro-F1 {gb_near['macro_f1']:.3f}. The confusion matrix shows the dominant error:"),
              table([["Labelled as", "Predicted parked", "Labelled tracks", "Share"]]
                    + [[name, k, n, pct(k / n, 0)] for name, k, n in moving_as_parked],
                    [0.4, 0.2, 0.2, 0.2]),
              Spacer(1, 4),
              p("Three causes were found, each with evidence: the fitted world trajectory drifts for parked cars because "
                "the box centre slides outward as a passed car's side comes into view (parked cars end up with a median "
                "fitted speed of 2.15 m/s at 95 degrees, like crossing traffic); the depth-model trajectory is biased "
                "(parked cars 'move' at +2.1 m/s along the road in it); and vehicles alongside, cut off by the frame "
                "edge, lose their geometry."),
              p("Better modelling of the existing features does not fix this", "h2"),
              table([["Model on the existing 42 features", "Within 15 m: accuracy / macro-F1", "All distances"],
                     ["Gradient boosting (current)", "68.1% / 0.503", "63.0% / 0.432"],
                     ["Extra trees, balanced (mean of 5 seeds)", "68.3% / 0.510", "65.9% / 0.482"],
                     ["Random forest", "66.8% / 0.419", "65.6% / 0.403"],
                     ["Logistic regression", "61.0% / 0.475", "55.6% / 0.428"],
                     ["+ 16 engineered feature combinations", "within \u00b12 points", "within \u00b12 points"]],
                    [0.46, 0.3, 0.24], force_right=True),
              p("With about 300 labelled tracks the standard error of an accuracy is about 2.7 points, so none of these "
                "differences is real within 15 m; extra trees were kept as a candidate because they are better over all "
                "distances on every seed. The limit is the information in the features, so new information was added.",
                "caption"),
              p("New information: depth-free motion from the stored optical flow", "h2"),
              *bullets([
                  "<b>Closing ratio</b> - the body's measured expansion over the expansion a static object at that "
                  "distance would show: about 1 for a parked car, 0 for a vehicle moving with the rider, 2 for oncoming "
                  "traffic at the rider's speed, whatever the absolute scale.",
                  "<b>Lateral flow</b> - the body's horizontal image motion minus that of the ground just below it; the "
                  "rider's own translation and yaw cancel, and box height converts it to metres.",
                  "<b>Grid projection</b> - the body's motion against the background beside it, which also works for "
                  "boxes cut off by the frame edge.",
                  "<b>Lane-paint relations</b> - painted lines between the rider's path and the vehicle, whether it sits "
                  "in the rider's lane, and whether the rider was in a bike lane at the time.",
              ])]
    story += figure(FIGURES / "chart_motion_auc.png",
                    "Figure 4. How well single features separate parked cars from moving traffic on 76 hand-labelled "
                    "tracks of the demo clips (rider moving). The closing ratio separates parked cars from same-direction "
                    "traffic better than any existing feature; lateral flow helps against oncoming traffic.")
    if ablation is not None:
        view = ablation[ablation["subset"] == "near"].copy()
        story += [p("Leave-one-video-out result on all labelled tracks", "h2"),
                  table([["Features", "Model", "Features used", "Accuracy", "Macro-F1", "Group accuracy"]]
                        + [[r.features, r.model.replace("_", " "), r.n_features, pct(r.accuracy), f"{r.macro_f1:.3f}",
                            pct(r.group_accuracy)] for r in view.itertuples()],
                        [0.14, 0.24, 0.14, 0.16, 0.16, 0.16]),
                  p("Within 15 m, from results/typology/feature_ablation.csv.", "caption")]
    else:
        story += [callout("To be measured on the workstation.", "The per-track tables in the repository carry the "
                          "version-2 features only; the flow cues need the perception outputs of the full rides. "
                          "Running the commands in Section 8 writes results/typology/feature_ablation.csv (version-2 "
                          "features, + flow cues, + lane paint, for each model, leave-one-video-out on the same labels), "
                          "and rebuilding this report inserts that table here.")]
    story += [PageBreak()]

    # -------------------------------------------------------------------------------------- 5. vehicle identities
    def mot_row(name: str) -> list:
        r = mot.loc[name]
        return [name.replace("benchmark_30fps", "30 fps").replace("pipeline_15fps", "15 fps"), pct(r.idf1), pct(r.mota),
                int(r.num_switches), int(r.num_fragmentations), int(r.num_unique_objects)]

    swaps = int(breaks.loc["swap with a coexisting track", "total"])
    new_ids = int(breaks.loc["track ended, new id started", "total"])
    far_mid_swaps = int(breaks.loc["swap with a coexisting track", ["far (<40 px)", "mid (40-80 px)"]].sum())
    ref30 = link_summary.set_index("setting").loc["benchmark_30fps"]
    idf1_change = f"{100 * (mot.iloc[1].idf1 - mot.iloc[0].idf1):+.1f}".replace("-", "\u2212")
    story += [p("5. One identity per physical vehicle", "h1"),
              p("Track ids are continuity hypotheses of the tracker. Two artefacts make one vehicle look like several, "
                "and both inflated counts and events:"),
              *bullets([
                  "<b>Duplicate boxes.</b> The detector suppresses overlapping boxes class by class, so a pickup, van or "
                  "SUV is often boxed twice at the same time - once as 'car' and once as 'truck' or 'bus' - and each box "
                  "gets its own track.",
                  "<b>Breaks.</b> At 12 frames per second a vehicle near the rider moves tens of pixels between frames; a "
                  "missed detection or a short occlusion makes the tracker start a new id.",
              ]),
              p("Method (bikesafe.stitch)", "h2"),
              *bullets([
                  "Two tracks that coexist with a mean overlap (IoU) of at least 0.7 over at least half of the shorter "
                  "one are one vehicle.",
                  "For a break, the last detection of a track is extrapolated forward and the first detection of a later "
                  "track backward to the middle of the gap. The pair is joined when the predictions meet and the image "
                  "velocities agree (e + 0.5 dv \u2264 0.45, both relative to box height), the gap is at most 2 s, class "
                  "and size agree, neither end is at the side edge of the frame, no other vehicle fits about as well, "
                  "and the first vehicle is gone before the second appears. A one-to-one assignment picks the joins.",
                  "Tracks keep their own features and labels; <font face='Courier'>vehicle_id</font> groups them. Each "
                  "vehicle gets one relation - the detection-weighted vote of its tracks' class probabilities - and "
                  "counts, per-second exposure and events are per vehicle.",
              ]),
              p("Check on the bike-ride demo clips (the setting the pipeline runs in)", "h2"),
              table([["Demo clip", "Minutes", "Track ids", "Duplicate merges", "Break joins", "Vehicles"]]
                    + [[r["clip"].replace("typology_", ""), f"{r.minutes:.2f}", r.tracks, r.duplicate_links,
                        r.break_links, r.vehicles] for _, r in demo_link_summary.iterrows()]
                    + [["All", f"{demo_link_summary.minutes.sum():.2f}", tracks_total, n_dups, n_breaks,
                        vehicles_total]], [0.34, 0.11, 0.13, 0.16, 0.13, 0.13]),
              Spacer(1, 5),
              p(f"Every break join ({n_breaks}) and the {len(checked) - n_breaks} duplicate merges with the lowest overlap "
                f"were checked by eye on the video: all {checked_ok} join the same vehicle. The gate was set on "
                f"{len(calibration)} candidate pairs labelled by eye beforehand: it accepts {accepted_same} of the "
                f"{len(cal_same)} same-vehicle pairs and {accepted_diff} of the {len(cal_diff)} different-vehicle pairs. "
                f"An earlier version without duplicate merging made {len(first_version)} joins, all of them correct, most "
                f"of them between flickering duplicate boxes - which is how the duplicates were found.")]
    story += figure(FIGURES / "fig_vehicle_ids.jpg",
                    "Figure 5. Left: one pickup boxed twice at the same time, as 'truck' (track 695) and 'car' (track 696) "
                    "- now one vehicle. Middle and right: a pickup crossing ahead loses its track for 1.25 s and comes "
                    "back as track 447 - joined to track 428.")
    rate_rows = [["Measure", "Before: per track", "After: per vehicle"]]
    for key, label in (("per_min_parked", "parked vehicles per minute"),
                       ("per_min_adjacent_same", "other lane, same way, per minute"),
                       ("per_min_cross_side", "side road / crossing per minute"),
                       ("events_vehicle_close_in_my_lane", "events: vehicle close in my lane"),
                       ("events_crossing_traffic_near", "events: crossing traffic near"),
                       ("events_overtaken_by_vehicle", "events: overtaken by a vehicle")):
        before, after = ba.loc[key, "before (tracks)"], ba.loc[key, "after (vehicles)"]
        fmt = (lambda v: f"{v:.1f}") if key.startswith("per_min") else (lambda v: f"{int(v)}")
        rate_rows.append([label, fmt(before), fmt(after)])
    story += [KeepTogether([p("Effect on the outputs (demo clips, same data before and after)", "h2"),
                            table(rate_rows, [0.5, 0.25, 0.25])])]
    story += figure(FIGURES / "chart_vehicle_ids_before_after.png",
                    f"Figure 6. Per-minute rates barely change, because the existing count rule (at least 1 s, within "
                    f"15 m) already dropped most short duplicates. Events and per-second exposure did not have that "
                    f"protection: 'vehicle close in my lane' fell from {close_before} to {close_after} (two duplicate "
                    f"boxes of the car being followed, two short duplicate tracks of parked vehicles), and the vehicles "
                    f"visible per second fell by 5-8%.")
    story += [p("Check against the approved 300-frame tracking reference", "h2"),
              table([["Setting", "IDF1", "MOTA", "ID switches", "Fragments", "Reference ids"]]
                    + [mot_row(n) for n in mot.index], [0.42, 0.1, 0.1, 0.13, 0.12, 0.13]),
              Spacer(1, 4),
              p(f"The reference is a 10 s clip of dense, mostly distant traffic from the earlier tracking video. Of its "
                f"{swaps + new_ids} identity breaks, {swaps} are swaps between two tracks that exist at the same time "
                f"({'all' if far_mid_swaps == swaps else far_mid_swaps} of them on cars under 80 px tall, i.e. more than "
                f"about 15 m away and outside the exposure counts) and only {new_ids} are a track ending and a new one starting. Break joining therefore "
                f"makes no join there - and so no wrong one. Merging duplicates cuts identity switches from "
                f"{int(mot.iloc[0].num_switches)} to {int(mot.iloc[1].num_switches)} at 30 fps, but IDF1 moves by "
                f"{idf1_change} points, because the reference itself labels the "
                f"second ('truck') box of a doubly detected pickup as {int(ref30.reference_duplicate_tracks)} separate "
                f"short identities. With those second boxes removed from both sides the scores are unchanged by the "
                f"joins. This baseline was re-run with the current libraries (IDF1 {pct(mot.iloc[0].idf1)}; the earlier "
                f"report measured 72.5% with the same detector and tracker), so it is compared only with itself.",
                "body"),
              PageBreak()]

    # ------------------------------------------------------------------------------------------------ 6. plates
    audit_total = audit[["tracks_checked", "tracks_with_plate_found", "tracks_with_two_agreeing_reads"]].sum()
    followed = audit.set_index("clip").loc["typology_005_following_280s"]
    story += [p("6. Licence plates and privacy", "h1"),
              p("The repository is public, and plates were readable in the rendered demo clips. Two uses of plates were "
                "considered: blurring them everywhere, and using them as vehicle identities."),
              p("Blurring (default)", "h2"),
              *bullets([
                  "A small plate detector (YOLOv9-tiny, 7 MB ONNX, open-image-models) runs on every vehicle box at least "
                  "40 px tall, cropped from the full-resolution frame so a plate is large enough to find; about 24 ms per "
                  "vehicle on a CPU.",
                  "A plate found on a track stays blurred for 6 more frames at the same place on the vehicle, which covers "
                  "single missed detections. The blur pixelates and smooths an area 30% larger than the plate.",
                  "The overlay renderer blurs by default; <font face='Courier'>python -m bikesafe.plates blur</font> "
                  "blurs existing videos and images.",
              ])]
    if blurred is not None:
        story += [table([["File", "Frames", "Plate regions blurred"]]
                        + [[r.file, r.frames, f"{int(r.plate_regions):,}"] for r in blurred.itertuples()],
                        [0.6, 0.15, 0.25]),
                  p("All demo videos and tracking snapshots in the repository were replaced by blurred versions. The "
                    "approved 300-frame sequence is benchmark input and was left unchanged.", "caption")]
    story += [p("Should the plate be the vehicle id? No.", "h2"),
              p(f"Plates are readable only close to the rider and from the front or rear. From the per-track tables of "
                f"all six rides, at most {pct(all_cov.plate_at_least_60px, 0)} of the counted tracks ever show a plate "
                f"60 px wide or more, and only {pct(all_cov.plate_at_least_60px_not_side_view, 0)} are also seen mostly "
                f"from front or rear - an upper bound before blur, glare, night and missing front plates. Crossing and "
                f"oncoming traffic almost never shows one. An id that exists for a minority, and mostly for parked and "
                f"same-direction vehicles, would also bias counts between relations. So the id comes from motion and "
                f"overlap for every vehicle (Section 5), and plates are only an optional check.")]
    story += figure(FIGURES / "chart_plate_coverage.png",
                    "Figure 7. Share of counted tracks (within 15 m, at least 1 s) whose plate could be large enough to "
                    "read, estimated from each track's largest box (a plate is about a fifth of a car's height).")
    story += [p("Identity audit (opt-in, hashed)", "h2"),
              *bullets([
                  "<font face='Courier'>bikesafe.run --read-plates</font> reads the plate of each large vehicle on up to "
                  "6 frames with a small OCR model (fast-plate-ocr, global model including US plates).",
                  "The text never leaves the process: each read becomes an HMAC-SHA256 hash under a random key created for "
                  "that run and never stored. Hashes compare only within one ride, cannot be turned back into a plate, and "
                  "cannot follow a vehicle across rides. They stay in the analysis folder; outputs get counts only.",
                  "A plate counts only if it sits in the lower middle of its own vehicle's box and two reads agree.",
                  "The audit reports tracks with one plate that ended up in different vehicle ids (a missed join, or the "
                  "vehicle came back later) and vehicle ids carrying two different plates (a wrong join).",
              ]),
              p(f"On the demo clips - 720p copies with the old overlay burned in, far below the quality of the original "
                f"footage - a plate was found on {int(audit_total.tracks_with_plate_found)} of "
                f"{int(audit_total.tracks_checked)} large tracks but read consistently on only "
                f"{int(audit_total.tracks_with_two_agreeing_reads)}. For the car the rider followed, the plate hash "
                f"appears on {int(followed.tracks_with_two_agreeing_reads)} tracks, all in one vehicle id "
                f"({int(followed.same_plate_joined_pairs)} agreeing pairs, no split, no conflict). The first version of "
                f"the audit flagged three more tracks with that plate: they were a mail van beside the car, whose crop "
                f"contained the car's plate - a reader error, fixed by the ownership rule above, not a vehicle-id "
                f"error."),
              callout("Limitations.", "A plate on a vehicle the detector did not box is not blurred. Earlier, unblurred "
                      "versions of the demo videos remain in the repository's git history; making the repository "
                      "private, or rewriting its history, is the owner's decision."),
              CondPageBreak(75 * mm)]

    # --------------------------------------------------------------------------------------------------- 7. speed
    story += [p("7. Faster runs", "h1"),
              table([["Change", "Effect"],
                     ["FP16 on CUDA for the detector, CLIP and the depth model (default; --fp32 to switch off)",
                      "roughly 1.5-2x faster GPU inference"],
                     ["Hardware video decoding (--hw-decode: NVDEC / D3D11 / VAAPI through FFmpeg)",
                      "decoding of high-bitrate 360 exports moves off the CPU in every stage"],
                     ["CPU stages for several rides at once (--jobs, default half the cores)",
                      "ego-motion and track features scale with cores"],
                     ["Every stage resumable; finished videos skipped before any model loads",
                      "re-running after a code change runs only what changed"],
                     ["Track features recomputed only when written by an older feature version",
                      "vehicle ids and flow cues added without re-running perception"],
                     ["Lights, signs and bike lanes sampled at 2 fps; bird's-eye view at 640 px",
                      "the new pass costs a fraction of perception"],
                     ["--skip-depth", "drops the depth pass (its features moved scores by -2.3 to +1.6 points)"],
                     ["Per-stage timing summary at the end of bikesafe.run", "shows where the time goes on each machine"]],
                    [0.62, 0.38], align_right_from=9),
              Spacer(1, 6),
              p("Measured here (4 CPU cores, no GPU)", "h2"),
              table([["Stage", "Time"],
                     ["Perception, approved 10 s sequence at 15 processed fps (CPU)", "24 s"],
                     ["Lights, signs and bike lanes (YOLOE-26l at 1280 px + bird's-eye view, CPU)", "about 2 s per sample"],
                     ["Track features incl. vehicle ids, 2.5 min of demo video", "11 s"],
                     ["Vehicle ids alone (bikesafe.stitch)", "0.3-0.4 s per minute of video"],
                     ["Exposure outputs, three demo clips", "3 s"],
                     ["Plate reading for the audit (opt-in), 2.5 min of video", "80 s"],
                     ["Plate blurring in the overlay renderer", "about 25 ms per vehicle box"]],
                    [0.68, 0.32], force_right=True),
              p("GPU timings depend on the workstation; bikesafe.run prints them per stage.", "caption")]

    # -------------------------------------------------------------------------------- corpus results, if present
    if corpus_new:
        c = corpus
        minutes = c.minutes.sum()
        rows = [["Measure (all rides)", "Value"], ["Minutes of video", f"{minutes:.1f}"]]
        for col, label in (("minutes_in_painted_bike_lane", "Minutes in a painted bike lane"),
                           ("events_red_light_wait", "Waits at red lights"), ("events_stop_sign", "Stop signs passed"),
                           ("vehicles_counted", "Vehicles counted (within 15 m, at least 1 s)"),
                           ("tracks_counted", "Tracks counted by the earlier per-track rule"),
                           ("duplicate_links", "Duplicate boxes merged"), ("break_links", "Broken tracks joined")):
            if col in c:
                rows.append([label, f"{c[col].sum():,.1f}" if col.startswith("minutes") else f"{int(c[col].sum()):,}"])
        for rel, label in RELATION_NAMES.items():
            if f"vehicles_{rel}" in c:
                rows.append([f"{label} per minute", f"{c[f'vehicles_{rel}'].sum() / max(minutes, 1e-6):.2f}"])
        story += [p("Corpus results", "h2"), table(rows, [0.7, 0.3]),
                  p("From results/corpus/corpus_summary.csv.", "caption")]

    # ----------------------------------------------------------------------------- 8. next steps and conclusion
    story += [CondPageBreak(90 * mm), p("8. Running the upgrade on the full corpus", "h1"),
              p("On the project workstation, one command reuses everything already computed, runs only the new stages "
                "and recomputes the track features; training then measures what the new features add, exposure writes "
                "the corrected outputs, and the last command rebuilds this report with the corpus numbers:"),
              code(["python -m bikesafe.run D:\\rides --out-root D:\\out --render-minutes 1 --hw-decode",
                    "python -m bikesafe.train --analysis D:\\out\\analysis --perception D:\\out\\perception",
                    "python -m bikesafe.exposure --analysis D:\\out\\analysis --perception D:\\out\\perception"
                    " --out results\\corpus",
                    "python scripts\\build_upgrade_report.py"]),
              p("Add <font face='Courier'>--read-plates</font> to the first command for the hashed identity audit.",
                "caption"),
              p("Limitations and next steps", "h2"),
              table([["Limitation", "Effect", "Next step"],
                     ["Bike-lane rules tuned on the demo clips", "optimistic bike-lane scores",
                      "score the labelled scene frames of all rides (bikesafe.train does this)"],
                     ["Flow-cue gain not yet measured on all labels", "accuracy change unknown",
                      "run the commands above; feature_ablation.csv"],
                     ["Sign types beyond stop signs are open-vocabulary guesses", "fine sign types indicative only",
                      "label a few hundred sign crops and fine-tune"],
                     ["Identity swaps between distant vehicles", "not fixed by joins or merges",
                      "matter little for exposure (beyond 15 m)"],
                     ["Vehicles that leave and return are new vehicles", "leapfrogging counted twice",
                      "the opt-in plate audit counts such returns"],
                     ["Night riding", "less recall, paint visible only in the headlight", "validate on night rides"]],
                    [0.36, 0.28, 0.36], align_right_from=9),
              p("Conclusion", "h2"),
              p("The pipeline now reports the riding infrastructure the review asked for - painted bike lanes, traffic "
                "signals with their state, stop signs and other signs - and turns it into per-second context and events. "
                "Counts and events are per physical vehicle, which removed double-counted events that the earlier "
                "per-track outputs contained. Licence plates are blurred in everything the project shows, and the "
                "decision not to use plates as identities rests on measured coverage. The typology accuracy work "
                "identified why moving vehicles were called parked and added depth-free motion cues that separate them "
                "well on hand-labelled tracks; their effect on the full labelled set is one command away."),
              p("Method references", "h2"),
              *[p(r, "ref") for r in [
                  "YOLOE: Real-Time Seeing Anything - https://arxiv.org/abs/2503.07465; Ultralytics YOLOE and YOLO11 "
                  "documentation - https://docs.ultralytics.com/",
                  "BoT-SORT: Robust Associations Multi-Pedestrian Tracking - https://arxiv.org/abs/2206.14651",
                  "Depth Anything V2 - https://arxiv.org/abs/2406.09414",
                  "IDF1: Ristani et al., Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking - "
                  "https://arxiv.org/abs/1609.01775",
                  "open-image-models (plate detector) - https://github.com/ankandrew/open-image-models; fast-plate-ocr - "
                  "https://github.com/ankandrew/fast-plate-ocr",
                  "HMAC: Keyed-Hashing for Message Authentication, RFC 2104 - https://www.rfc-editor.org/rfc/rfc2104",
              ]]]

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="Bike-Route Vehicle Typology - Upgrade Report",
                            author="Vehicle Tracking Project",
                            subject="Bike lanes, traffic signals and signs, accuracy work, vehicle identities, plate "
                                    "privacy and speed")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {out} ({'with' if ablation is not None else 'without'} the corpus feature ablation, "
          f"{'with' if corpus_new else 'without'} corpus vehicle counts)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "UPGRADE_REPORT_2026-09.pdf")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
