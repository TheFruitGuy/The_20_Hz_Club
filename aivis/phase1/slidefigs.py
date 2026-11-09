"""Figures of the final midpoint slides, in the deck palette.

    python -m aivis.phase1.slidefigs                  # all figures
    python -m aivis.phase1.slidefigs loudness         # one figure

Reads the Phase I tables (outputs/phase1/tables) and writes results/phase1/slides/<slide>_<name>.png.

    seed_misses        slide 2: how many of the 5 NAVE seeds find a blue whale call, against chance
    listening_window   slide 3: where NAVE listens in pitch, recall at the 2019/20 pitch, eval audio shifted up
    threshold_vs_snr   slide 4: median model score / own threshold per SNR bin, all classes
    error_sketch       slide 4: how each model goes wrong (drawn by hand, no data)
    pitch_window       slide 5: Z-call pitch at Kerguelen, 2014/15 against 2019/20, with NAVE's window
    loudness           slide 5: calls found and false alarms in the loudness probe, -20 to +20 dB
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.interpolate import PchipInterpolator

from .. import paths
from .core import TABLES

OUT = paths.ROOT / "results" / "phase1" / "slides"
DPI = 200

# deck palette
TEAL, ORANGE, INK, GRAY = "#026E97", "#E8702A", "#1F2327", "#5B6770"
LIGHT, MID, GRID, PALE, TINT = "#F1F3F5", "#8A949C", "#E3E7EA", "#C9D0D5", "#E8F2F6"
ORANGE_LIGHT, TEAL_LIGHT, GRAY_LIGHT = "#F6CBB2", "#A6CBD9", "#D5DADE"
FONT = ["Arial", "Liberation Sans", "DejaVu Sans"]

MODELS = ["yolo", "whalevad", "nave"]
NAME = {"yolo": "YOLO", "whalevad": "Whale-VAD", "nave": "NAVE"}
COLOR = {"yolo": MID, "whalevad": TEAL, "nave": ORANGE}
MARKER = {"yolo": "s", "whalevad": "^", "nave": "o"}
CLASS_INDEX = {"bmabz": 0, "d": 1, "bp": 2}
DEV_SITES = ["casey2017", "kerguelen2014", "kerguelen2015"]
SEEDS = ["hit_s0", "hit_s1", "hit_s42", "hit_s666", "hit_s1337"]


def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(TABLES / name)


def style(size: float, **extra) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": size,
                         "axes.edgecolor": PALE, "axes.labelcolor": GRAY, "xtick.color": GRAY,
                         "ytick.color": GRAY, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", **extra})


def title(ax, text: str, size: float = 14) -> None:
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", color=INK, pad=14)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print("[slidefigs]", name)


# ---- numbers shared by slides 3 and 5 ----

def nave_window() -> tuple[float, float]:
    """Pitch shifts (Hz) at which NAVE still finds at least half of the probe Z-calls (linear interpolation)."""
    pr = table("probe_results.parquet")
    sh = pd.concat([pr[pr.probe == "shift"], pr[pr.probe == "base"].assign(value=0.0)])
    r = sh[(sh.model == "nave") & (sh.cls == "bmabz")].groupby("value").detected.mean().sort_index()
    x, y = r.index.to_numpy(), r.to_numpy()
    top = int(np.argmax(y))

    def crossing(i, j):
        return x[i] + (0.5 - y[i]) * (x[j] - x[i]) / (y[j] - y[i])

    lo = next(crossing(i - 1, i) for i in range(top, 0, -1) if y[i - 1] < 0.5 <= y[i])
    hi = next(crossing(i, i + 1) for i in range(top, len(y) - 1) if y[i] >= 0.5 > y[i + 1])
    return float(lo), float(hi)


def zcall_pitch() -> tuple[pd.Series, pd.Series]:
    """Z-call peak frequency at Kerguelen: dev years 2014/15 (labels), eval years 2019/20 (detections)."""
    z = table("zcall_peak_by_year.parquet")
    return z[z.year.isin([2014, 2015])].hz, z[z.year.isin([2019, 2020])].hz


# ---- slide 2 ----

def seed_misses():
    """Share of blue whale calls found by 0 to 5 NAVE seeds, against seeds that fail independently."""
    style(14)
    s = table("seed_agree_labels.parquet")
    s = s[s.cls == "bmabz"]
    chance = np.array([1.0])
    for q in s[SEEDS].mean():                       # Poisson binomial: seeds fail independently
        chance = np.convolve(chance, [1 - q, q])
    obs = s.n_seeds_hit.value_counts(normalize=True).reindex(range(6), fill_value=0).to_numpy()

    fig = plt.figure(figsize=(8.88, 4.52))
    ax = fig.add_axes([0.11, 0.2, 0.87, 0.68])
    k = np.arange(6)
    ax.bar(k, chance, width=0.78, color="none", edgecolor=MID, lw=1.8, ls=(0, (4, 3)), zorder=2)
    ax.bar(k, obs, width=0.52, color=[ORANGE] + [TEAL] * 5, zorder=3)
    for i in k:
        ax.text(i, obs[i] + 0.008, f"{obs[i] * 100:.0f} %", ha="center", va="bottom", fontsize=15,
                fontweight="bold", color=ORANGE if i == 0 else INK, zorder=4)
    ax.text(-0.48, chance[0] + 0.02, f"{chance[0] * 100:.1f} %", ha="center", va="bottom", fontsize=12, color=GRAY)
    for i in (3, 5):
        ax.text(i + 0.42, chance[i] + 0.012, f"{chance[i] * 100:.1f} %", ha="left", va="bottom", fontsize=12,
                color=GRAY)
    ax.set_xticks(k, ["0\nmissed by all", "1", "2", "3", "4", "5\nfound by all"], fontsize=14)
    ax.set_xlabel("how many of the 5 NAVE seeds find the call", fontsize=14, labelpad=8)
    ax.set_ylabel("share of blue whale calls (BMABZ)", fontsize=14)
    ax.set_xlim(-0.6, 5.85)
    ax.set_ylim(0, 0.55)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    handles = [Patch(color=TEAL), Patch(facecolor="none", edgecolor=MID, lw=1.8, ls=(0, (4, 3)))]
    fig.legend(handles, ["what the 5 seeds actually do", "expected if the seeds failed independently"],
               loc="upper left", bbox_to_anchor=(0.115, 1.0), frameon=False, fontsize=13.5, handlelength=2.2,
               handleheight=0.9)
    save(fig, "2_seed_misses")


# ---- slide 3 ----

def _pitch_panel(fig, left, head, lo, hi, ylim):
    ax = fig.add_axes([left, 0.1, 0.11, 0.72])
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.axhspan(lo, hi, color=TINT, zorder=0)
    for y in (lo, hi):
        ax.axhline(y, color=TEAL, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ticks = [25, 25.5, 26, 26.5, 27]
    ax.set_yticks(ticks, [f"{t:g}" if t != 27 else "27 Hz" for t in ticks], fontsize=12)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    fig.text(left - 0.045, 0.905, head, fontsize=14.5, fontweight="bold", color=INK)
    return ax


def _bar(ax, y, color, dashed=False):
    h = 0.2
    kw = dict(facecolor="none", edgecolor=color, lw=2, ls=(0, (3, 2))) if dashed else dict(color=color)
    ax.add_patch(FancyBboxPatch((0.18, y - h / 2), 0.64, h, boxstyle="round,pad=0,rounding_size=0.03",
                                mutation_aspect=0.12, zorder=3, **kw))


def listening_window():
    """Three steps: NAVE's pitch window and the training pitch, the 2019/20 pitch, the 2019/20 audio shifted up."""
    style(14)
    dev, ev = zcall_pitch()
    w_lo, w_hi = nave_window()
    dev_hz, ev_hz = dev.median(), ev.median()
    lo, hi = dev_hz + w_lo, dev_hz + w_hi

    f1 = table("f1_dev_shift.parquet")
    f1 = f1[(f1.model == "nave") & (f1.cls == "bmabz")].groupby("condition")[["tp", "fn"]].sum()
    recall = f1.tp / (f1.tp + f1.fn)
    hours = table("site_stats.parquet").groupby("dataset").duration_s.sum() / 3600
    f2 = table("f2_eval_shift.parquet")
    f2 = f2[(f2.model == "nave") & (f2.cls == "bmabz")].copy()
    f2["n"] = f2.per_hour * f2.site.map(hours)
    per_hour = f2.groupby("condition").n.sum() / hours[f2.site.unique()].sum()
    before, after = per_hour["original"], per_hour.drop("original").iloc[0]

    fig = plt.figure(figsize=(12.48, 3.3))
    ylim = (24.9, 27.3)
    texts = [(f"{recall['original'] * 100:.0f}", TEAL, "of 100 known\nblue whale calls found"),
             (f"{recall.drop('original').iloc[0] * 100:.0f}", ORANGE, "of 100 known\nblue whale calls found"),
             (f"×{after / before:.1f}", TEAL, f"NAVE blue whale calls\nper hour: {before:.1f} → {after:.1f}")]
    heads = ["1  Training pitch", "2  2019/20 pitch", "3  2019/20 audio shifted up"]
    for i, left in enumerate([0.06, 0.395, 0.73]):
        ax = _pitch_panel(fig, left, heads[i], lo, hi, ylim)
        if i == 0:
            _bar(ax, dev_hz, TEAL)
            ax.text(0.5, hi - 0.05, "where NAVE\nlistens", ha="center", va="top", fontsize=12,
                    fontweight="bold", color=TEAL, linespacing=1.0)
        elif i == 1:
            _bar(ax, ev_hz, ORANGE)
        else:
            _bar(ax, ev_hz, ORANGE, dashed=True)
            _bar(ax, dev_hz, TEAL)
            ax.annotate("", (0.5, dev_hz - 0.12), (0.5, ev_hz - 0.05),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6", color=GRAY, lw=2.2),
                        zorder=4)
        big, color, small = texts[i]
        fig.text(left + 0.13, 0.48, big, fontsize=34, fontweight="bold", color=color)
        fig.text(left + 0.13, 0.28, small, fontsize=12.5, color=GRAY, linespacing=1.15)
    save(fig, "3_listening_window")


# ---- slide 4 ----

SNR_BINS = [-np.inf, 0, 2, 4, 6, 10, np.inf]
SNR_LABELS = ["< 0", "0–2", "2–4", "4–6", "6–10", "> 10"]


def threshold_ratio() -> pd.DataFrame:
    """Median of (label peak score / the model's LOSO threshold for that site and class) per SNR bin."""
    thr = json.loads((TABLES / "thresholds.json").read_text())
    lab = table("labels_validation.parquet")[["label_id", "dataset", "cls", "snr_db"]]
    rows = []
    for m in MODELS:
        for site in DEV_SITES:
            p = table(f"peaks/{m}_{site}.parquet").merge(lab, on="label_id")
            t = p.cls.map(lambda c: thr[m]["loso_iou0.3"][site][CLASS_INDEX[c]])
            rows.append(pd.DataFrame({"model": m, "snr": p.snr_db, "ratio": p.peak / t}))
    d = pd.concat(rows)
    d["bin"] = pd.cut(d.snr, SNR_BINS, right=False, labels=range(len(SNR_LABELS)))
    return d.groupby(["bin", "model"], observed=True).ratio.median().unstack()


def _split_line(ax, x, y, color, z):
    """Solid above the threshold, thin and dashed below, split where the line crosses 1."""
    for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:]):
        parts = [(x0, y0, x1, y1)]
        if (y0 - 1) * (y1 - 1) < 0:
            xc = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
            parts = [(x0, y0, xc, 1.0), (xc, 1.0, x1, y1)]
        for a, b, c, d in parts:
            above = min(b, d) >= 1
            ax.plot([a, c], [b, d], color=color, zorder=z, solid_capstyle="round",
                    **(dict(lw=3.2) if above else dict(lw=1.7, ls=(0, (3, 2)), alpha=0.8)))


def threshold_vs_snr():
    """Median score of a labelled call relative to the model's own threshold, by how much the call stands out."""
    style(13)
    r = threshold_ratio()
    x = np.arange(len(SNR_LABELS))
    fig, ax = plt.subplots(figsize=(6.24, 3.24))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.18)
    ax.axhspan(0, 1, color=LIGHT, zorder=0)
    ax.axhline(1, color=INK, lw=1.6, zorder=1)
    for z, m in enumerate(MODELS, start=2):
        y = r[m].to_numpy()
        _split_line(ax, x, y, COLOR[m], z)
        up = y >= 1
        ax.scatter(x[up], y[up], s=85, marker=MARKER[m], color=COLOR[m], zorder=z + 0.5)
        ax.scatter(x[~up], y[~up], s=36, marker=MARKER[m], facecolor="white", edgecolor=COLOR[m], lw=1.5,
                   zorder=z + 0.5)
        ax.plot([], [], color=COLOR[m], lw=3.5, label=NAME[m])
    ax.text(-0.25, 1.04, "detection threshold", fontsize=12, fontweight="bold", color=INK)
    ax.text(5.3, 0.1, "below: missed", ha="right", fontsize=11.5, color=GRAY)
    ax.set_xticks(x, SNR_LABELS)
    ax.set_xlim(-0.3, 5.4)
    ax.set_ylim(0, 2.6)
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("how much the call stands out from the noise (SNR, dB)")
    ax.set_ylabel("model score ÷ threshold")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.2)
    title(ax, "Faint calls never reach the threshold", 15)
    save(fig, "4_threshold_vs_snr")


def error_sketch():
    """Sketch of the typical error of each model (not data): YOLO doubles and misses, Whale-VAD adds noise,
    NAVE merges close calls."""
    style(12)
    fig = plt.figure(figsize=(6.24, 3.24))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1248)
    ax.set_ylim(648, 0)                             # pixel coordinates of the slide image
    ax.axis("off")
    calls = [(355, 447), (520, 612), (648, 740), (836, 928), (1001, 1093)]

    def box(x0, x1, y, color=None, edge=None, dashed=False, dx=0, dy=0):
        ax.add_patch(FancyBboxPatch((x0 + dx, y - 22 + dy), x1 - x0, 44, boxstyle="round,pad=0,rounding_size=8",
                                    facecolor=color or "white", edgecolor=edge or "none",
                                    lw=1.8 if edge else 0, ls=(0, (3, 2)) if dashed else "-"))

    def label(x, y, text, color=GRAY, size=10.5, **kw):
        ax.text(x, y, text, ha="center", va="center", color=color, fontsize=size, **kw)

    ax.text(55, 58, "How each model goes wrong (sketch)", fontsize=15, fontweight="bold", color=INK, va="center")
    rows = {"real calls": 145, "YOLO": 265, "Whale-VAD": 385, "NAVE": 505}
    for name, y in rows.items():
        ax.text(262, y - 2, name, ha="right", va="center", fontsize=13, fontweight="bold",
                color=GRAY if name == "real calls" else INK)

    for i, (x0, x1) in enumerate(calls, start=1):
        box(x0, x1, 145, PALE)
        label((x0 + x1) / 2, 144, str(i), color=INK, size=12, fontweight="bold")

    y = rows["YOLO"]
    for i in (0, 2, 4):                             # missed
        box(*calls[i], y, edge=MID, dashed=True)
        label(sum(calls[i]) / 2, y + 50, "missed")
    for i in (1, 3):                                # found twice, by two overlapping windows
        box(*calls[i], y, MID)
        box(*calls[i], y, "#6B757D", dx=26, dy=-10)
        label(sum(calls[i]) / 2 + 13, y - 53, "found twice")

    y = rows["Whale-VAD"]
    for c in calls:
        box(*c, y, TEAL)
    for x0, x1 in ((776, 811), (1136, 1171)):       # short noise detections
        box(x0, x1, y, TEAL)
        label((x0 + x1) / 2, y + 50, "noise", color=ORANGE)

    y = rows["NAVE"]
    for x0, x1 in (calls[0], (calls[1][0], calls[2][1]), calls[3], calls[4]):
        box(x0, x1, y, ORANGE)
    label(630, y + 50, "2 and 3 merged into one")

    ax.annotate("", (1185, 596), (358, 596), arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.5",
                                                             color=PALE, lw=2.2))
    ax.text(1188, 566, "time", ha="right", va="center", fontsize=10.5, color=GRAY)
    save(fig, "4_error_sketch")


# ---- slide 5 ----

def pitch_window():
    """Z-call pitch at Kerguelen in 2014/15 and 2019/20, with the lower edge of NAVE's listening window."""
    style(13)
    dev, ev = zcall_pitch()
    edges = np.round(np.arange(24.8, 26.651, 0.05), 2)
    lo = round((dev.median() + nave_window()[0]) / 0.05) * 0.05     # window edge, on the 0.05 Hz bin grid
    below = (ev < lo).mean()

    fig, ax = plt.subplots(figsize=(6.24, 3.24))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.18)
    ax.axvspan(lo, 27.5, color=TINT, zorder=0)
    ax.hist(dev, bins=edges, color=TEAL, alpha=0.85, zorder=2, label="2014/15 (dev)")
    ax.hist(ev[ev >= lo], bins=edges, color="#F3B48A", zorder=3, label="2019/20, inside")
    ax.hist(ev[ev < lo], bins=edges, color=ORANGE, zorder=3, label="2019/20, below")
    ax.text(lo + 0.04, 71, "where NAVE listens", fontsize=12, fontweight="bold", color=TEAL, va="top")
    ax.annotate(f"{below * 100:.0f} % of the 2019/20\nZ-calls fall below", (lo - 0.15, 38), (24.82, 67),
                fontsize=12.5, fontweight="bold", color=ORANGE, va="top", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.4, shrinkA=0, relpos=(0.68, 0)))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.89), frameon=False, fontsize=11.5, handlelength=2.2)
    ax.set_xlim(24.8, 26.6)
    ax.set_ylim(0, 75)
    ax.set_xticks(np.arange(25, 26.6, 0.25))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Z-call pitch at Kerguelen (Hz)")
    ax.set_ylabel("number of calls")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    title(ax, "Z-calls inside and outside the window", 15)
    save(fig, "5_pitch_window")


def loudness():
    """Stream graph of the loudness probe: calls found above zero and false alarms below, stacked by model,
    summed over the 112 probe clips and all classes."""
    style(13)
    pr = table("probe_results.parquet")
    lv = pd.concat([pr[pr.probe == "level"], pr[pr.probe == "base"].assign(value=0.0)])
    s = lv.groupby(["value", "model"])[["detected", "false_alarms"]].sum().unstack()
    clips = int(lv.groupby(["value", "model"]).size().max())
    db = s.index.to_numpy()
    xs = np.linspace(db.min(), db.max(), 300)

    def smooth(col):
        return np.clip(PchipInterpolator(db, col.to_numpy())(xs), 0, None)

    gap = 4                                         # white line at zero
    fig, ax = plt.subplots(figsize=(6.24, 3.24))
    fig.subplots_adjust(left=0.13, right=0.84, top=0.85, bottom=0.19)
    top, bottom = np.zeros_like(xs), np.zeros_like(xs)
    for m in MODELS:
        found, fa = smooth(s["detected"][m]), smooth(s["false_alarms"][m])
        ax.fill_between(xs, top, top + found, color=COLOR[m], lw=0)
        ax.fill_between(xs, -gap - bottom, -gap - bottom - fa, lw=0,
                        color={"yolo": GRAY_LIGHT, "whalevad": TEAL_LIGHT, "nave": ORANGE_LIGHT}[m])
        ax.text(xs[-1] + 0.6, top[-1] + found[-1] / 2, NAME[m], va="center", fontsize=12, fontweight="bold",
                color=GRAY if m == "yolo" else COLOR[m], clip_on=False)
        top, bottom = top + found, bottom + fa
    for y in (200, -200, -400):
        ax.axhline(y, color=GRID, lw=0.9, alpha=0.5, zorder=3)
    ax.text(-19.3, 27, "calls found", color="white", fontsize=12.5, fontweight="bold", va="center", zorder=4)
    ax.text(-19.3, -35, "false alarms", color=INK, fontsize=12.5, fontweight="bold", va="center", zorder=4)
    ax.set_xlim(db.min(), db.max())
    ax.set_ylim(-480, 240)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([-400, -200, 0, 200], ["400", "200", "0", "200"])
    ax.set_xticks([-20, -10, 0, 10, 20], ["−20", "−10", "0", "+10", "+20 dB"])
    ax.set_xlabel("whole clip quieter or louder")
    ax.set_ylabel(f"events in {clips} test clips")
    title(ax, "Loudness changes what each model finds, all classes", 15)
    save(fig, "5_loudness")


FIGURES = {"seed_misses": seed_misses, "listening_window": listening_window, "threshold_vs_snr": threshold_vs_snr,
           "error_sketch": error_sketch, "pitch_window": pitch_window, "loudness": loudness}


def main(names: list[str]) -> None:
    for n in names or FIGURES:
        if n not in FIGURES:
            raise SystemExit(f"unknown figure {n!r}, use one of {', '.join(FIGURES)}")
        FIGURES[n]()


if __name__ == "__main__":
    main(sys.argv[1:])
