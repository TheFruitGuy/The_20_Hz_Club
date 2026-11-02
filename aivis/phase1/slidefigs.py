"""Figures for the midpoint slides, in the deck palette.

    python -m aivis.phase1.slidefigs            # all figures
    python -m aivis.phase1.slidefigs drift      # one figure

Writes results/phase1/slides/<name>.png from the Phase I tables (outputs/phase1/tables).

    seed_misses   short deck slide 2: how many of the 5 NAVE seeds find a call, against chance
    error_types   long deck: what happens to each labelled call and each detection, per model and class
    drift         short deck slide 3: Z-call pitch per year, and BMABZ recall when dev audio is shifted by -0.6 Hz
    loudness      short deck slide 4: detection rate at 0 dB and +20 dB, per model and class
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .. import paths
from .core import TABLES

OUT = paths.ROOT / "results" / "phase1" / "slides"

# deck palette
TEAL, ORANGE, INK, GRAY, LIGHT, MID, GRID = "#026E97", "#E8702A", "#1F2327", "#5B6770", "#F1F3F5", "#8A949C", "#E3E7EA"
ORANGE_TINT = "#FDF0E7"
MODELS = ["yolo", "whalevad", "nave"]
CLASSES = ["bmabz", "d", "bp"]
NAME = {"yolo": "YOLO", "whalevad": "Whale-VAD", "nave": "NAVE"}
CLASS_NAME = {"bmabz": "BMABZ", "d": "D", "bp": "BP"}
COLOR = {"yolo": MID, "whalevad": TEAL, "nave": ORANGE}
MARKER = {"yolo": "s", "whalevad": "^", "nave": "o"}
SEEDS = ["hit_s0", "hit_s1", "hit_s42", "hit_s666", "hit_s1337"]
FONT = ["Arial", "Liberation Sans", "DejaVu Sans"]


def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(TABLES / name)


def style(size: int, **extra) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": size,
                         "axes.edgecolor": "#C9D0D5", "axes.labelcolor": GRAY, "xtick.color": GRAY,
                         "ytick.color": GRAY, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", **extra})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("[slidefigs]", name)


def seed_misses():
    """Share of labelled calls found by 0 to 5 NAVE seeds, against seeds that fail independently."""
    style(15)
    s = table("seed_agree_labels.parquet")
    p = s[SEEDS].mean().to_numpy()
    chance = np.array([1.0])
    for q in p:                                    # Poisson binomial: seeds fail independently
        chance = np.convolve(chance, [1 - q, q])
    obs = s.n_seeds_hit.value_counts(normalize=True).sort_index().reindex(range(6), fill_value=0).to_numpy()
    fig = plt.figure(figsize=(16.64, 7.0))         # same shape as the slide body
    ax = fig.add_axes([0.07, 0.14, 0.9, 0.8])
    k = np.arange(6)
    ax.bar(k, chance, width=0.72, color="white", edgecolor=MID, lw=2, ls=(0, (4, 3)),
           label="if the 5 seeds failed independently")
    ax.bar(k, obs, width=0.5, color=[ORANGE] + [TEAL] * 5)
    ax.bar([0], [0], color=TEAL, label="what the 5 seeds actually do")
    for i in k:
        ax.text(i, obs[i] + 0.012, f"{obs[i]*100:.0f} %", ha="center", fontsize=18, fontweight="bold",
                color=ORANGE if i == 0 else INK)
        ax.text(i + 0.30, chance[i] + 0.012, f"{chance[i]*100:.1f} %", ha="left", fontsize=13, color=MID)
    ax.annotate(f"{obs[0]/chance[0]:.0f}× more often than chance", (0, obs[0]), (0.55, 0.43), fontsize=20,
                color=ORANGE, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5))
    ax.set_xticks(k, [f"{i}" for i in k], fontsize=17)
    ax.set_xlabel("number of NAVE seeds (of 5) that find the call", fontsize=16)
    ax.set_yticks([0, .1, .2, .3, .4], ["0 %", "10 %", "20 %", "30 %", "40 %"])
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=False, fontsize=15, loc="upper center", ncol=2)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "seed_misses")


LABEL_OUTCOME = {"hit": "#026E97", "fragmented": "#2E86AB", "merged": "#5FA3C4", "near_miss": "#A9CFE0",
                 "wrong_class": "#E8702A", "missed_almost": "#C9D0D5", "missed_silent": "#5B6770"}
EVENT_OUTCOME = {"tp": "#026E97", "duplicate": "#E8702A", "boundary": "#7FB8D2", "cross_class": "#F3B48A",
                 "near_call": "#C9D0D5", "background": "#5B6770"}
WHITE_TEXT = {"hit", "fragmented", "wrong_class", "missed_silent", "tp", "duplicate", "background"}


def _stacked(ax, tab, colors):
    for i, m in enumerate(MODELS):
        row = tab.loc[m] if m in tab.index else pd.Series(dtype=float)
        tot, left = row.sum() or 1, 0.0
        for o in colors:
            v = row.get(o, 0) / tot
            if v > 0:
                ax.barh(i, v, left=left, color=colors[o], edgecolor="white", linewidth=1.5, height=0.74)
                if v >= 0.12:
                    ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=12, fontweight="bold",
                            color="white" if o in WHITE_TEXT else INK)
            left += v
    ax.set_yticks(range(3), [NAME[m] for m in MODELS], fontsize=13)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.tick_params(length=0)


def error_types():
    """Outcome of every labelled call (top) and every detection (bottom), dev sites, LOSO thresholds."""
    style(12, **{"ytick.color": INK})
    eo = table("event_outcomes_loso.parquet")
    eo = eo[eo.dataset.isin(["casey2017", "kerguelen2014", "kerguelen2015"])]
    lo = table("label_outcomes_loso.parquet").merge(table("labels_validation.parquet")[["label_id", "cls"]],
                                                     on="label_id")
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.3), sharex=True)
    for j, c in enumerate(CLASSES):
        _stacked(axes[0, j], lo[lo.cls == c].groupby(["model", "outcome"]).size().unstack(fill_value=0), LABEL_OUTCOME)
        axes[0, j].set_title(f"{CLASS_NAME[c]}: labelled calls", fontsize=14, fontweight="bold", color=INK, loc="left")
        _stacked(axes[1, j], eo[eo.cls == c].groupby(["model", "outcome"]).size().unstack(fill_value=0), EVENT_OUTCOME)
        axes[1, j].set_title(f"{CLASS_NAME[c]}: detections", fontsize=14, fontweight="bold", color=INK, loc="left")
        if j:
            axes[0, j].set_yticklabels([])
            axes[1, j].set_yticklabels([])
    h1 = [mpl.patches.Patch(color=v, label=k.replace("_", " ")) for k, v in LABEL_OUTCOME.items()]
    h2 = [mpl.patches.Patch(color=v, label=k.replace("_", " ")) for k, v in EVENT_OUTCOME.items()]
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(handles=h1, loc="lower left", ncol=7, bbox_to_anchor=(0.085, 0.06), fontsize=11, frameon=False,
               handlelength=1.2, columnspacing=1.1)
    fig.legend(handles=h2, loc="lower left", ncol=6, bbox_to_anchor=(0.085, 0.0), fontsize=11, frameon=False,
               handlelength=1.2, columnspacing=1.1)
    fig.text(0.012, 0.1, "calls", fontsize=11, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.04, "detections", fontsize=11, fontweight="bold", color=INK, va="center")
    save(fig, "error_types")


def _line_style():
    style(13, **{"axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.titleweight": "bold",
                 "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlesize": 15})


def drift():
    """Left: Z-call peak frequency per year at Kerguelen. Right: BMABZ recall, original against shifted by -0.6 Hz."""
    _line_style()
    zp = table("zcall_peak_by_year.parquet")
    med = zp.groupby("year").hz.median()
    q = zp.groupby("year").hz.quantile([0.25, 0.75]).unstack()
    f1 = table("f1_dev_shift.parquet")
    f1 = f1[f1.cls == "bmabz"].groupby(["model", "condition"])[["tp", "fn"]].sum()
    rec = (f1.tp / (f1.tp + f1.fn)).unstack()

    fig, (a, c) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for yr in med.index:
        col = MID if yr in (2019, 2020) else TEAL
        a.plot([yr, yr], [q.loc[yr, 0.25], q.loc[yr, 0.75]], color=col, lw=4, solid_capstyle="round")
        a.scatter([yr], [med[yr]], color=col, s=110, zorder=3)
    for yr, dx, dy in ((2005, 0.7, 0), (2014, -0.7, -0.22), (2015, 0.7, 0.12), (2019, -0.7, -0.15), (2020, 0.7, 0)):
        a.text(yr + dx, med[yr] + dy, f"{med[yr]:.1f} Hz", va="center", ha="left" if dx > 0 else "right",
               fontsize=15, color=INK, fontweight="bold")
    a.axvspan(2013.3, 2015.7, color=LIGHT, zorder=0)
    a.text(2014.5, 25.12, "dev sites", ha="center", fontsize=12, color=GRAY)
    a.text(2019.5, 25.12, "eval sites", ha="center", fontsize=12, color=GRAY)
    a.set_ylim(25.0, 27.6)
    a.set_xlim(2003, 2022.5)
    a.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    a.set_ylabel("Z-call peak frequency (Hz)")
    a.grid(axis="x", visible=False)
    a.set_title("Z-calls get lower every year")

    for i, m in enumerate(MODELS[::-1]):
        o, s = rec.loc[m, "original"], rec.loc[m, "shift -0.6 Hz"]
        c.plot([s, o], [i, i], color=GRID, lw=10, solid_capstyle="round", zorder=1)
        c.scatter([o], [i], s=260, color=COLOR[m], zorder=3)
        c.scatter([s], [i], s=260, facecolor="white", edgecolor=COLOR[m], lw=3, zorder=3)
        c.text(o + 0.02, i, f"{o:.2f}", va="center", fontsize=14, color=INK)
        c.text(s - 0.02, i, f"{s:.2f}", va="center", ha="right", fontsize=14, color=INK)
        c.text((o + s) / 2, i + 0.28, f"−{(o - s)*100:.0f} pp", ha="center", fontsize=13, color=ORANGE,
               fontweight="bold")
    c.set_yticks(range(3), [NAME[m] for m in MODELS[::-1]], fontsize=15)
    c.set_ylim(-0.6, 2.7)
    c.set_xlim(0.05, 0.65)
    c.set_xlabel("BMABZ recall, Kerguelen 2014/15  (filled: original, open: shifted)")
    c.set_title("Shift the dev audio to 2019/20 pitch")
    c.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "drift")


def loudness():
    """Detection rate of the probe calls at 0 dB and with the whole clip 20 dB louder."""
    _line_style()
    pr = table("probe_results.parquet")
    lv = pd.concat([pr[pr.probe == "level"], pr[pr.probe == "base"].assign(value=0.0)])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for m in MODELS:
            r = lv[(lv.cls == c) & (lv.model == m)].groupby("value").detected.mean()
            y0, y1 = r.loc[0.0], r.loc[20.0]
            ax.plot([0, 1], [y0, y1], color=COLOR[m], lw=3, marker=MARKER[m], ms=9)
            ax.text(1.06, y1, f"{NAME[m]} {(y1 - y0)*100:+.0f} pp", va="center", fontsize=12,
                    color=COLOR[m] if m != "yolo" else GRAY, fontweight="bold")
        ax.set_xticks([0, 1], ["0 dB", "+20 dB"])
        ax.set_xlim(-0.15, 1.75)
        ax.set_title(CLASS_NAME[c])
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("detection rate")
    fig.tight_layout()
    save(fig, "loudness")


FIGURES = {"seed_misses": seed_misses, "error_types": error_types, "drift": drift, "loudness": loudness}


def main(names: list[str]) -> None:
    for n in names or FIGURES:
        if n not in FIGURES:
            raise SystemExit(f"unknown figure {n!r}, use one of {', '.join(FIGURES)}")
        FIGURES[n]()


if __name__ == "__main__":
    main(sys.argv[1:])
