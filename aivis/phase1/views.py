"""Phase I views (matplotlib), used by notebooks/02_phase1.ipynb and by `python -m aivis.phase1.views`,
which writes every figure to results/phase1/figures/.

Colours: one fixed colour per model everywhere (YOLO blue, Whale-VAD orange, NAVE aqua); the three
pass the colour-blindness checks as a set. Outcomes are coloured by how close they came to a hit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import paths
from . import build, core
from .core import CLASSES, MODELS, TABLES

MODEL_COLOR = {"yolo": "#2a78d6", "whalevad": "#eb6834", "nave": "#1baf7a", "yolo_nms": "#7fb0ea"}
NAME = {"yolo": "YOLO", "whalevad": "Whale-VAD", "nave": "NAVE", "yolo_nms": "YOLO + NMS"}
CLASS_NAME = {"bmabz": "BMABZ", "d": "D", "bp": "BP"}
LABEL_OUT_COLOR = {"hit": "#1baf7a", "fragmented": "#c98500", "merged": "#eda100", "near_miss": "#f5c96b",
                   "wrong_class": "#6250d6", "missed_almost": "#b9b8b2", "missed_silent": "#77766f"}
EVENT_OUT_COLOR = {"tp": "#1baf7a", "duplicate": "#2a78d6", "boundary": "#eda100", "cross_class": "#6250d6",
                   "near_call": "#b9b8b2", "background": "#77766f"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VAL = paths.SITES["validation"]
EVAL = paths.SITES["evaluation"]
FIG_DIR = paths.ROOT / "results" / "phase1" / "figures"


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
    })


def t(name: str) -> pd.DataFrame:
    return pd.read_parquet(TABLES / name)


def models_present(df, col="model"):
    return [m for m in MODELS if m in set(df[col])]


# ------------------------------------------------------------------ idea 1: outcomes

def stacked_share(ax, table: pd.DataFrame, order, colors, models):
    """Horizontal 100 % bars, one per model, segments in outcome order with a 2 px surface gap."""
    for i, m in enumerate(models):
        left = 0.0
        row = table.loc[m] if m in table.index else pd.Series(dtype=float)
        tot = row.sum() or 1
        for o in order:
            v = row.get(o, 0) / tot
            if v > 0:
                ax.barh(i, v, left=left, color=colors[o], edgecolor="#fcfcfb", linewidth=1.5, height=0.72)
                if v >= 0.12:
                    ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                            color="white" if o in ("hit", "wrong_class", "missed_silent", "tp", "duplicate",
                                                    "cross_class", "background") else INK)
            left += v
    ax.set_yticks(range(len(models)), [NAME[m] for m in models])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)


def fig_label_outcomes(variant="loso"):
    lo = t(f"label_outcomes_{variant}.parquet").merge(t("labels_validation.parquet")[["label_id", "dataset", "cls"]], on="label_id")
    models = models_present(lo)
    sites = [s for s in VAL if s in set(lo.dataset)]
    fig, axes = plt.subplots(len(CLASSES), len(sites), figsize=(3.6 * len(sites), 1.0 + 1.1 * len(CLASSES) * len(models) / 3),
                             sharex=True, squeeze=False)
    for i, c in enumerate(CLASSES):
        for j, s in enumerate(sites):
            g = lo[(lo.cls == c) & (lo.dataset == s)]
            tab = g.groupby(["model", "outcome"]).size().unstack(fill_value=0)
            stacked_share(axes[i, j], tab, list(LABEL_OUT_COLOR), LABEL_OUT_COLOR, models)
            n = len(g) // max(len(models), 1)
            axes[i, j].set_title(f"{s} · {CLASS_NAME[c]} (n={n})", fontsize=9)
            if j:
                axes[i, j].set_yticklabels([])
    handles = [mpl.patches.Patch(color=v, label=k.replace("_", " ")) for k, v in LABEL_OUT_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("What happens to each labelled call (LOSO thresholds)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def fig_event_outcomes(variant="loso"):
    eo = t(f"event_outcomes_{variant}.parquet")
    models = models_present(eo)
    sites = [s for s in VAL if s in set(eo.dataset)]
    fig, axes = plt.subplots(len(CLASSES), len(sites), figsize=(3.6 * len(sites), 1.0 + 1.1 * len(CLASSES) * len(models) / 3),
                             sharex=True, squeeze=False)
    for i, c in enumerate(CLASSES):
        for j, s in enumerate(sites):
            g = eo[(eo.cls == c) & (eo.dataset == s)]
            tab = g.groupby(["model", "outcome"]).size().unstack(fill_value=0)
            stacked_share(axes[i, j], tab, list(EVENT_OUT_COLOR), EVENT_OUT_COLOR, models)
            counts = g.groupby("model").size()
            axes[i, j].set_title(f"{s} · {CLASS_NAME[c]} (" + ", ".join(f"{NAME[m][0]}={counts.get(m, 0)}" for m in models) + ")",
                                 fontsize=8)
            if j:
                axes[i, j].set_yticklabels([])
    handles = [mpl.patches.Patch(color=v, label=k.replace("_", " ")) for k, v in EVENT_OUT_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("What each detection is (LOSO thresholds)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def upset(ax_bar, ax_dots, counts: pd.Series, models, color="#52514e", top=None):
    """Simple UpSet: bars = size of each exact combination, dots = which models are in it."""
    counts = counts.sort_values(ascending=False)
    if top:
        counts = counts.head(top)
    x = np.arange(len(counts))
    ax_bar.bar(x, counts.values, color=color, width=0.7)
    for xi, v in zip(x, counts.values):
        ax_bar.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax_bar.set_xticks([])
    ax_bar.grid(axis="x", visible=False)
    for yi, m in enumerate(models):
        inn = [m in k.split("+") for k in counts.index]
        ax_dots.scatter(x, [yi] * len(x), s=40, c=[MODEL_COLOR[m] if v else GRID for v in inn], zorder=3)
    for xi, k in zip(x, counts.index):
        ys = [yi for yi, m in enumerate(models) if m in k.split("+")]
        if len(ys) > 1:
            ax_dots.plot([xi, xi], [min(ys), max(ys)], color=MUTED, lw=1.5, zorder=2)
    ax_dots.set_yticks(range(len(models)), [NAME[m] for m in models])
    ax_dots.set_xticks([])
    ax_dots.set_ylim(len(models) - 0.5, -0.5)
    ax_dots.grid(False)
    for sp in ("left", "bottom"):
        ax_dots.spines[sp].set_visible(False)


def fig_upsets(variant="loso"):
    w = t(f"labels_wide_{variant}.parquet")
    reg = t(f"regions_{variant}.parquet")
    models = [m for m in MODELS if f"hit_{m}" in w]
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1.2], hspace=0.05, wspace=0.35)
    for j, c in enumerate(CLASSES):
        lab = w[w.cls == c].hit_combo.replace({"none": "(no model)"}).value_counts()
        lab = lab.rename(index=lambda k: k if k != "(no model)" else "")
        ab, ad = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        upset(ab, ad, lab, models, color="#1baf7a")
        ab.set_title(f"{CLASS_NAME[c]} labels: hit by", fontsize=9)
        if j:
            ad.set_yticklabels([])
        fa = reg[(reg.cls == c) & (reg.kind == "false_alarm")].combo.value_counts()
        ab2, ad2 = fig.add_subplot(gs[0, 3 + j]), fig.add_subplot(gs[1, 3 + j])
        upset(ab2, ad2, fa, models, color="#77766f")
        ab2.set_title(f"{CLASS_NAME[c]} false alarms", fontsize=9)
        ad2.set_yticklabels([])
    fig.suptitle("Agreement on the dev sites: hits (left, last bar = missed by all) and false alarms (right)",
                 x=0.01, ha="left")
    return fig


# ---------------------------------------------------------------- idea 6 and slicing

def fig_recall_vs(feature="snr_db", bins=None, variant="loso", xlabel="SNR of the label (dB)"):
    w = t(f"labels_wide_{variant}.parquet")
    models = [m for m in MODELS if f"hit_{m}" in w]
    bins = bins if bins is not None else np.arange(-2, 13, 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, CLASSES):
        g = w[w.cls == c].copy()
        g["bin"] = pd.cut(g[feature], bins)
        for m in models:
            r = g.groupby("bin", observed=True)[f"hit_{m}"].agg(["mean", "size"])
            r = r[r["size"] >= 20]
            x = [b.mid for b in r.index]
            ax.plot(x, r["mean"], color=MODEL_COLOR[m], marker="o", ms=4, label=NAME[m])
        n = g.groupby("bin", observed=True).size()
        ax2 = ax.twinx()
        ax2.bar([b.mid for b in n.index], n.values, width=(bins[1] - bins[0]) * 0.9, color=GRID, zorder=0)
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(CLASS_NAME[c])
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("recall (bins with 20+ labels)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Recall by {xlabel.split(' (')[0]} (grey: number of labels)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_fine_labels(variant="loso"):
    w = t(f"labels_wide_{variant}.parquet")
    models = [m for m in MODELS if f"hit_{m}" in w]
    fines = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
    sites = [s for s in VAL if s in set(w.dataset)]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9ec2ec", "#2a78d6", "#0d3f80"])
    for ax, m in zip(axes, models):
        mat = np.full((len(fines), len(sites)), np.nan)
        cnt = np.zeros_like(mat)
        for i, f in enumerate(fines):
            for j, s in enumerate(sites):
                g = w[(w.fine == f) & (w.dataset == s)]
                cnt[i, j] = len(g)
                if len(g) >= 30:
                    mat[i, j] = g[f"hit_{m}"].mean()
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(fines)):
            for j in range(len(sites)):
                txt = "n<30" if np.isnan(mat[i, j]) else f"{mat[i, j]:.0%}\n{int(cnt[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if not np.isnan(mat[i, j]) and mat[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(sites)), sites, rotation=20)
        ax.set_yticks(range(len(fines)), fines)
        ax.set_title(NAME[m])
        ax.grid(False)
    fig.suptitle("Recall per fine label and site (count below; cells with fewer than 30 labels left empty)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_box_style(variant="loso"):
    lo = t(f"label_outcomes_{variant}.parquet").merge(t("labels_validation.parquet")[["label_id", "cls", "duration"]], on="label_id")
    lo = lo[lo.outcome.isin(["hit", "near_miss"])]
    models = models_present(lo)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4), sharey="row")
    for j, c in enumerate(CLASSES):
        for i, (col, lab) in enumerate((("onset_err", "onset error (s), detection minus label"),
                                        ("offset_err", "offset error (s)"))):
            ax = axes[i, j]
            data = [lo[(lo.cls == c) & (lo.model == m)][col].dropna().clip(-6, 6) for m in models]
            new_api = tuple(int(v) for v in mpl.__version__.split(".")[:2]) >= (3, 10)
            kw = {"orientation": "horizontal"} if new_api else {"vert": False}
            parts = ax.violinplot([d if len(d) else [0] for d in data], positions=range(len(models)),
                                  showmedians=True, widths=0.8, **kw)
            for b, m in zip(parts["bodies"], models):
                b.set_facecolor(MODEL_COLOR[m])
                b.set_alpha(0.75)
            for k in ("cbars", "cmins", "cmaxes", "cmedians"):
                if k in parts:
                    parts[k].set_color(INK)
                    parts[k].set_linewidth(1)
            ax.axvline(0, color=MUTED, lw=1)
            ax.set_yticks(range(len(models)), [NAME[m] for m in models])
            ax.set_xlabel(lab)
            if i == 0:
                ax.set_title(CLASS_NAME[c])
            for k, d in enumerate(data):
                if len(d):
                    ax.text(6.2, k, f"med {np.median(d):+.2f}", va="center", fontsize=7, color=MUTED)
    fig.suptitle("Box edges of hits and near-misses (clipped at ±6 s)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- idea 5: ranking

def fig_threshold_curves():
    sw = t("sweep.parquet")
    sw = sw[sw.iou == 0.3]
    th = build.thresholds()
    models = models_present(sw)
    fig, axes = plt.subplots(len(models), 3, figsize=(12, 2.7 * len(models)), sharey=True, squeeze=False)
    site_style = {s: ls for s, ls in zip(VAL, ["-", "--", ":"])}
    for i, m in enumerate(models):
        for j, c in enumerate(CLASSES):
            ax = axes[i, j]
            for s in VAL:
                g = sw[(sw.model == m) & (sw.cls == c) & (sw.site == s)].sort_values("thr")
                f1 = 2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)
                ax.plot(g.thr, f1, color=MODEL_COLOR[m], ls=site_style[s], lw=1.6, label=s)
            ax.axvline(th[m]["insample_iou0.3"][j], color=INK, lw=1, label="in-sample")
            for s in VAL:
                ax.plot([th[m]["loso_iou0.3"][s][j]], [0.02], marker="v", color=MUTED, ms=6)
            d = th[m]["default"][j]
            ax.axvline(d, color=MUTED, lw=1, ls=(0, (1, 2)), label="published/default")
            ax.set_title(f"{NAME[m]} · {CLASS_NAME[c]}", fontsize=9)
            if m == "yolo":
                ax.set_xscale("log")
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel("F1 (IoU 0.3)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("F1 against the threshold per dev site (triangles: LOSO thresholds; solid: in-sample; dotted: default)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_bump():
    r = t("ranking.parquet")
    r = r[r.cls.isna() & (r.dataset == "joined")]
    variants = []
    for th in ("default", "insample", "loso"):
        for iou in (0.1, 0.3, 0.5):
            for mac in ("macro_A", "macro_B"):
                variants.append((th, iou, mac))
    models = [m for m in ["yolo", "yolo_nms", "whalevad", "nave"] if m in set(r.model)]
    vals = pd.DataFrame({m: [r[(r.model == m) & (r.thresholds == th) & (r.iou == iou)][mac].mean()
                             for th, iou, mac in variants] for m in models})
    ranks = vals.rank(axis=1, ascending=False)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    x = np.arange(len(variants))
    for m in models:
        a1.plot(x, ranks[m], color=MODEL_COLOR[m], marker="o", ms=6, label=NAME[m])
        a2.plot(x, vals[m], color=MODEL_COLOR[m], marker="o", ms=4, label=NAME[m])
    a1.set_yticks(range(1, len(models) + 1))
    a1.invert_yaxis()
    a1.set_ylabel("rank")
    a1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    a2.set_ylabel("score (dev sites joined)")
    a2.set_xticks(x, [f"{th}\nIoU {iou}\n{mac[-1]}" for th, iou, mac in variants], fontsize=7)
    for k in (6, 12):
        for ax in (a1, a2):
            ax.axvline(k - 0.5, color=MUTED, lw=0.8)
    fig.suptitle("Does the ranking survive the scoring choices? (thresholds x IoU x macro A/B)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- idea 2: agreement

def fig_agreement():
    ag = t("agreement.parquet")
    models = models_present(ag)
    sites = [s for s in VAL + EVAL if s in set(ag.dataset)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    w = 0.8 / len(models)
    for j, c in enumerate(CLASSES):
        for i, (col, lab) in enumerate((("per_hour", "detections per hour"),
                                        ("confirmed_share", "share confirmed by another model"),
                                        ("est_precision", "estimated precision (from dev combos)"))):
            ax = axes[i, j]
            for k, m in enumerate(models):
                g = ag[(ag.model == m) & (ag.cls == c)].set_index("dataset").reindex(sites)
                ax.bar(np.arange(len(sites)) + (k - (len(models) - 1) / 2) * w, g[col], width=w * 0.92,
                       color=MODEL_COLOR[m], label=NAME[m])
                if col == "est_precision":
                    ax.scatter(np.arange(len(sites)) + (k - (len(models) - 1) / 2) * w, g["dev_precision"],
                               marker="_", s=60, color=INK, zorder=3)
            ax.axvline(len([s for s in sites if s in VAL]) - 0.5, color=MUTED, lw=1)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 0:
                ax.set_title(CLASS_NAME[c])
            if i == 2:
                ax.set_xticks(range(len(sites)), sites, rotation=35, ha="right")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Dev sites (left of the line) vs new sites (right). Black ticks: real precision on dev",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_timeline(cls="bmabz"):
    reg = t("regions_all.parquet")
    reg = reg[reg.cls == cls]
    models = [m for m in MODELS if m in reg]
    sites = [s for s in VAL + EVAL if s in set(reg.dataset)]
    fig, axes = plt.subplots(len(sites), 1, figsize=(13, 1.1 * len(sites) + 0.6), squeeze=False)
    for ax, s in zip(axes[:, 0], sites):
        g = reg[reg.dataset == s]
        days = pd.date_range(g.day.min(), g.day.max(), freq="D")
        for k, m in enumerate(models):
            n = g[g[m]].groupby("day").size().reindex(days, fill_value=0)
            ax.bar(days, n.values, bottom=0, width=0.9, color=MODEL_COLOR[m], alpha=0.55 if k else 0.8)
        allthree = g[g.combo == "+".join(models)].groupby("day").size().reindex(days, fill_value=0)
        ax.plot(days, allthree.values, color=INK, lw=1)
        ax.set_ylabel(s, rotation=0, ha="right", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{CLASS_NAME[cls]} detections per day and model over each deployment (black: all three agree)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- probes

PROBE_X = {"level": "gain (dB)", "shift": "frequency shift (Hz)", "context": "kept context (+-s)",
           "snr": "SNR of the pasted call (dB)", "position": "call centre in the 120 s clip (s)"}


def probe_table():
    pr = t("probe_results.parquet")
    base = pr[pr.probe == "base"].set_index(["call_id", "model"]).peak.rename("base_peak")
    ref = pr[(pr.probe == "shift") & (pr.value == 0)].set_index(["call_id", "model"]).peak.rename("pasted_peak")
    return pr.join(base, on=["call_id", "model"]).join(ref, on=["call_id", "model"])


def fig_probe_curves(probe: str, y="detected"):
    pr = probe_table()
    g = pr[pr.probe == probe]
    if probe == "level":
        g = pd.concat([g, pr[pr.probe == "base"].assign(probe="level", value=0.0)])
    models = models_present(g)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for m in models:
            r = g[(g.cls == c) & (g.model == m)].groupby("value")[y].mean()
            ax.plot(r.index, r.values, color=MODEL_COLOR[m], marker="o", ms=4, label=NAME[m])
        ax.set_title(f"{CLASS_NAME[c]} (n={g[g.cls == c].call_id.nunique()} calls)")
        ax.set_xlabel(PROBE_X.get(probe, "value"))
        if probe == "position":
            for b in (30, 45, 60, 75):
                ax.axvline(b, color=MODEL_COLOR["yolo"], lw=0.8, ls=":")
            ax.axvline(58, color=MODEL_COLOR["nave"], lw=0.8, ls=":")
            ax.axvline(60, color=MODEL_COLOR["nave"], lw=0.8, ls=":")
    axes[0].set_ylabel("detection rate" if y == "detected" else "mean peak score")
    if y == "detected":
        axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Probe: {probe}" + (" (dotted: YOLO window borders, NAVE tile edges)" if probe == "position" else ""),
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def snr50(g: pd.DataFrame) -> float:
    """SNR at which half the calls are found (logistic fit; nan if it does not cross 50 %)."""
    from scipy.optimize import curve_fit
    x, y = g.value.to_numpy(float), g.detected.to_numpy(float)
    if y.mean() in (0, 1):
        return np.nan
    try:
        (a, b), _ = curve_fit(lambda v, a, b: 1 / (1 + np.exp(-(v - a) / b)), x, y, p0=(0, 2), maxfev=4000,
                              bounds=([-30, 0.2], [30, 20]))
        return float(a)
    except Exception:  # noqa: BLE001
        return np.nan


def snr50_table() -> pd.DataFrame:
    pr = t("probe_results.parquet")
    g = pr[pr.probe == "snr"]
    rows = [{"model": m, "cls": c, "bg_site": b, "snr50": snr50(x)} for (m, c, b), x in g.groupby(["model", "cls", "bg_site"])]
    return pd.DataFrame(rows).pivot_table(index=["cls", "bg_site"], columns="model", values="snr50")


def fig_transient():
    pr = t("probe_results.parquet")
    g = pr[pr.probe == "transient"]
    base = pr[pr.probe == "base"].groupby(["model", "cls"]).detected.mean()
    models = models_present(g)
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 2.4 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, c in enumerate(CLASSES):
            mat = g[(g.model == m) & (g.cls == c)].pivot_table(index="value", columns="offset", values="detected") \
                - base.get((m, c), np.nan)
            ax = axes[i, j]
            ax.imshow(mat.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
            for (a, b), v in np.ndenumerate(mat.values):
                ax.text(b, a, f"{v:+.0%}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(mat.shape[1]), [f"{x:+.0f}s" for x in mat.columns])
            ax.set_yticks(range(mat.shape[0]), [f"+{y:.0f} dB" for y in mat.index])
            ax.set_title(f"{NAME[m]} · {CLASS_NAME[c]}", fontsize=9)
            ax.grid(False)
    fig.suptitle("Distant transient: change in detection rate vs the unchanged clip (blue: more, red: fewer)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_bgswap():
    pr = t("probe_results.parquet")
    g = pr[pr.probe == "bgswap"]
    models = models_present(g)
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3), sharey=True)
    axes = np.atleast_1d(axes)
    sites = [s for s in VAL + EVAL if s in set(g.bg_site)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9ec2ec", "#2a78d6", "#0d3f80"])
    for ax, m in zip(axes, models):
        mat = g[g.model == m].pivot_table(index="cls", columns="bg_site", values="detected").reindex(index=CLASSES, columns=sites)
        ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (a, b), v in np.ndenumerate(mat.values):
            ax.text(b, a, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(sites)), sites, rotation=35, ha="right")
        ax.set_yticks(range(3), [CLASS_NAME[c] for c in CLASSES])
        ax.set_title(NAME[m])
        ax.grid(False)
    fig.suptitle("The same dev calls pasted into background from every site: detection rate", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- comparison browser

def show_region(dataset: str, filename: str, t_center: float, span: float = 60.0, variant: str = "loso",
                speed: int = 10, audio: bool = False):
    """One spectrogram with the labels and YOLO's boxes in the time-frequency plane, and the Whale-VAD and
    NAVE score curves with their thresholds below. t_center in seconds from the file start."""
    import soundfile as sf
    split = core.split_of(dataset)
    path = paths.audio_dir(split) / dataset / filename
    sr = 250
    t0 = max(0.0, t_center - span / 2)
    with sf.SoundFile(str(path)) as f:
        f.seek(int(t0 * sr))
        x = f.read(int(span * sr), dtype="float32", always_2d=True)[:, 0]
    fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    ax = axes[0]
    ax.specgram(x, NFFT=256, Fs=sr, noverlap=246, cmap="Greys", xextent=(t0, t0 + len(x) / sr))
    ax.set_ylim(0, 125)
    ax.set_ylabel("Hz")
    ax.grid(False)
    fe = core.epoch(filename)
    if split == "validation":
        gt = core.load_labels("validation")
        g = gt[(gt.dataset == dataset) & (gt.t1 > fe + t0) & (gt.t0 < fe + t0 + span)]
        for r in g.itertuples():
            ax.add_patch(mpl.patches.Rectangle((r.t0 - fe, r.low_frequency), r.duration, r.high_frequency - r.low_frequency,
                                               fill=False, ec=INK, lw=1.6))
            ax.text(r.t0 - fe, r.high_frequency + 1, r.fine, fontsize=7)
    th = build.thresholds()
    if (core.ROOT / "yolo" / dataset / "boxes_raw.parquet").exists():
        ev = core.yolo_events(core.yolo_boxes(dataset))
        ev = ev[(ev.filename == filename) & (ev.end_s > t0) & (ev.start_s < t0 + span)]
        thr = dict(zip(CLASSES, build.thr_for("yolo", dataset, variant)))
        ev = ev[ev.conf >= ev.cls.map(thr)]
        for r in ev.itertuples():
            ax.add_patch(mpl.patches.Rectangle((r.start_s, r.low_frequency), r.end_s - r.start_s,
                                               r.high_frequency - r.low_frequency, fill=False, ec=MODEL_COLOR["yolo"],
                                               lw=1.2, ls="--"))
            ax.text(r.start_s, r.low_frequency - 4, f"{r.cls} {r.conf:.2f}", fontsize=6, color=MODEL_COLOR["yolo"])
    ax.set_title(f"{dataset} / {filename}  (black: labels, dashed blue: YOLO boxes above its threshold)", fontsize=9)
    for k, (m, ax) in enumerate(zip(("whalevad", "nave"), axes[1:3])):
        f = core.frame_probs_dir(m, dataset) / filename.replace(".wav", ".npz")
        if f.exists():
            with np.load(f) as z:
                sm = core.smooth(z["probs"].astype(np.float32))
            i0, i1 = int(t0 / core.HOP_S), int((t0 + span) / core.HOP_S)
            tt = np.arange(i0, min(i1, len(sm))) * core.HOP_S
            thr = build.thr_for(m, dataset, variant)
            for ci, (c, ls) in enumerate(zip(CLASSES, ["-", "--", ":"])):
                ax.plot(tt, sm[i0:i1, ci], color=MODEL_COLOR[m], ls=ls, lw=1.4, label=CLASS_NAME[c])
                ax.axhline(thr[ci], color=MODEL_COLOR[m], ls=ls, lw=0.7, alpha=0.6)
            ax.set_ylim(0, 1)
            ax.set_ylabel(NAME[m])
            ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax = axes[3]
    if (core.ROOT / "yolo" / dataset / "boxes_raw.parquet").exists():
        ev = core.yolo_events(core.yolo_boxes(dataset))
        ev = ev[(ev.filename == filename) & (ev.end_s > t0) & (ev.start_s < t0 + span)]
        for ci, c in enumerate(CLASSES):
            e = ev[ev.cls == c]
            ax.hlines(e.conf, e.start_s, e.end_s, color=MODEL_COLOR["yolo"], lw=2, alpha=0.4 + 0.2 * ci)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1)
        ax.set_ylabel("YOLO conf")
    ax.set_xlabel("seconds from file start")
    fig.tight_layout()
    if audio:
        from IPython.display import Audio, display
        display(Audio(x, rate=sr * speed))
    return fig


def browser(variant: str = "loso"):
    """ipywidgets browser over label outcomes and false-alarm regions."""
    import ipywidgets as W
    from IPython.display import display
    lo = t(f"label_outcomes_{variant}.parquet").merge(t("labels_validation.parquet"), on="label_id")
    reg = t(f"regions_{variant}.parquet")
    src = W.ToggleButtons(options=["labels", "false alarms"], value="labels")
    model = W.Dropdown(options=[(NAME[m], m) for m in models_present(lo)], description="model")
    outcome = W.Dropdown(options=list(LABEL_OUT_COLOR), description="outcome")
    combo = W.Dropdown(options=sorted(reg.combo.unique()), description="fired")
    cls = W.Dropdown(options=[("all", None)] + [(CLASS_NAME[c], c) for c in CLASSES], description="class")
    site = W.Dropdown(options=[("all", None)] + [(s, s) for s in VAL], description="site")
    idx = W.IntSlider(0, 0, 0, description="example")
    span = W.IntSlider(60, 20, 300, step=10, description="span (s)")
    play = W.Checkbox(False, description="audio x10")
    out = W.Output()

    def rows():
        if src.value == "labels":
            g = lo[(lo.model == model.value) & (lo.outcome == outcome.value)]
        else:
            g = reg[(reg.combo == combo.value) & (reg.kind == "false_alarm")]
        if cls.value:
            g = g[g.cls == cls.value]
        if site.value:
            g = g[g.dataset == site.value]
        return g.reset_index(drop=True)

    def draw(*_):
        g = rows()
        idx.max = max(len(g) - 1, 0)
        out.clear_output(wait=True)
        with out:
            if g.empty:
                print("nothing here")
                return
            r = g.iloc[min(idx.value, len(g) - 1)]
            fe = core.epoch(r.filename)
            print(f"{len(g)} examples. {r.dataset} {r.filename} {r.cls} at {r.t0 - fe:.1f} s"
                  + (f", SNR {r.snr_db:.1f} dB, best IoU {r.best_iou:.2f}" if src.value == "labels" else ""))
            show_region(r.dataset, r.filename, (r.t0 + r.t1) / 2 - fe, span.value, variant, audio=play.value)
            plt.show()

    for w in (src, model, outcome, combo, cls, site, idx, span, play):
        w.observe(draw, "value")
    display(W.VBox([W.HBox([src, cls, site]), W.HBox([model, outcome, combo]), W.HBox([idx, span, play]), out]))
    draw()


# ------------------------------------------------------------------------- all figures

FIGURES = {
    "1a_label_outcomes": fig_label_outcomes, "1b_event_outcomes": fig_event_outcomes, "1c_upsets": fig_upsets,
    "6a_recall_vs_snr": fig_recall_vs, "6b_recall_vs_duration": lambda: fig_recall_vs("duration", np.arange(0, 21, 2), xlabel="label duration (s)"),
    "6c_fine_labels": fig_fine_labels, "6d_box_style": fig_box_style,
    "5a_threshold_curves": fig_threshold_curves, "5b_bump": fig_bump,
    "2a_agreement": fig_agreement, "2b_timeline_bmabz": lambda: fig_timeline("bmabz"),
    "2c_timeline_d": lambda: fig_timeline("d"), "2d_timeline_bp": lambda: fig_timeline("bp"),
    "4a_level": lambda: fig_probe_curves("level"), "4b_shift": lambda: fig_probe_curves("shift"),
    "4c_context": lambda: fig_probe_curves("context"), "4d_position": lambda: fig_probe_curves("position"),
    "4e_transient": fig_transient, "4f_bgswap": fig_bgswap, "3a_snr": lambda: fig_probe_curves("snr"),
    "4g_level_peak": lambda: fig_probe_curves("level", y="peak"),
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+")
    a = p.parse_args(argv)
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if a.only and name not in a.only:
            continue
        try:
            fig = fn()
            fig.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[views] {name}")
        except FileNotFoundError as e:
            print(f"[views] {name}: skipped ({Path(str(e.filename or e)).name} missing)")
        except Exception as e:  # noqa: BLE001
            print(f"[views] {name}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
