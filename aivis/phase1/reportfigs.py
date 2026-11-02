"""Summary figures for the Phase I report: python -m aivis.phase1.reportfigs -> results/phase1/report/."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from .. import paths
from . import core, views as V
from .core import CLASSES, MODELS, TABLES

OUT = paths.ROOT / "results" / "phase1" / "report"
C, N, INK, MUTED, GRID = V.MODEL_COLOR, V.NAME, V.INK, V.MUTED, V.GRID


def t(n):
    return pd.read_parquet(TABLES / n)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[reportfigs]", name)


def title(fig, main, sub=None):
    """Title and subtitle above the axes (placed in points, so they never overlap whatever the figure height)."""
    h_pt = fig.get_figheight() * 72
    lines = sub.count("\n") + 1 if sub else 0
    y_main = 1.0 + (6 + 13 * lines) / h_pt
    fig.text(0.01, y_main, main, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 1.0 + 4 / h_pt, sub, ha="left", va="bottom", fontsize=9, color=MUTED)


# 1 headline scores with intervals
def fig_scores():
    b = t("bootstrap_macro_b.parquet")
    b = b[(b.site == "joined") & b.what.isin(MODELS)].set_index("what")
    r = t("ranking.parquet")
    nms = r[(r.model == "yolo_nms") & (r.thresholds == "loso") & (r.iou == 0.3) & (r.dataset == "joined") & r.cls.isna()].macro_B.iloc[0]
    e2 = t("e2_macro.parquet")
    nave_c = e2[(e2.model == "nave") & (e2.dataset == "joined") & (e2.condition.str.contains("retuned"))].macro_B.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 2.9))
    rows = [("yolo", "YOLO baseline"), ("whalevad", "Whale-VAD"), ("nave", "NAVE")]
    for i, (m, lab) in enumerate(rows):
        p, lo, hi = b.loc[m, ["point", "lo", "hi"]]
        ax.plot([lo, hi], [i, i], color=C[m], lw=3, solid_capstyle="round")
        ax.scatter([p], [i], s=70, color=C[m], zorder=3)
        ax.text(hi + 0.01, i, f"{p:.2f}", va="center", fontsize=9, color=INK)
    ax.scatter([nms], [0], s=70, facecolor="white", edgecolor=C["yolo"], lw=2, zorder=3)
    ax.annotate(f"duplicates merged: {nms:.2f}", (nms, 0), (nms + 0.005, -0.42), fontsize=8, color=MUTED)
    ax.scatter([nave_c], [2], s=70, facecolor="white", edgecolor=C["nave"], lw=2, zorder=3)
    ax.annotate(f"centre stitching: {nave_c:.2f}", (nave_c, 2), (nave_c - 0.005, 2.32), fontsize=8, color=MUTED, ha="right")
    ax.set_yticks(range(3), [x[1] for x in rows])
    ax.set_ylim(-0.7, 2.6)
    ax.set_xlim(0.25, 0.62)
    ax.set_xlabel("macro B on the three dev sites (LOSO thresholds, IoU 0.3)")
    ax.grid(axis="y", visible=False)
    title(fig, "NAVE leads clearly; YOLO and Whale-VAD are not separable",
          "Dots: score, bars: 95 % bootstrap interval over recordings. Open circles: the same model with a decoding fix.")
    fig.tight_layout()
    save(fig, "01_scores")


# 2 kinds of errors, pooled over the dev sites
def fig_errors():
    eo = t("event_outcomes_loso.parquet")
    eo = eo[eo.dataset.isin(paths.SITES["validation"])]
    lo = t("label_outcomes_loso.parquet").merge(t("labels_validation.parquet")[["label_id", "cls"]], on="label_id")
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), sharex=True)
    for j, c in enumerate(CLASSES):
        tab = lo[lo.cls == c].groupby(["model", "outcome"]).size().unstack(fill_value=0)
        V.stacked_share(axes[0, j], tab, list(V.LABEL_OUT_COLOR), V.LABEL_OUT_COLOR, list(MODELS))
        axes[0, j].set_title(f"{V.CLASS_NAME[c]}: labelled calls", fontsize=10)
        tab = eo[eo.cls == c].groupby(["model", "outcome"]).size().unstack(fill_value=0)
        V.stacked_share(axes[1, j], tab, list(V.EVENT_OUT_COLOR), V.EVENT_OUT_COLOR, list(MODELS))
        axes[1, j].set_title(f"{V.CLASS_NAME[c]}: detections", fontsize=10)
        if j:
            axes[0, j].set_yticklabels([])
            axes[1, j].set_yticklabels([])
    h1 = [mpl.patches.Patch(color=v, label=k.replace("_", " ")) for k, v in V.LABEL_OUT_COLOR.items()]
    h2 = [mpl.patches.Patch(color=v, label=k.replace("_", " ")) for k, v in V.EVENT_OUT_COLOR.items()]
    fig.legend(handles=h1, loc="lower left", ncol=7, bbox_to_anchor=(0.0, -0.06), fontsize=8, title="calls", title_fontsize=8)
    fig.legend(handles=h2, loc="lower left", ncol=6, bbox_to_anchor=(0.0, -0.15), fontsize=8, title="detections", title_fontsize=8)
    title(fig, "Each model fails differently: YOLO duplicates, NAVE merges BP pulses, Whale-VAD misses D",
          "Top: what happens to each labelled call. Bottom: what each detection is. Three dev sites pooled.")
    fig.tight_layout()
    save(fig, "02_errors")


# 3 frequency drift
def zcall_peaks():
    def peak(split, site, fn, s, e):
        with sf.SoundFile(str(paths.audio_dir(split) / site / fn)) as f:
            f.seek(max(0, int(s * 250)))
            x = f.read(int((e - s) * 250), dtype="float32")
        if len(x) < 250:
            return np.nan
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=16384)) ** 2
        fr = np.fft.rfftfreq(16384, 1 / 250)
        sel = (fr >= 23) & (fr <= 29)
        return fr[sel][np.argmax(X[sel])]
    rows = []
    tr = core.load_labels("train", ["kerguelen2005"])
    va = core.load_labels("validation", ["kerguelen2014", "kerguelen2015"])
    for split, g in (("train", tr), ("validation", va)):
        g = g[g.fine.isin(["bma", "bmz"])]
        for site, gs in g.groupby("dataset"):
            for r in gs.sample(min(150, len(gs)), random_state=1).itertuples():
                rows.append((int(site[-4:]), "labels", peak(split, site, r.filename, r.start_s, r.end_s)))
    reg = t("regions_all.parquet")
    for site in ("kerguelen2019", "kerguelen2020"):
        g = reg[(reg.dataset == site) & (reg.cls == "bmabz") & (reg.combo == "yolo+whalevad+nave")]
        for r in g.sample(min(150, len(g)), random_state=1).itertuples():
            fe = core.epoch(r.filename)
            rows.append((int(site[-4:]), "detections", peak("evaluation", site, r.filename, r.t0 - fe, r.t1 - fe)))
    return pd.DataFrame(rows, columns=["year", "source", "hz"])


def fig_drift():
    cache = TABLES / "zcall_peak_by_year.parquet"
    zp = pd.read_parquet(cache) if cache.exists() else zcall_peaks()
    zp.to_parquet(cache)
    pr = t("probe_results.parquet")
    sh = pr[(pr.probe == "shift") & (pr.cls == "bmabz")].groupby(["model", "value"]).detected.mean().unstack(0)
    f1 = t("f1_dev_shift.parquet")
    f1 = f1[f1.cls == "bmabz"].groupby(["model", "condition"])[["tp", "fn"]].sum()
    rec = (f1.tp / (f1.tp + f1.fn)).unstack()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    med = zp.groupby("year").hz.median()
    q = zp.groupby("year").hz.quantile([0.25, 0.75]).unstack()
    for yr in med.index:
        col = MUTED if yr in (2019, 2020) else INK
        a.plot([yr, yr], [q.loc[yr, 0.25], q.loc[yr, 0.75]], color=col, lw=3)
        a.scatter([yr], [med[yr]], color=col, s=40, zorder=3)
    for yr, dx, dy in ((2005, 0.6, 0), (2014, -0.6, -0.25), (2015, 0.6, 0.12), (2019, -0.6, -0.15), (2020, 0.6, 0)):
        if yr in med.index:
            a.text(yr + dx, med[yr] + dy, f"{med[yr]:.1f}", va="center", ha="left" if dx > 0 else "right", fontsize=8)
    a.axvspan(2013.3, 2015.7, color=GRID, zorder=0)
    a.text(2014.5, 25.15, "dev sites", ha="center", fontsize=8, color=MUTED)
    a.text(2019.5, 25.15, "eval", ha="center", fontsize=8, color=MUTED)
    a.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    a.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
    a.set_ylim(25.0, 27.6)
    a.set_xlabel("year (Kerguelen)")
    a.set_ylabel("Z-call peak frequency (Hz)")
    a.set_title("Calls drift down ~0.13 Hz per year", fontsize=10)
    a.set_xlim(2003, 2022)
    for m in MODELS:
        b.plot(sh.index, sh[m], color=C[m], marker="o", ms=4, label=N[m])
    b.axvspan(-0.7, -0.5, color=GRID, zorder=0)
    b.text(-0.8, 0.78, "drift to\n2019/20", ha="right", fontsize=8, color=MUTED, va="top")
    b.set_xlabel("frequency shift of the call (Hz)")
    b.set_ylabel("BMABZ detection rate")
    b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    b.set_title("A -0.5 Hz shift costs a third of the detections", fontsize=10)
    b.legend(fontsize=8, loc="upper right")
    x = np.arange(3)
    for k, cond in enumerate(["original", "shift -0.6 Hz"]):
        vals = [rec.loc[m, cond] for m in MODELS]
        c.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=[C[m] for m in MODELS], alpha=1.0 if k == 0 else 0.45)
        for xi, v in zip(x, vals):
            c.text(xi + (k - 0.5) * 0.38, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    c.set_xticks(x, [N[m] for m in MODELS])
    c.set_ylabel("BMABZ recall, Kerguelen 2014/15")
    c.set_title("Real dev audio shifted by -0.6 Hz\n(solid: original, light: shifted)", fontsize=10)
    c.grid(axis="x", visible=False)
    title(fig, "Blue whale Z-calls have moved down in frequency, and all three detectors are tuned to the old one",
          "Left: measured Z-call frequency at Kerguelen (median and middle half). Middle: dev calls shifted in the probe. Right: whole dev recordings shifted (follow-up F1).")
    fig.tight_layout()
    save(fig, "03_frequency_drift")


# 4 tile edges
def fig_edges():
    h = t("e2_stitch.parquet")
    h = h[h.model == "nave"]
    order = ["cut by edge", "0-2 s", "2-5 s", "5-10 s", "> 10 s"]
    conds = [("2 s, mean", "NAVE as published (2 s overlap, averaged)", "#b9b8b2"),
             ("half-tile overlap, centre, retuned", "30 s overlap, centre stitching", C["nave"])]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(order))
    for k, (cond, lab, col) in enumerate(conds):
        r = h[h.condition == cond].groupby("bucket", observed=True).hit.mean().reindex(order)
        ax.bar(x + (k - 0.5) * 0.38, r.values, width=0.36, color=col, label=lab)
        for xi, v in zip(x, r.values):
            ax.text(xi + (k - 0.5) * 0.38, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    n = h[h.condition == conds[0][0]].groupby("bucket", observed=True).size().reindex(order)
    ax.set_xticks(x, [f"{o}\n(n={v})" for o, v in zip(order, n.values)])
    ax.set_xlabel("distance between the call and the nearest edge of NAVE's 60 s tiles")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper right")
    title(fig, "NAVE missed calls at its tile edges; a decoding change fixes it (+0.025 macro B)")
    fig.tight_layout()
    save(fig, "04_tile_edges")


# 5 level
def fig_level():
    pr = t("probe_results.parquet")
    lv = pd.concat([pr[pr.probe == "level"], pr[pr.probe == "base"].assign(value=0.0)])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for m in MODELS:
            r = lv[(lv.cls == c) & (lv.model == m)].groupby("value").detected.mean()
            ax.plot(r.index, r.values, color=C[m], marker="o", ms=4, label=N[m])
        ax.set_title(V.CLASS_NAME[c], fontsize=10)
        ax.set_xlabel("gain applied to the whole clip (dB)")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("detection rate")
    axes[0].legend(fontsize=8, loc="lower left")
    title(fig, "Only YOLO ignores the recording level; Whale-VAD needs loud calls, NAVE breaks when too loud",
          "Real dev calls in their own context, louder or quieter. NAVE's drop comes from its linear magnitude channel, its PCEN channel is level-invariant.")
    fig.tight_layout()
    save(fig, "05_level")


# 6 SNR
def fig_snr():
    pr = t("probe_results.parquet")
    g = pr[pr.probe == "snr"]
    s50 = V.snr50_table().groupby("cls").median()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for m in MODELS:
            r = g[(g.cls == c) & (g.model == m)].groupby("value").detected.mean()
            ax.plot(r.index, r.values, color=C[m], marker="o", ms=4,
                    label=f"{N[m]} (SNR50 {s50.loc[c, m]:.1f} dB)" if np.isfinite(s50.loc[c, m]) else N[m])
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(V.CLASS_NAME[c], fontsize=10)
        ax.set_xlabel("SNR of the pasted call (dB, nominal)")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("detection rate")
    title(fig, "NAVE finds the faintest calls; YOLO needs about 6 dB more for BMABZ and never gets BP right",
          "Real calls pasted into call-free background of four sites at controlled SNR. SNR50 = SNR at which half are found.")
    fig.tight_layout()
    save(fig, "06_snr")


# 7 fine labels
def fig_fine():
    w = t("labels_wide_loso.parquet")
    fines = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
    names = {"bma": "bma (Z-call A)", "bmb": "bmb (Z-call B)", "bmz": "bmz (Z-call)", "bmd": "bmd (blue D)",
             "bpd": "bpd (fin 40 Hz)", "bp20": "bp20 (fin pulse)", "bp20plus": "bp20plus (fin + higher)"}
    mat = pd.DataFrame({N[m]: [w[w.fine == f][f"hit_{m}"].mean() for f in fines] for m in MODELS}, index=[names[f] for f in fines])
    n = [int((w.fine == f).sum()) for f in fines]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9ec2ec", "#2a78d6", "#0d3f80"])
    ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(mat.values):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(3), mat.columns)
    ax.set_yticks(range(len(fines)), [f"{x}  n={k}" for x, k in zip(mat.index, n)])
    ax.grid(False)
    ax.add_patch(mpl.patches.Rectangle((-0.5, 2.5), 3, 2, fill=False, ec=INK, lw=1.5))
    ax.text(2.6, 3.5, "the D class", fontsize=8, va="center", color=MUTED)
    title(fig, "D is two problems: blue whale D is found, fin whale 40 Hz calls almost never",
          "Recall per fine label, three dev sites pooled (LOSO thresholds).")
    fig.tight_layout()
    save(fig, "07_fine_labels")


# 8 transients: probe vs real
def fig_transient():
    pr = t("probe_results.parquet")
    g = pr[pr.probe == "transient"]
    base = pr[pr.probe == "base"].groupby(["model", "cls"]).detected.mean()
    eff = g[g.value == 30].groupby(["model", "cls"]).detected.mean() - base
    tr = t("transient_vs_recall.parquet").merge(t("labels_validation.parquet")[["label_id", "snr_db"]], on="label_id")
    tr = tr[tr.snr_db >= tr.snr_db.quantile(2 / 3)]
    tr["burst"] = pd.cut(tr.burst_vs_call, [-99, 0, 5, 99], labels=["no louder sound", "0 to 5 dB louder", "> 5 dB louder"])
    rr = tr.groupby("burst", observed=True)[[f"hit_{m}" for m in MODELS]].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(3)
    for k, m in enumerate(MODELS):
        v = eff.loc[m].reindex(CLASSES).values
        a.bar(x + (k - 1) * 0.27, v, width=0.25, color=C[m], label=N[m])
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(x, [V.CLASS_NAME[c] for c in CLASSES])
    a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    a.set_ylabel("change in detection rate")
    a.set_title("Probe: a +30 dB burst 10 to 30 s away", fontsize=10)
    a.legend(fontsize=8)
    a.grid(axis="x", visible=False)
    for m in MODELS:
        b.plot(range(3), rr[f"hit_{m}"].values, color=C[m], marker="o", label=N[m])
    b.set_xticks(range(3), rr.index)
    b.set_ylabel("recall")
    b.set_xlabel("loudest other second in the same 60 s, compared with the call")
    b.set_title("Real dev recordings, clear calls (top third SNR)", fontsize=10)
    title(fig, "NAVE suffers most from loud synthetic bursts, but in real recordings it loses the least",
          "Real loud sounds are rarely more than 10 dB above the call, the probe used +30 dB.")
    fig.tight_layout()
    save(fig, "08_transients")


# 9 agreement caution
def fig_agreement():
    ag = t("agreement.parquet")
    dev = ag[ag.dataset.isin(paths.SITES["validation"])].dropna(subset=["dev_precision"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for m in MODELS:
        g = dev[dev.model == m]
        ax.scatter(g.dev_precision, g.est_precision, color=C[m], s=40, label=N[m])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("real precision (dev site, from the labels)")
    ax.set_ylabel("precision estimated from model agreement")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.text(0.98, 0.04, "one dot = model x site x class", ha="right", fontsize=7, color=MUTED, transform=ax.transAxes)
    title(fig, "Agreement is a poor stand-in for labels",
          "Estimated from how often each model combination is real on the three dev sites together,\nyet off by up to 0.4 on single sites.")
    fig.tight_layout()
    save(fig, "09_agreement")


def main():
    V.style()
    for f in (fig_scores, fig_errors, fig_edges, fig_level, fig_snr, fig_fine, fig_transient, fig_agreement, fig_drift):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("[reportfigs] FAILED", f.__name__, e)


if __name__ == "__main__":
    main()
