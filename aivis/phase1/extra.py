"""Extra Phase I analyses that need no new model runs.

    python -m aivis.phase1.extra bootstrap    # 95 % intervals for macro B and the differences between models
    python -m aivis.phase1.extra level        # recording level per file vs recall per model (C5)
    python -m aivis.phase1.extra transient    # loudest burst in a call's tile vs recall per model (C4, H6)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import soundfile as sf

from .. import paths
from . import build, core
from .core import CLASSES, MODELS, TABLES

VAL = paths.SITES["validation"]


def say(m):
    print(f"[extra] {m}", flush=True)


# -------------------------------------------------------------------------- bootstrap

def per_file_counts() -> pd.DataFrame:
    """tp, fn, fp per model, site, file and class at the LOSO operating point (official matching)."""
    w = pd.read_parquet(TABLES / "labels_wide_loso.parquet")
    eo = pd.read_parquet(TABLES / "event_outcomes_loso.parquet")
    eo = eo[eo.dataset.isin(VAL)]
    rows = []
    for m in MODELS:
        lab = w.assign(tp=w[f"hit_{m}"].astype(int), fn=(~w[f"hit_{m}"]).astype(int)) \
            .groupby(["dataset", "filename", "cls"])[["tp", "fn"]].sum()
        fp = eo[(eo.model == m) & (eo.gt_match < 0)].groupby(["dataset", "filename", "cls"]).size().rename("fp")
        t = lab.join(fp, how="outer").fillna(0).astype(int).reset_index().assign(model=m)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def macro_b(tp, fp, fn):
    """tp/fp/fn arrays (..., 3 classes) -> macro B like the leaderboard (classes without detections skipped in P)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
    P = np.nanmean(p, axis=-1)
    R = np.nanmean(r, axis=-1)
    P, R = np.nan_to_num(P), np.nan_to_num(R)
    return np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)


def bootstrap(a) -> None:
    c = per_file_counts()
    files = c[["dataset", "filename"]].drop_duplicates().reset_index(drop=True)
    files["fid"] = np.arange(len(files))
    c = c.merge(files, on=["dataset", "filename"])
    n_f = len(files)
    # array (model, file, class, [tp, fp, fn])
    arr = np.zeros((len(MODELS), n_f, 3, 3))
    for (mi, m) in enumerate(MODELS):
        g = c[c.model == m]
        for k, col in enumerate(("tp", "fp", "fn")):
            np.add.at(arr[mi, :, :, k], (g.fid.to_numpy(), g.cls.map({x: i for i, x in enumerate(CLASSES)}).to_numpy()), g[col].to_numpy())
    rng = np.random.default_rng(0)
    by_site = {s: files.index[files.dataset == s].to_numpy() for s in VAL}
    B = a.n
    res = {f"{m}": [] for m in MODELS}
    res_site = {(m, s): [] for m in MODELS for s in VAL}
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_site.values()])
        tot = arr[:, idx].sum(axis=1)                               # (model, class, 3)
        mb = macro_b(tot[..., 0], tot[..., 1], tot[..., 2])
        for mi, m in enumerate(MODELS):
            res[m].append(mb[mi])
        for s, ix in by_site.items():
            sel = idx[np.isin(idx, ix)]
            t = arr[:, sel].sum(axis=1)
            v = macro_b(t[..., 0], t[..., 1], t[..., 2])
            for mi, m in enumerate(MODELS):
                res_site[(m, s)].append(v[mi])
    full = arr.sum(axis=1)
    point = dict(zip(MODELS, macro_b(full[..., 0], full[..., 1], full[..., 2])))
    rows = []
    for m in MODELS:
        v = np.array(res[m])
        rows.append({"what": m, "site": "joined", "point": point[m], "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)})
        for s in VAL:
            v = np.array(res_site[(m, s)])
            fs = arr[MODELS.index(m)][by_site[s]].sum(axis=0)
            rows.append({"what": m, "site": s, "point": float(macro_b(fs[:, 0], fs[:, 1], fs[:, 2])),
                         "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)})
    for x, y in (("nave", "whalevad"), ("nave", "yolo"), ("whalevad", "yolo")):
        d = np.array(res[x]) - np.array(res[y])
        rows.append({"what": f"{x} - {y}", "site": "joined", "point": point[x] - point[y],
                     "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "share_above_0": float((d > 0).mean())})
    out = pd.DataFrame(rows)
    out.to_parquet(TABLES / "bootstrap_macro_b.parquet")
    say(f"bootstrap over {n_f} recordings, {B} resamples, stratified by site:\n" + out.round(3).to_string(index=False))


# ------------------------------------------------------------------------------ level

def level(a) -> None:
    """Per recording: broadband RMS level, recall per model (files with 20+ labels)."""
    st = pd.read_parquet(TABLES / "site_stats.parquet")
    w = pd.read_parquet(TABLES / "labels_wide_loso.parquet")
    rec = w.groupby(["dataset", "filename"]).agg(n=("label_id", "size"), **{f"recall_{m}": (f"hit_{m}", "mean") for m in MODELS},
                                                 snr=("snr_db", "median")).reset_index()
    rec = rec.merge(st[["dataset", "filename", "rms_db", "noise_5-30Hz_db", "noise_30-60Hz_db"]], on=["dataset", "filename"])
    rec = rec[rec.n >= 20]
    rec.to_parquet(TABLES / "level_vs_recall.parquet")
    say("median recording level per site (dB re full scale):\n" + st.groupby("dataset").rms_db.median().round(1).to_string())
    rows = []
    for s, g in list(rec.groupby("dataset")) + [("all", rec)]:
        for m in MODELS:
            rows.append({"site": s, "model": m, "spearman_level_recall": g.rms_db.corr(g[f"recall_{m}"], method="spearman"),
                         "spearman_snr_recall": g.snr.corr(g[f"recall_{m}"], method="spearman"), "files": len(g)})
    say("rank correlation of recording level and recall per file:\n"
        + pd.DataFrame(rows).pivot_table(index="site", columns="model", values="spearman_level_recall").round(2).to_string())
    pd.DataFrame(rows).to_parquet(TABLES / "level_corr.parquet")


# -------------------------------------------------------------------------- transient

def transient(a) -> None:
    """For every dev label: how loud the loudest 1 s burst in the rest of its NAVE tile is, relative to the
    call's own broadband level. Then recall per model against that (C4, H6)."""
    w = pd.read_parquet(TABLES / "labels_wide_loso.parquet")
    out = TABLES / "transient_in_tile.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["label_id"])
    have = set(done.label_id)
    rows = []
    import time
    t_end = time.time() + a.budget_s
    for (ds, fn), g in w.groupby(["dataset", "filename"]):
        if set(g.label_id) <= have:
            continue
        x, sr = sf.read(str(paths.audio_dir("validation") / ds / fn), dtype="float32", always_2d=True)
        x = x[:, 0].astype(np.float64)
        n = len(x) // sr
        # broadband power per 1 s frame, 5-124 Hz (simple high-pass by removing the mean of each second)
        fr = x[:n * sr].reshape(n, sr)
        fr = fr - fr.mean(axis=1, keepdims=True)
        pw = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-20)
        med = np.median(pw)
        for r in g.itertuples():
            c = (r.start_s + r.end_s) / 2
            k = int(c // 58)
            a0, a1 = k * 58, k * 58 + 60                      # NAVE tile that starts before the call
            if a1 > n:
                a0, a1 = max(0, n - 60), n
            s0, s1 = int(max(r.start_s - 1, 0)), int(min(r.end_s + 1, n))
            mask = np.ones(a1 - a0, bool)
            mask[max(s0 - a0, 0):max(s1 - a0, 0)] = False
            other = pw[a0:a1][mask]
            call = pw[s0:max(s1, s0 + 1)].max() if s1 > s0 else np.nan
            rows.append({"label_id": r.label_id, "tile_max_db": float(other.max() - med) if len(other) else np.nan,
                         "call_db": float(call - med), "file_median_db": float(med)})
        if time.time() > t_end:
            break
    new = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    new.to_parquet(out)
    left = len(w) - len(new)
    say(f"transient: {len(new)} labels done, {left} left" + (" (run again)" if left else ""))
    if left:
        return
    t = w.merge(new, on="label_id")
    t["burst_vs_call"] = t.tile_max_db - t.call_db
    t["bin"] = pd.cut(t.burst_vs_call, [-99, 0, 5, 10, 15, 99], labels=["burst below call", "0-5 dB", "5-10 dB", "10-15 dB", "> 15 dB"])
    tab = t.pivot_table(index="bin", values=[f"hit_{m}" for m in MODELS], aggfunc="mean", observed=True).round(3)
    tab["labels"] = t.groupby("bin", observed=True).size()
    say("recall by how much louder the loudest other second in the tile is than the call:\n" + tab.to_string())
    t[["label_id", "dataset", "cls", "burst_vs_call", "bin"] + [f"hit_{m}" for m in MODELS]].to_parquet(TABLES / "transient_vs_recall.parquet")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["bootstrap", "level", "transient"])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--budget-s", type=float, default=150)
    a = p.parse_args(argv)
    {"bootstrap": bootstrap, "level": level, "transient": transient}[a.step](a)


if __name__ == "__main__":
    main()
