"""Pre-registered tests EN1 to EN4 (ai-interactions/preregistration.md, 25.10.2026): do the three models combine?

    python -m aivis.phase1.ensemble route     # per-class best model (tables only, seconds)
    python -m aivis.phase1.ensemble vote      # events confirmed by other models (tables only, a minute)
    python -m aivis.phase1.ensemble hybrid    # exploratory: voting for BMABZ only, NAVE for D and BP
    python -m aivis.phase1.ensemble fuse [--workers 8]   # weighted score fusion (needs the Phase I frame
                                              # probabilities and YOLO boxes: on the server, CPU only)
    python -m aivis.phase1.ensemble report

Everything is tuned like the Phase I thresholds: leave one dev site out, choose on the other two (best
pooled F1 per class), count on the held-out site, pool the counts of the three held-out sites, macro B.
Every variant is compared with NAVE alone through the same code path (EN1 to EN3 baselines).

route   for each class the single model with the best pooled F1 on the tuning sites, at its LOSO threshold
vote    vote2: events that at least two models detect (overlapping events of the class from 2 or more
        models form a group; the group keeps the boundaries of its best member, NAVE before Whale-VAD
        before YOLO); nave_confirmed: NAVE's events that at least one other model also detects
fuse    per class, fused score = w_yolo * YOLO track + w_wvad * Whale-VAD + w_nave * NAVE (frame models:
        smoothed probabilities as in their pipeline; YOLO: highest box confidence covering the frame), weights
        on a 0.25 grid summing to 1, thresholded with NAVE's post-processing (merge 0.5 s, 0.5 to 30 s);
        weights and threshold tuned LOSO per class
Tables: ens_route, ens_vote, ens_fuse, ens_fuse_weights, ens_fuse_counts.
"""
from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .. import paths
from . import build, core
from .core import CLASSES, GRID, MODELS, TABLES

VAL = paths.SITES["validation"]
ORDER = {"nave": 0, "whalevad": 1, "yolo": 2}          # whose boundaries a vote group keeps


def say(msg):
    print(f"[ensemble] {msg}", flush=True)


def labels() -> pd.DataFrame:
    return pd.read_parquet(TABLES / "labels_validation.parquet")


def macro_b(counts: pd.DataFrame) -> dict:
    """counts: cls, tp, fp, fn (summed over the held-out sites) -> macro B, mean P, mean R, per-class F1."""
    c = counts.groupby("cls")[["tp", "fp", "fn"]].sum().reindex(CLASSES, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = c.tp / (c.tp + c.fp)
        rec = c.tp / (c.tp + c.fn)
    p, r = float(np.nanmean(prec)) if prec.notna().any() else 0.0, float(np.nanmean(rec))
    out = {"macro_B": 0.0 if p + r == 0 else 2 * p * r / (p + r), "mean_P": p, "mean_R": r}
    for k in CLASSES:
        tp, fp, fn = c.loc[k]
        out[f"f1_{k}"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out


# --------------------------------------------------------------------------------- route

def route(a) -> pd.DataFrame:
    sw = pd.read_parquet(TABLES / "sweep.parquet")
    sw = sw[np.isclose(sw.iou, 0.3)]
    rows, held = [], {"route": [], **{m: [] for m in MODELS}}
    for s in VAL:
        for c in CLASSES:
            best = None
            for m in MODELS:
                tune = sw[(sw.model == m) & (sw.cls == c) & (sw.site != s)]
                thr = build.best_thr(tune)
                t = tune[np.isclose(tune.thr, thr)][["tp", "fp", "fn"]].sum()
                f = float(build.f1(t.tp, t.fp, t.fn))
                h = sw[(sw.model == m) & (sw.cls == c) & (sw.site == s) & np.isclose(sw.thr, thr)][["tp", "fp", "fn"]].sum()
                held[m].append({"cls": c, **h.to_dict()})
                if best is None or f > best[1]:
                    best = (m, f, thr, h)
            rows.append({"held_out": s, "cls": c, "model": best[0], "tune_f1": best[1], "thr": best[2]})
            held["route"].append({"cls": c, **best[3].to_dict()})
    res = pd.DataFrame([{"variant": k, **macro_b(pd.DataFrame(v))} for k, v in held.items()])
    pd.DataFrame(rows).to_parquet(TABLES / "ens_route_choice.parquet")
    res.to_parquet(TABLES / "ens_route.parquet")
    say("route, chosen model per held-out site and class:\n" + pd.DataFrame(rows).pivot(index="held_out", columns="cls", values="model").to_string())
    say("route:\n" + res.round(3).to_string(index=False))
    return res


# ---------------------------------------------------------------------------------- vote

def vote_variants() -> dict[str, pd.DataFrame]:
    """Event sets of the single models (LOSO thresholds) and of the voting rules, dev sites only."""
    evs = []
    for m in MODELS:
        e = pd.read_parquet(build.events_path(m, "loso"))
        e = e[e.dataset.isin(VAL)][["dataset", "filename", "cls", "start_s", "end_s", "conf", "t0", "t1"]].assign(model=m)
        evs.append(e)
    allev = pd.concat(evs, ignore_index=True).sort_values(["dataset", "filename", "cls", "t0"]).reset_index(drop=True)
    out = {"vote2": [], "nave_confirmed": [], "vote3": []}
    for (_, _, _), g in allev.groupby(["dataset", "filename", "cls"], sort=False):
        t0, t1, mm = g.t0.to_numpy(), g.t1.to_numpy(), g.model.to_numpy()
        idx = g.index.to_numpy()
        end, members = t1[0], [0]
        groups = []
        for i in range(1, len(g)):
            if t0[i] < end:                       # overlaps the running group
                end = max(end, t1[i])
                members.append(i)
            else:
                groups.append(members)
                end, members = t1[i], [i]
        groups.append(members)
        for mem in groups:
            models = set(mm[mem])
            best = min(mem, key=lambda i: (ORDER[mm[i]], -g.conf.iloc[i]))
            if len(models) >= 2:
                out["vote2"].append(idx[best])
            if len(models) >= 3:
                out["vote3"].append(idx[best])
            if "nave" in models and len(models) >= 2:
                out["nave_confirmed"] += [idx[i] for i in mem if mm[i] == "nave"]
    return {**{k: allev.loc[sorted(set(v))] for k, v in out.items()}, **{m: allev[allev.model == m] for m in MODELS}}


def site_counts(p: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["dataset", "filename", "t0"]).reset_index(drop=True)
    t, _ = core.score(p, gt)
    return t[t.dataset != "joined"][["dataset", "cls", "tp", "fp", "fn"]]


def vote(a) -> pd.DataFrame:
    gt = labels()
    rows = []
    for k, p in vote_variants().items():
        rows.append({"variant": k, "events": len(p), **macro_b(site_counts(p, gt))})
    res = pd.DataFrame(rows)
    res.to_parquet(TABLES / "ens_vote.parquet")
    say("vote:\n" + res.round(3).to_string(index=False))
    return res


def hybrid(a) -> pd.DataFrame:
    """Exploratory (chosen after seeing EN3): voting for BMABZ only, NAVE for D and BP. Also the honest
    version of the idea: per class and held-out site, the event set (NAVE, vote2, nave_confirmed) with the
    best pooled F1 on the two tuning sites."""
    gt = labels()
    V = vote_variants()
    C = {k: site_counts(p, gt) for k, p in V.items() if k in ("nave", "vote2", "nave_confirmed")}
    rows = []
    for name, src in (("bmabz_vote2", "vote2"), ("bmabz_nave_confirmed", "nave_confirmed")):
        cnt = pd.concat([C[src][C[src].cls == "bmabz"], C["nave"][C["nave"].cls != "bmabz"]])
        rows.append({"variant": name, **macro_b(cnt)})
    held, choice = [], []
    for s in VAL:
        for c in CLASSES:
            best = None
            for k, cnt in C.items():
                tune = cnt[(cnt.dataset != s) & (cnt.cls == c)][["tp", "fp", "fn"]].sum()
                f = float(build.f1(tune.tp, tune.fp, tune.fn))
                if best is None or f > best[1]:
                    best = (k, f)
            choice.append({"held_out": s, "cls": c, "set": best[0], "tune_f1": best[1]})
            held.append(C[best[0]][(C[best[0]].dataset == s) & (C[best[0]].cls == c)])
    rows.append({"variant": "per_class_choice_loso", **macro_b(pd.concat(held))})
    rows.append({"variant": "nave", **macro_b(C["nave"])})
    res = pd.DataFrame(rows)
    res.to_parquet(TABLES / "ens_hybrid.parquet")
    say("hybrid, chosen set per held-out site and class:\n" + pd.DataFrame(choice).pivot(index="held_out", columns="cls", values="set").to_string())
    say("hybrid (exploratory):\n" + res.round(3).to_string(index=False))
    # per site, so the gain can be checked site by site
    per_site = []
    for s in VAL:
        cnt = pd.concat([C["vote2"][(C["vote2"].cls == "bmabz") & (C["vote2"].dataset == s)],
                         C["nave"][(C["nave"].cls == "bmabz") & (C["nave"].dataset == s)].assign(dataset="nave_" + s)])
        for k, g in (("vote2", C["vote2"]), ("nave", C["nave"])):
            h = g[(g.dataset == s) & (g.cls == "bmabz")][["tp", "fp", "fn"]].sum()
            per_site.append({"site": s, "set": k, "f1_bmabz": float(build.f1(h.tp, h.fp, h.fn)),
                             "P": h.tp / max(h.tp + h.fp, 1), "R": h.tp / max(h.tp + h.fn, 1)})
    ps = pd.DataFrame(per_site)
    ps.to_parquet(TABLES / "ens_hybrid_sites.parquet")
    say("BMABZ per site:\n" + ps.round(3).to_string(index=False))
    return res


# ---------------------------------------------------------------------------------- fuse

WEIGHTS = [tuple(float(x) for x in w) for w in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), repeat=3) if abs(sum(w) - 1) < 1e-9]


def yolo_track(ev: pd.DataFrame, n: int) -> np.ndarray:
    tr = np.zeros((n, 3), np.float32)
    for r in ev.itertuples():
        a, b = int(max(0, np.floor(r.start_s / core.HOP_S))), int(min(n, np.ceil(r.end_s / core.HOP_S)))
        if b > a:
            k = CLASSES.index(r.cls)
            tr[a:b, k] = np.maximum(tr[a:b, k], r.conf)
    return tr


def fuse_file(args) -> pd.DataFrame:
    site, fn, yev = args
    tracks = {}
    for m in ("whalevad", "nave"):
        f = core.frame_probs_dir(m, site) / fn.replace(".wav", ".npz")
        with np.load(f) as z:
            tracks[m] = core.smooth(z["probs"].astype(np.float32))
    n = min(len(tracks["whalevad"]), len(tracks["nave"]))
    Y, W, N = yolo_track(yev, n), tracks["whalevad"][:n], tracks["nave"][:n]
    cols = {k: [] for k in ("wi", "ci", "thr", "start_s", "end_s", "conf")}
    for wi, (wy, ww, wn) in enumerate(WEIGHTS):
        F = wy * Y + ww * W + wn * N
        for ci in range(3):
            for thr in GRID:
                s, e, conf = core.runs_to_events(F[:, ci], float(thr))
                if len(s):
                    cols["wi"].append(np.full(len(s), wi, np.int16))
                    cols["ci"].append(np.full(len(s), ci, np.int8))
                    cols["thr"].append(np.full(len(s), thr, np.float32))
                    cols["start_s"].append(s.astype(np.float32))
                    cols["end_s"].append(e.astype(np.float32))
                    cols["conf"].append(conf.astype(np.float32))
    if not cols["wi"]:
        return pd.DataFrame(columns=list(cols) + ["filename"])
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df["filename"] = fn
    return df


def count_cell(args) -> list[dict]:
    site, c, ev_sc, g_sc = args
    out = []
    have = set()
    for (wi, thr), p in ev_sc.groupby(["wi", "thr"], sort=False):
        p = p.sort_values(["filename", "start_s"]).reset_index(drop=True)
        pm, gm = core.official_match(p, g_sc, 0.3)
        thr = round(float(thr), 2)                    # float32 thresholds -> the same float64 values as GRID
        have.add((int(wi), thr))
        out.append({"site": site, "cls": c, "wi": int(wi), "thr": thr, "tp": int((gm >= 0).sum()),
                    "fn": int((gm < 0).sum()), "fp": int((pm < 0).sum())})
    for wi in range(len(WEIGHTS)):                     # combinations without any event
        for thr in GRID:
            if (wi, float(thr)) not in have:
                out.append({"site": site, "cls": c, "wi": wi, "thr": float(thr), "tp": 0, "fn": len(g_sc), "fp": 0})
    return out


def clean_counts(c: pd.DataFrame) -> pd.DataFrame:
    """One row per (site, class, weights, threshold). Counts written before the float32 fix carry two rows
    for some thresholds (0.05 and 0.0500000007): the real one and an empty filler; keep the real one."""
    c = c.assign(thr=c.thr.astype(float).round(2))
    c = c.assign(_n=c.tp + c.fp).sort_values("_n", ascending=False)
    return c.drop_duplicates(["site", "cls", "wi", "thr"]).drop(columns="_n").reset_index(drop=True)


def fuse(a) -> pd.DataFrame:
    gt = labels()
    out_counts = TABLES / "ens_fuse_counts.parquet"
    if out_counts.exists() and not a.force:
        counts = clean_counts(pd.read_parquet(out_counts))
    else:
        res = []
        say(f"fuse: {len(WEIGHTS)} weight combinations x 3 classes x {len(GRID)} thresholds per file")
        for site in VAL:                               # one site at a time keeps the memory small
            yev = core.yolo_events(core.yolo_boxes(site))
            by_file = dict(tuple(yev.groupby("filename")))
            files = sorted(f.stem + ".wav" for f in core.frame_probs_dir("nave", site).glob("*.npz"))
            files = [f for f in files if (core.frame_probs_dir("whalevad", site) / f.replace(".wav", ".npz")).exists()]
            parts = []
            with ProcessPoolExecutor(a.workers) as ex:
                for i, df in enumerate(ex.map(fuse_file, [(site, f, by_file.get(f, yev.iloc[:0])) for f in files], chunksize=4), 1):
                    parts.append(df)
                    if i % 50 == 0:
                        say(f"fuse: {site} {i}/{len(files)} files")
            ev = pd.concat(parts, ignore_index=True)
            ev["dataset"] = site
            ev["cls"] = np.array(CLASSES)[ev.ci.to_numpy()]
            ev = core.add_abs(ev)
            g = gt[gt.dataset == site]
            cells = [(site, c, ev[ev.cls == c], g[g.cls == c].reset_index(drop=True)) for c in CLASSES]
            with ProcessPoolExecutor(3) as ex:
                for r in ex.map(count_cell, cells):
                    res += r
            say(f"fuse: {site} counted ({len(ev)} events over all combinations)")
            del ev, parts
        counts = clean_counts(pd.DataFrame(res))
        counts.to_parquet(out_counts)
    rows, wrows = [], []
    variants = {"fusion": range(len(WEIGHTS))}
    for name, w in (("nave", (0, 0, 1)), ("whalevad", (0, 1, 0)), ("yolo_track", (1, 0, 0))):
        variants[name] = [WEIGHTS.index(tuple(float(x) for x in w))]
    for name, allowed in variants.items():
        held = []
        for s in VAL:
            for c in CLASSES:
                tune = counts[(counts.site != s) & (counts.cls == c) & counts.wi.isin(list(allowed))]
                t = tune.groupby(["wi", "thr"])[["tp", "fp", "fn"]].sum()
                f = build.f1(t.tp.to_numpy(), t.fp.to_numpy(), t.fn.to_numpy())
                wi, thr = t.index[int(np.argmax(f))]
                h = counts[(counts.site == s) & (counts.cls == c) & (counts.wi == wi) & np.isclose(counts.thr, thr)]
                held.append({"cls": c, **h[["tp", "fp", "fn"]].sum().to_dict()})
                if name == "fusion":
                    wy, ww, wn = WEIGHTS[wi]
                    wrows.append({"held_out": s, "cls": c, "w_yolo": wy, "w_whalevad": ww, "w_nave": wn, "thr": thr,
                                  "tune_f1": float(f.max())})
        rows.append({"variant": name, **macro_b(pd.DataFrame(held))})
    res = pd.DataFrame(rows)
    W = pd.DataFrame(wrows)
    res.to_parquet(TABLES / "ens_fuse.parquet")
    W.to_parquet(TABLES / "ens_fuse_weights.parquet")
    say("fusion weights chosen per held-out site and class:\n" + W.round(2).to_string(index=False))
    say("fuse:\n" + res.round(3).to_string(index=False))
    return res


def report(a) -> None:
    parts = []
    for n in ("ens_route", "ens_vote", "ens_fuse"):
        f = TABLES / f"{n}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f).assign(test=n[4:]))
    if not parts:
        raise SystemExit("nothing yet (route, vote, fuse)")
    t = pd.concat(parts, ignore_index=True)
    pd.set_option("display.width", 200)
    for test, g in t.groupby("test", sort=False):
        base = g[g.variant.isin(["nave"])].macro_B
        b = float(base.iloc[0]) if len(base) else np.nan
        g = g.assign(gain_vs_nave=g.macro_B - b)
        say(f"{test}:\n" + g.drop(columns="test").round(3).to_string(index=False))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["route", "vote", "hybrid", "fuse", "report", "all"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    steps = ["route", "vote", "fuse", "report"] if a.step == "all" else [a.step]
    for s in steps:
        {"route": route, "vote": vote, "hybrid": hybrid, "fuse": fuse, "report": report}[s](a)


if __name__ == "__main__":
    main()
