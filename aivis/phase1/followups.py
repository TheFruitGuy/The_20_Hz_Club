"""Pre-registered follow-ups F1, F2 and E1 (ai-interactions/preregistration.md, 22.10.2026).

    python -m aivis.phase1.followups make      # frequency-shifted copies of four sites
    python -m aivis.phase1.followups run       # the three models on them, NAVE with 30 s overlap
    python -m aivis.phase1.followups report    # tables f1_dev_shift, f2_eval_shift, e1_overlap
    python -m aivis.phase1.followups all

F1  kerguelen2014 and kerguelen2015 shifted down by 0.6 Hz (single sideband, whole recording)
F2  kerguelen2019 and kerguelen2020 shifted up by 0.6 Hz
E1  NAVE on the three dev sites with 30 s tile overlap instead of 2 s
E2  NAVE (30 s overlap) and Whale-VAD (15 s overlap, half its 30 s tile) with centre stitching, thresholds retuned LOSO

    python -m aivis.phase1.followups e2        # run and report E2 only
Thresholds as in the main run: LOSO on the dev sites, tuned on all three dev sites for the eval sites.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import hilbert

from .. import paths, predict
from . import build, core
from .core import CLASSES, MODELS, TABLES

ROOT = core.ROOT / "followups"
SHIFT_HZ = 0.6
SHIFTS = {"kerguelen2014": -SHIFT_HZ, "kerguelen2015": -SHIFT_HZ,
          "kerguelen2019": +SHIFT_HZ, "kerguelen2020": +SHIFT_HZ}


def say(msg):
    print(f"[followups] {msg}", flush=True)


def shift_file(args):
    src, dst, hz = args
    if dst.exists():
        return
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)
    x = x[:, 0]
    t = np.arange(len(x)) / sr
    y = np.real(hilbert(x) * np.exp(2j * np.pi * hz * t))
    tmp = dst.with_name(dst.name + ".part")          # not *.wav, so a half-written file is never picked up
    sf.write(str(tmp), y.astype(np.float32), sr, subtype="FLOAT", format="WAV")
    tmp.rename(dst)


def make(a) -> None:
    jobs = []
    for site, hz in SHIFTS.items():
        split = core.split_of(site)
        out = ROOT / "audio" / site
        out.mkdir(parents=True, exist_ok=True)
        jobs += [(w, out / w.name, hz) for w in sorted((paths.audio_dir(split) / site).glob("*.wav"))]
    todo = [j for j in jobs if not j[1].exists()]
    say(f"make: {len(todo)} of {len(jobs)} files to shift")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(shift_file, todo, chunksize=4), 1):
            if i % 100 == 0:
                say(f"make: {i}/{len(todo)}")
    say("make: done")


def thr_arg(model: str, site: str) -> list[str]:
    return ["--thresholds", ",".join(str(t) for t in build.thr_for(model, site, "loso"))] if model != "yolo" else ["--conf", "0.01"]


def run(a) -> None:
    for site in SHIFTS:
        for m in MODELS:
            out = ROOT / "runs" / m / f"{site}_shift"
            if (out / "run.json").exists():
                continue
            say(f"run: {m} {site} shifted")
            predict.main([m, "--audio-dir", str(ROOT / "audio" / site), "--dataset", site, "--out", str(out),
                          "--overwrite", "--tag", "followup", "--no-frame-probs", "--device", a.device, *thr_arg(m, site)])
    for site in paths.SITES["validation"]:
        out = ROOT / "runs" / "nave" / f"{site}_ov30"
        if (out / "run.json").exists():
            continue
        say(f"run: nave {site} overlap 30 s")
        predict.main(["nave", "--split", "validation", "--sites", site, "--out", str(out), "--overwrite", "--tag", "followup",
                      "--no-frame-probs", "--overlap-s", "30", "--device", a.device, *thr_arg("nave", site)])


def load_run(model: str, name: str, site: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "runs" / model / name / "detections.csv")
    d = d.rename(columns={"annotation": "cls", "confidence": "conf"})
    if model == "yolo":
        thr = dict(zip(CLASSES, build.thr_for("yolo", site, "loso")))
        d = d[d.conf >= d.cls.map(thr)]
    return core.add_abs(d.reset_index(drop=True))


def baseline(model: str, site: str) -> pd.DataFrame:
    ev = pd.read_parquet(build.events_path(model, "loso"))
    return ev[ev.dataset == site].reset_index(drop=True)


def edge_margin(centre: np.ndarray, half: np.ndarray, seg=60.0, step=58.0) -> np.ndarray:
    """Margin between a call and the nearest edge of the best tile containing its centre (negative: cut)."""
    out = np.empty(len(centre))
    for i, (c, h) in enumerate(zip(centre, half)):
        k0 = max(0, int(np.floor((c - seg) / step)))
        best = -np.inf
        for k in range(k0, int(c // step) + 1):
            a = k * step
            if a <= c < a + seg:
                best = max(best, min(c - a, a + seg - c))
        out[i] = (best if np.isfinite(best) else 0.0) - h
    return out


def report(a) -> None:
    gt = core.load_labels("validation")
    rows = []
    # F1: dev sites shifted down
    for site in ("kerguelen2014", "kerguelen2015"):
        g = gt[gt.dataset == site].reset_index(drop=True)
        for m in MODELS:
            if not (ROOT / "runs" / m / f"{site}_shift" / "run.json").exists():
                continue
            for cond, ev in (("original", baseline(m, site)), ("shift -0.6 Hz", load_run(m, f"{site}_shift", site))):
                t, mac = core.score(ev, g)
                for r in t[t.dataset == site].itertuples():
                    rows.append({"model": m, "site": site, "condition": cond, "cls": r.cls, "recall": r.recall,
                                 "precision": r.precision, "f1": r.f1, "tp": r.tp, "fp": r.fp, "fn": r.fn})
    f1 = pd.DataFrame(rows)
    if len(f1):
        f1.to_parquet(TABLES / "f1_dev_shift.parquet")
        piv = f1.pivot_table(index=["site", "cls"], columns=["model", "condition"], values="recall").round(3)
        say("F1, recall at LOSO thresholds:\n" + piv.to_string())
        drop = f1.pivot_table(index=["model", "cls"], columns="condition", values=["tp", "fn"], aggfunc="sum")
        rec = drop["tp"] / (drop["tp"] + drop["fn"])
        say("F1, recall pooled over both sites, change in points:\n"
            + ((rec["shift -0.6 Hz"] - rec["original"]) * 100).round(1).unstack().to_string())
    # F2: eval sites shifted up, detections per hour
    st = pd.read_parquet(TABLES / "site_stats.parquet")
    hours = st.groupby("dataset").duration_s.sum() / 3600
    rows = []
    for site in ("kerguelen2019", "kerguelen2020"):
        for m in MODELS:
            if not (ROOT / "runs" / m / f"{site}_shift" / "run.json").exists():
                continue
            for cond, ev in (("original", baseline(m, site)), ("shift +0.6 Hz", load_run(m, f"{site}_shift", site))):
                for c in CLASSES:
                    rows.append({"model": m, "site": site, "condition": cond, "cls": c,
                                 "per_hour": float((ev.cls == c).sum() / hours[site])})
    f2 = pd.DataFrame(rows)
    if len(f2):
        f2.to_parquet(TABLES / "f2_eval_shift.parquet")
        piv = f2.pivot_table(index=["site", "cls"], columns=["model", "condition"], values="per_hour").round(2)
        say("F2, detections per hour:\n" + piv.to_string())
        tot = f2.pivot_table(index=["model", "cls"], columns="condition", values="per_hour", aggfunc="sum")
        say("F2, change in % (both sites):\n" + ((tot["shift +0.6 Hz"] / tot["original"] - 1) * 100).round(1).unstack().to_string())
    # E1: NAVE with 30 s overlap
    rows, mrows = [], []
    for site in paths.SITES["validation"]:
        if not (ROOT / "runs" / "nave" / f"{site}_ov30" / "run.json").exists():
            continue
        g = gt[gt.dataset == site].reset_index(drop=True)
        margin = edge_margin(((g.start_s + g.end_s) / 2).to_numpy(), (g.duration / 2).to_numpy())
        bucket = pd.cut(margin, [-99, 0, 2, 5, 10, 999], labels=["cut by edge", "0-2 s", "2-5 s", "5-10 s", "> 10 s"])
        for cond, ev in (("overlap 2 s", baseline("nave", site)), ("overlap 30 s", load_run("nave", f"{site}_ov30", site))):
            _, gm = core.official_match(ev, g, 0.3)
            rows.append(pd.DataFrame({"site": site, "condition": cond, "cls": g.cls, "bucket": bucket, "hit": gm >= 0}))
            t, mac = core.score(ev, g)
            mrows.append({"site": site, "condition": cond, "macro_B": float(mac[mac.dataset == site].macro_B.iloc[0])})
    if rows:
        e1 = pd.concat(rows, ignore_index=True)
        e1.to_parquet(TABLES / "e1_overlap.parquet")
        say("E1, NAVE recall by margin to the old tile edge:\n"
            + e1.pivot_table(index="bucket", columns="condition", values="hit", aggfunc="mean", observed=True).round(3).to_string())
        say("E1, recall per class:\n" + e1.pivot_table(index="cls", columns="condition", values="hit").round(3).to_string())
        mb = pd.DataFrame(mrows)
        say("E1, macro B per site:\n" + mb.pivot_table(index="site", columns="condition", values="macro_B").round(3).to_string())
        # all three dev sites joined (each scored on its own, like the main run)
        joined = []
        for cond in ("overlap 2 s", "overlap 30 s"):
            evs = pd.concat([baseline("nave", s) if cond == "overlap 2 s" else load_run("nave", f"{s}_ov30", s)
                             for s in paths.SITES["validation"]], ignore_index=True)
            _, mac = core.score(evs, gt)
            joined.append({"condition": cond, "macro_B_joined": float(mac[mac.dataset == "joined"].macro_B.iloc[0])})
        say("E1, macro B joined:\n" + pd.DataFrame(joined).round(3).to_string(index=False))


# ---------------------------------------------------------------------------------- E2

TILE = {"nave": (60.0, 58.0), "whalevad": (30.0, 28.0)}       # tile length and step of the 2 s runs
E2_OVERLAP = {"nave": "30", "whalevad": "15"}                  # half a tile for each model


def e2(a) -> None:
    gt = core.load_labels("validation")
    val = paths.SITES["validation"]
    for m in ("nave", "whalevad"):
        for site in val:
            out = ROOT / "runs" / m / f"{site}_ov30c"
            if (out / "run.json").exists():
                continue
            say(f"e2: {m} {site}, {E2_OVERLAP[m]} s overlap, centre stitching")
            predict.main([m, "--split", "validation", "--sites", site, "--out", str(out), "--overwrite", "--tag", "followup",
                          "--overlap-s", E2_OVERLAP[m], "--stitch", "centre", "--device", a.device, *thr_arg(m, site)])
    sweep, hits, macro_rows = [], [], []
    for m in ("nave", "whalevad"):
        # sweep over the grid on the new probabilities, then LOSO thresholds as in the main run
        evs_grid = {}
        for site in val:
            g = gt[gt.dataset == site].reset_index(drop=True)
            fp_dir = ROOT / "runs" / m / f"{site}_ov30c" / "frame_probs" / site
            parts = []
            for f in sorted(fp_dir.glob("*.npz")):
                with np.load(f) as z:
                    sm = core.smooth(z["probs"].astype(np.float32))
                for ci, c in enumerate(CLASSES):
                    for thr in core.GRID:
                        st, en, cf = core.runs_to_events(sm[:, ci], thr)
                        if len(st):
                            parts.append(pd.DataFrame({"dataset": site, "filename": f.stem + ".wav", "cls": c, "cls_order": ci,
                                                       "thr": thr, "start_s": st, "end_s": en, "conf": cf}))
            ev = core.add_abs(pd.concat(parts, ignore_index=True).sort_values(["filename", "cls_order", "start_s"], kind="stable"))
            evs_grid[site] = ev
            for c in CLASSES:
                gc = g[g.cls == c].reset_index(drop=True)
                ec = ev[ev.cls == c]
                for thr in core.GRID:
                    pe = ec[np.isclose(ec.thr, thr)].reset_index(drop=True)
                    pm, gm = core.official_match(pe, gc, 0.3)
                    sweep.append({"model": m, "site": site, "cls": c, "thr": float(thr),
                                  "tp": int((gm >= 0).sum()), "fn": int((gm < 0).sum()), "fp": int((pm < 0).sum())})
            say(f"e2: sweep {m} {site}")
        sw = pd.DataFrame([r for r in sweep if r["model"] == m])
        loso = {s: [build.best_thr(sw[(sw.cls == c) & (sw.site != s)]) for c in CLASSES] for s in val}
        say(f"e2: {m} LOSO thresholds half-tile centre {loso}, 2 s mean {build.thresholds()[m]['loso_iou0.3']}")
        seg, step = TILE[m]
        conds = {"2 s, mean": {s: baseline(m, s) for s in val},
                 "half-tile overlap, centre, old thresholds": {s: core.add_abs(load_run(m, f"{s}_ov30c", s)) for s in val},
                 "half-tile overlap, centre, retuned": {}}
        for s in val:
            ev = evs_grid[s]
            parts = [ev[(ev.cls == c) & np.isclose(ev.thr, t)] for c, t in zip(CLASSES, loso[s])]
            conds["half-tile overlap, centre, retuned"][s] = pd.concat(parts).sort_values(["filename", "cls_order", "start_s"], kind="stable").reset_index(drop=True)
        for cond, by_site in conds.items():
            allev = pd.concat(by_site.values(), ignore_index=True)
            t, mac = core.score(allev, gt)
            for r in mac.itertuples():
                macro_rows.append({"model": m, "condition": cond, "dataset": r.dataset, "macro_A": r.macro_A, "macro_B": r.macro_B})
            for s in val:
                g = gt[gt.dataset == s].reset_index(drop=True)
                margin = edge_margin(((g.start_s + g.end_s) / 2).to_numpy(), (g.duration / 2).to_numpy(), seg, step)
                bucket = pd.cut(margin, [-99, 0, 2, 5, 10, 999], labels=["cut by edge", "0-2 s", "2-5 s", "5-10 s", "> 10 s"])
                _, gm = core.official_match(by_site[s].reset_index(drop=True), g, 0.3)
                hits.append(pd.DataFrame({"model": m, "condition": cond, "site": s, "cls": g.cls, "bucket": bucket, "hit": gm >= 0}))
    pd.DataFrame(sweep).to_parquet(TABLES / "e2_sweep.parquet")
    h = pd.concat(hits, ignore_index=True)
    h.to_parquet(TABLES / "e2_stitch.parquet")
    mac = pd.DataFrame(macro_rows)
    mac.to_parquet(TABLES / "e2_macro.parquet")
    say("E2, recall by margin to the old tile edge:\n" + h.pivot_table(index=["model", "bucket"], columns="condition", values="hit",
                                                                     aggfunc="mean", observed=True).round(3).to_string())
    say("E2, recall per class:\n" + h.pivot_table(index=["model", "cls"], columns="condition", values="hit").round(3).to_string())
    say("E2, macro B:\n" + mac.pivot_table(index=["model", "dataset"], columns="condition", values="macro_B").round(3).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["make", "run", "report", "e2", "all"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    for s in (["make", "run", "report"] if a.step == "all" else [a.step]):
        {"make": make, "run": run, "report": report, "e2": e2}[s](a)


if __name__ == "__main__":
    main()
