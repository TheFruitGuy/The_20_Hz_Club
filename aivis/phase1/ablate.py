"""Pre-registered tests L1 to L4 (NAVE ablation ladder) and A1 to A4 (frequency-shift augmentation),
logs/preregistration.md, 23.10.2026.

    python -m aivis.phase1.ablate list                      # which checkpoints are there
    python -m aivis.phase1.ablate run [--only A B ...] [--seeds 0 1]
    python -m aivis.phase1.ablate report

Checkpoints (5 seeds each: 0, 1, 42, 666, 1337):
    A    no PCEN, no FDY, kernel 31        checkpoints/nave_ladder/nave_pcen0_fdy0_k31_seed<s>.pt
    B    + FDY                             checkpoints/nave_ladder/nave_pcen0_fdy1_k31_seed<s>.pt
    C    + PCEN                            checkpoints/nave_ladder/nave_pcen1_fdy1_k31_seed<s>.pt
    D    + kernel 153 (= NAVE)             checkpoints/nave_seeds/nave_k153_seed<s>.pt
    NOMAG, LOGMAG  NAVE retrained without the linear magnitude channel / with a normalized log magnitude
                   (aivis.phase1.train_input), outputs/phase1/input_runs/
    AUG  NAVE retrained with frequency-shift augmentation (aivis.phase1.train_aug)
                                           outputs/phase1/aug_runs/nave_s<s>_..._fshift1.5_*/nave_best.pt
    FINE AUG with 5 outputs per call type (aivis.phase1.train_fine, aivis.nave_heads)
                                           outputs/phase1/fine_runs/nave_s<s>_..._fine5_fshift1.5_*/nave_best.pt
Per checkpoint: the dev sites (2 s pipeline), Kerguelen 2014/15 shifted by -0.6 Hz (follow-up F1), the
base, level and shift probe clips, and for D and AUG also the original Kerguelen 2019/20 audio.
Thresholds as everywhere: LOSO over the dev sites, and tuned on all three for probes and eval sites.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths, predict
from . import core, followups, probes, seeds
from .core import CLASSES, TABLES

SEEDS = [0, 1, 42, 666, 1337]
RUNGS = {"A": "pcen0_fdy0_k31", "B": "pcen0_fdy1_k31", "C": "pcen1_fdy1_k31", "D": "k153", "AUG": "aug",
         "NOMAG": "nomag", "LOGMAG": "logmag", "FINE": "fine5"}
ROOT = core.ROOT / "ablate"
VAL = paths.SITES["validation"]
EVAL_K = ("kerguelen2019", "kerguelen2020")
PROBES = ("base", "level", "shift")


def say(m):
    print(f"[ablate] {m}", flush=True)


CKPT = paths.ROOT / "checkpoints"
SHARED = {"AUG": CKPT / "nave_aug" / "nave_k153_fshift1.5_seed{seed}.pt",
          "NOMAG": CKPT / "nave_input" / "nave_k153_nomag_seed{seed}.pt",
          "LOGMAG": CKPT / "nave_input" / "nave_k153_logmag_seed{seed}.pt",
          "FINE": CKPT / "nave_fine" / "nave_k153_fine5_fshift1.5_seed{seed}.pt"}


def ckpt(rung: str, seed: int) -> Path | None:
    """The committed checkpoint in checkpoints/ if there is one, else the newest finished run in outputs/."""
    if rung == "D":
        return seeds.ckpt(seed)
    if rung in SHARED and Path(str(SHARED[rung]).format(seed=seed)).exists():
        return Path(str(SHARED[rung]).format(seed=seed))
    if rung in ("NOMAG", "LOGMAG"):
        runs = sorted(Path(core.ROOT / "input_runs").glob(f"nave_s{seed}_*_{RUNGS[rung]}_*"))
        runs = [r for r in runs if (r / "nave_best.pt").exists() and (r / "nave_epoch_40.pt").exists()]
        return runs[-1] / "nave_best.pt" if runs else None
    if rung == "FINE":
        runs = sorted(Path(core.ROOT / "fine_runs").glob(f"nave_s{seed}_*fine5*"))
        runs = [r for r in runs if (r / "nave_best.pt").exists() and (r / "nave_epoch_40.pt").exists()]
        return runs[-1] / "nave_best.pt" if runs else None
    if rung == "AUG":
        runs = sorted(Path(core.ROOT / "aug_runs").glob(f"nave_s{seed}_*fshift*"))
        runs = [r for r in runs if (r / "nave_best.pt").exists() and (r / "nave_epoch_40.pt").exists()]
        return runs[-1] / "nave_best.pt" if runs else None
    p = paths.ROOT / "checkpoints" / "nave_ladder" / f"nave_{RUNGS[rung]}_seed{seed}.pt"
    return p if p.exists() else None


def dirs(rung: str, seed: int) -> dict[str, Path]:
    r = ROOT / f"{rung}_seed{seed}"
    d = {"2s": r / "val_2s", "shift": r / "shift", "probe": r / "probe60", "eval": r / "eval"}
    if rung == "D":   # the seed runs already have these
        s = seeds.run_dirs(seed)
        d.update({"2s": s["2s"], "shift": s["shift"]})
    return d


def probe_files() -> list[str]:
    man = pd.read_parquet(probes.PROBE_ROOT / "manifest.parquet")
    return sorted(man[(man.dataset == "probe60") & man.probe.isin(PROBES)].filename)


def run(a) -> None:
    files = probe_files()
    for rung in a.only:
        for seed in a.seeds:
            w = ckpt(rung, seed)
            if w is None:
                say(f"{rung} seed {seed}: no checkpoint (yet), skipped")
                continue
            d = dirs(rung, seed)
            base = ["nave", "--weights", str(w), "--device", a.device]
            for site in VAL:
                seeds.go([*base, "--split", "validation", "--sites", site], d["2s"] / site)
            for site in seeds.SHIFTED:
                seeds.go([*base, "--audio-dir", str(followups.ROOT / "audio" / site), "--dataset", site], d["shift"] / site)
            seeds.go([*base, "--audio-dir", str(probes.CLIPS / "probe60"), "--dataset", "probe60", "--files", *files], d["probe"])
            if rung in ("D", "AUG"):
                for site in EVAL_K:
                    seeds.go([*base, "--split", "evaluation", "--sites", site], d["eval"] / site)
    say("run: done")


def eval_rows(rung: str, seed: int, thr: list[float]) -> list[dict]:
    """BMABZ/D/BP detections per hour on the original Kerguelen 2019/20 audio at the given thresholds."""
    d = dirs(rung, seed)["eval"]
    st = pd.read_parquet(TABLES / "site_stats.parquet")
    hours = st.groupby("dataset").duration_s.sum() / 3600
    rows = []
    for site in EVAL_K:
        fp = d / site / "frame_probs" / site
        if not fp.exists():
            continue
        ev = seeds.at(seeds.grid_events(fp, site), thr)
        for c in CLASSES:
            rows.append({"rung": rung, "seed": seed, "site": site, "cls": c, "per_hour": float((ev.cls == c).sum() / hours[site])})
    return rows


def probe_rows(rung: str, seed: int, thr: list[float]) -> list[dict]:
    man = pd.read_parquet(probes.PROBE_ROOT / "manifest.parquet")
    man = man[(man.dataset == "probe60") & man.probe.isin(PROBES)]
    fp = dirs(rung, seed)["probe"] / "frame_probs" / "probe60"
    t = dict(zip(CLASSES, thr))
    rows = []
    for r in man.itertuples():
        f = fp / r.filename.replace(".wav", ".npz")
        if not f.exists():
            continue
        with np.load(f) as z:
            sm = core.smooth(z["probs"].astype(np.float32))
        ci = CLASSES.index(r.cls)
        s, e, _ = core.runs_to_events(sm[:, ci], float(core.GRID[np.argmin(np.abs(core.GRID - t[r.cls]))]))
        inter = np.clip(np.minimum(r.call_t1, e) - np.maximum(r.call_t0, s), 0, None)
        iou = inter / ((r.call_t1 - r.call_t0) + (e - s) - inter) if len(s) else np.zeros(0)
        f0 = int(max(0, np.floor(r.call_t0 / core.HOP_S)))
        f1 = int(min(len(sm), np.ceil(r.call_t1 / core.HOP_S)))
        rows.append({"rung": rung, "seed": seed, "call_id": r.call_id, "cls": r.cls, "probe": r.probe, "value": r.value,
                     "peak": float(sm[f0:max(f1, f0 + 1), ci].max()), "detected": bool((iou > 0.3).any())})
    return rows


def report(a) -> None:
    gt = core.load_labels("validation")
    scores, f1, prob, evr = [], [], [], []
    for rung in a.only:
        for seed in a.seeds:
            d = dirs(rung, seed)
            if not all((d["2s"] / s / "run.json").exists() for s in VAL):
                say(f"report: {rung} seed {seed} not run, skipped")
                continue
            evs = {s: seeds.grid_events(d["2s"] / s / "frame_probs" / s, s) for s in VAL}
            sw = seeds.sweep(evs, gt)
            loso, ins = seeds.tune(sw)
            allev = pd.concat([seeds.at(evs[s], loso[s]) for s in VAL], ignore_index=True)
            _, mac = core.score(allev, gt)
            scores.append({"rung": rung, "seed": seed, "macro_B": float(mac[mac.dataset == "joined"].macro_B.iloc[0]),
                           "thr_insample": ins})
            for site in seeds.SHIFTED:
                if not (d["shift"] / site / "run.json").exists():
                    continue
                g = gt[gt.dataset == site].reset_index(drop=True)
                sh = seeds.grid_events(d["shift"] / site / "frame_probs" / site, site)
                for cond, ev in (("original", seeds.at(evs[site], loso[site])), ("shifted", seeds.at(sh, loso[site]))):
                    for c in CLASSES:
                        gc = g[g.cls == c].reset_index(drop=True)
                        _, gm = core.official_match(ev[ev.cls == c].reset_index(drop=True), gc, 0.3)
                        f1.append({"rung": rung, "seed": seed, "site": site, "condition": cond, "cls": c,
                                   "tp": int((gm >= 0).sum()), "n": len(gc)})
            if (d["probe"] / "run.json").exists():
                prob += probe_rows(rung, seed, ins)
            if rung in ("D", "AUG"):
                evr += eval_rows(rung, seed, ins)
            say(f"report: {rung} seed {seed} done")
    order = [r for r in RUNGS if r in a.only]
    sc = pd.DataFrame(scores)
    sc.to_parquet(TABLES / "ablate_scores.parquet")
    say("LOSO macro B (dev joined), per rung and seed:\n"
        + sc.pivot_table(index="seed", columns="rung", values="macro_B").reindex(columns=order).round(3).to_string()
        + "\nmean:\n" + sc.groupby("rung").macro_B.agg(["mean", "std"]).reindex(order).round(3).to_string())
    if f1:
        t = pd.DataFrame(f1)
        t.to_parquet(TABLES / "ablate_f1.parquet")
        r = t.groupby(["rung", "seed", "cls", "condition"])[["tp", "n"]].sum()
        rec = (r.tp / r.n).unstack("condition")
        rec["change"] = (rec["shifted"] - rec["original"]) * 100
        ch = rec["change"].unstack("cls")
        say("F1-style test, recall change in points at -0.6 Hz (Kerguelen 2014/15), per seed:\n" + ch.round(1).to_string()
            + "\nmean per rung:\n" + ch.groupby("rung").mean().reindex(order).round(1).to_string())
    if prob:
        pr = pd.DataFrame(prob)
        pr.to_parquet(TABLES / "ablate_probes.parquet")
        sh = pr[pr.probe == "shift"]
        curve = sh[sh.cls == "bmabz"].groupby(["rung", "seed", "value"]).detected.mean().unstack("value")
        curve.rename(columns=str).to_parquet(TABLES / "ablate_shift.parquet")
        m = curve.groupby("rung").mean().reindex(order)
        say("shift probe, BMABZ detection rate by shift (mean over seeds):\n" + m.round(2).to_string())
        drop = pd.DataFrame({"drop at -0.5 Hz": (curve[0.0] - curve[-0.5]) * 100, "drop at -1 Hz": (curve[0.0] - curve[-1.0]) * 100})
        say("drops in points, per seed:\n" + drop.round(1).to_string() + "\nmean per rung:\n"
            + drop.groupby("rung").mean().reindex(order).round(1).to_string())
        lv = pd.concat([pr[pr.probe == "level"], pr[pr.probe == "base"].assign(value=0.0)])
        lvt = lv.groupby(["rung", "seed", "cls", "value"]).detected.mean().unstack("value")
        lvt.rename(columns=str).to_parquet(TABLES / "ablate_level.parquet")
        ld = ((lvt[0.0] - lvt[-20.0]) * 100).unstack("cls")
        say("level probe, detection lost at -20 dB in points (mean over seeds):\n"
            + ld.groupby("rung").mean().reindex(order).round(1).to_string()
            + "\nlost at +20 dB:\n" + ((lvt[0.0] - lvt[20.0]) * 100).unstack("cls").groupby("rung").mean().reindex(order).round(1).to_string())
    if evr:
        e = pd.DataFrame(evr)
        e.to_parquet(TABLES / "ablate_eval.parquet")
        piv = e.groupby(["rung", "cls"]).per_hour.mean().unstack("cls")
        say("detections per hour on the original Kerguelen 2019/20 audio (mean over seeds and the two sites):\n" + piv.round(2).to_string())
        if {"D", "AUG"} <= set(e.rung):
            say("AUG vs D, change in %:\n" + ((piv.loc["AUG"] / piv.loc["D"] - 1) * 100).round(1).to_string())


def list_(a) -> None:
    for rung in RUNGS:
        print(rung, " ".join(f"{s}:{'yes' if ckpt(rung, s) else '-'}" for s in SEEDS))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["list", "run", "report", "all"])
    p.add_argument("--only", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS, choices=SEEDS)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    for s in (["run", "report"] if a.step == "all" else [a.step]):
        {"list": list_, "run": run, "report": report}[s](a)


if __name__ == "__main__":
    main()
