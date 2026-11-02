"""Pre-registered seed tests S1 to S6 (logs/preregistration.md, 22.10.2026): all five NAVE checkpoints
through the same analyses. Seed 666 is the released checkpoint every earlier result used; seeds 0, 1, 42
and 1337 were not looked at before the pre-registration.

    python -m aivis.phase1.seeds run [--seeds 0 1 ...]   # model runs, resumable, one seed per GPU if you like
    python -m aivis.phase1.seeds report                  # tables seeds_scores, seeds_f1, seeds_probes, seeds_channels

Per seed: dev sites with the published 2 s pipeline and with centre stitching (30 s overlap), the
frequency-shifted Kerguelen 2014/15 audio of follow-up F1, and the 60 s probe clips. Thresholds are tuned
for each seed the same way as in the main run (LOSO over the dev sites, IoU 0.3).
Needs the main run, the probe clips (aivis.phase1.probes make) and the shifted audio (aivis.phase1.followups make).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import paths, predict
from . import build, core, followups, probes
from .core import CLASSES, TABLES

CKPT_DIR = paths.ROOT / "checkpoints" / "nave_seeds"
SEEDS = {0: "04cd78a429671fe5", 1: "a993b0100c5d6858", 42: "916afcbf33ea3484", 666: "45c4466447a7c2e0",
         1337: "332240e4bd730771"}          # sha256 prefixes
RELEASED = 666
ROOT = core.ROOT / "seeds"
VAL = paths.SITES["validation"]
SHIFTED = ("kerguelen2014", "kerguelen2015")


def say(m):
    print(f"[seeds] {m}", flush=True)


def ckpt(seed: int) -> Path:
    p = CKPT_DIR / f"nave_k153_seed{seed}.pt"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if not h.startswith(SEEDS[seed]):
        raise SystemExit(f"{p} has sha256 {h[:16]}, expected {SEEDS[seed]}")
    return p


def run_dirs(seed: int) -> dict[str, Path]:
    r = ROOT / f"seed{seed}"
    return {"2s": r / "val_2s", "centre": r / "val_centre", "shift": r / "shift", "probe": r / "probe60"}


def go(args: list[str], out: Path) -> None:
    if (out / "run.json").exists():
        return
    say(f"run: {out.relative_to(core.ROOT)}")
    predict.main([*args, "--out", str(out), "--overwrite", "--tag", "seeds"])


def run(a) -> None:
    for seed in a.seeds:
        w = str(ckpt(seed))
        d = run_dirs(seed)
        base = ["nave", "--weights", w, "--device", a.device]
        for site in VAL:
            go([*base, "--split", "validation", "--sites", site], d["2s"] / site)
            go([*base, "--split", "validation", "--sites", site, "--overlap-s", "30", "--stitch", "centre"], d["centre"] / site)
        for site in SHIFTED:
            go([*base, "--audio-dir", str(followups.ROOT / "audio" / site), "--dataset", site], d["shift"] / site)
        go([*base, "--audio-dir", str(probes.CLIPS / "probe60"), "--dataset", "probe60"], d["probe"])
    say("run: done")


# ------------------------------------------------------------------------------ report

def grid_events(prob_dir: Path, site: str) -> pd.DataFrame:
    parts = []
    for f in sorted(prob_dir.glob("*.npz")):
        with np.load(f) as z:
            sm = core.smooth(z["probs"].astype(np.float32))
        for ci, c in enumerate(CLASSES):
            for thr in core.GRID:
                s, e, cf = core.runs_to_events(sm[:, ci], thr)
                if len(s):
                    parts.append(pd.DataFrame({"dataset": site, "filename": f.stem + ".wav", "cls": c, "cls_order": ci,
                                               "thr": thr, "start_s": s, "end_s": e, "conf": cf}))
    ev = pd.concat(parts, ignore_index=True).sort_values(["filename", "cls_order", "start_s"], kind="stable")
    return core.add_abs(ev.reset_index(drop=True))


def at(ev: pd.DataFrame, thr: list[float]) -> pd.DataFrame:
    parts = [ev[(ev.cls == c) & np.isclose(ev.thr, t)] for c, t in zip(CLASSES, thr)]
    return pd.concat(parts).sort_values(["filename", "cls_order", "start_s"], kind="stable").reset_index(drop=True)


def sweep(evs: dict[str, pd.DataFrame], gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, ev in evs.items():
        g = gt[gt.dataset == site]
        for c in CLASSES:
            gc = g[g.cls == c].reset_index(drop=True)
            ec = ev[ev.cls == c]
            for thr in core.GRID:
                p = ec[np.isclose(ec.thr, thr)].reset_index(drop=True)
                pm, gm = core.official_match(p, gc, 0.3)
                rows.append({"site": site, "cls": c, "thr": float(thr), "tp": int((gm >= 0).sum()),
                             "fn": int((gm < 0).sum()), "fp": int((pm < 0).sum())})
    return pd.DataFrame(rows)


def tune(sw: pd.DataFrame) -> tuple[dict, list]:
    loso = {s: [build.best_thr(sw[(sw.cls == c) & (sw.site != s)]) for c in CLASSES] for s in VAL}
    insample = [build.best_thr(sw[sw.cls == c]) for c in CLASSES]
    return loso, insample


def probe_rows(seed: int, thr: list[float]) -> list[dict]:
    man = pd.read_parquet(probes.PROBE_ROOT / "manifest.parquet")
    man = man[(man.dataset == "probe60") & man.probe.isin(["base", "level", "shift", "transient"])]
    fp = run_dirs(seed)["probe"] / "frame_probs" / "probe60"
    t = dict(zip(CLASSES, thr))
    rows = []
    for r in man.itertuples():
        with np.load(fp / r.filename.replace(".wav", ".npz")) as z:
            sm = core.smooth(z["probs"].astype(np.float32))
        ci = CLASSES.index(r.cls)
        f0 = int(max(0, np.floor(r.call_t0 / core.HOP_S)))
        f1 = int(min(len(sm), np.ceil(r.call_t1 / core.HOP_S)))
        s, e, _ = core.runs_to_events(sm[:, ci], float(core.GRID[np.argmin(np.abs(core.GRID - t[r.cls]))]))
        inter = np.clip(np.minimum(r.call_t1, e) - np.maximum(r.call_t0, s), 0, None)
        iou = inter / ((r.call_t1 - r.call_t0) + (e - s) - inter) if len(s) else np.zeros(0)
        rows.append({"seed": seed, "call_id": r.call_id, "cls": r.cls, "probe": r.probe, "value": r.value,
                     "offset": getattr(r, "offset", np.nan), "peak": float(sm[f0:max(f1, f0 + 1), ci].max()),
                     "detected": bool((iou > 0.3).any())})
    return rows


def channel_test(seed: int, device: str) -> list[dict]:
    """+20 dB given to all channels, only to magnitude and phase, or only to PCEN (36 dev calls)."""
    from ..models import nave
    model, spec, _ = nave.load(ckpt(seed), device)
    calls = pd.read_parquet(TABLES / "labels_validation.parquet")
    rows = []
    for c in CLASSES:
        g = calls[(calls.cls == c) & (calls.snr_db > 4) & (calls.duration < 20)]
        g = g[((g.start_s + g.end_s) / 2 > 40) & ((g.start_s + g.end_s) / 2 < 3500)].sample(12, random_state=1)
        ci = CLASSES.index(c)
        for r in g.itertuples():
            ctr = (r.start_s + r.end_s) / 2
            x = torch.from_numpy(probes.read("validation", r.dataset, r.filename, ctr - 30, 60).astype(np.float32))[None].to(device)
            f0, f1 = int((30 - r.duration / 2) / 0.02), int((30 + r.duration / 2) / 0.02)
            with torch.no_grad():
                F0, Fg = spec(x), spec(x * 10.0)
                for name, feat in (("original", F0), ("all channels", Fg),
                                   ("magnitude and phase only", torch.cat([Fg[:, :3], F0[:, 3:]], 1)),
                                   ("PCEN only", torch.cat([F0[:, :3], Fg[:, 3:]], 1))):
                    p = torch.sigmoid(model(feat))[0, :, ci].float().cpu().numpy()
                    rows.append({"seed": seed, "cls": c, "label_id": r.label_id, "condition": name, "peak": float(p[f0:f1].max())})
    return rows


def report(a) -> None:
    gt = core.load_labels("validation")
    score_rows, f1_rows, probe_all, chan_all = [], [], [], []
    for seed in a.seeds:
        d = run_dirs(seed)
        if not all((d[k] / s / "run.json").exists() for k in ("2s", "centre") for s in VAL):
            say(f"report: seed {seed} not complete, skipped")
            continue
        res = {}
        for kind in ("2s", "centre"):
            evs = {s: grid_events(d[kind] / s / "frame_probs" / s, s) for s in VAL}
            sw = sweep(evs, gt)
            loso, ins = tune(sw)
            allev = pd.concat([at(evs[s], loso[s]) for s in VAL], ignore_index=True)
            _, mac = core.score(allev, gt)
            res[kind] = {"evs": evs, "loso": loso, "insample": ins}
            for r in mac.itertuples():
                score_rows.append({"seed": seed, "pipeline": kind, "dataset": r.dataset, "macro_B": r.macro_B, "macro_A": r.macro_A})
        say(f"report: seed {seed} scored")
        loso = res["2s"]["loso"]
        for site in SHIFTED:
            if not (d["shift"] / site / "run.json").exists():
                continue
            g = gt[gt.dataset == site].reset_index(drop=True)
            shifted = grid_events(d["shift"] / site / "frame_probs" / site, site)
            for cond, ev in (("original", at(res["2s"]["evs"][site], loso[site])), ("shift -0.6 Hz", at(shifted, loso[site]))):
                for c in CLASSES:
                    gc = g[g.cls == c].reset_index(drop=True)
                    _, gm = core.official_match(ev[ev.cls == c].reset_index(drop=True), gc, 0.3)
                    f1_rows.append({"seed": seed, "site": site, "condition": cond, "cls": c, "tp": int((gm >= 0).sum()), "n": len(gc)})
        if (d["probe"] / "run.json").exists():
            probe_all += probe_rows(seed, res["2s"]["insample"])
        chan_all += channel_test(seed, a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        say(f"report: seed {seed} done")

    sc = pd.DataFrame(score_rows)
    sc.to_parquet(TABLES / "seeds_scores.parquet")
    j = sc[sc.dataset == "joined"].pivot_table(index="seed", columns="pipeline", values="macro_B")
    j["gain"] = j["centre"] - j["2s"]
    say("S1/S4, LOSO macro B (dev joined) per seed:\n" + j.round(3).to_string()
        + f"\n  new seeds mean 2 s {j.drop(RELEASED, errors='ignore')['2s'].mean():.3f} sd {j.drop(RELEASED, errors='ignore')['2s'].std():.3f}, "
          f"mean gain {j.drop(RELEASED, errors='ignore')['gain'].mean():+.3f}")
    if f1_rows:
        f1 = pd.DataFrame(f1_rows)
        f1.to_parquet(TABLES / "seeds_f1.parquet")
        r = f1.groupby(["seed", "cls", "condition"])[["tp", "n"]].sum()
        rec = (r.tp / r.n).unstack()
        rec["change_points"] = (rec["shift -0.6 Hz"] - rec["original"]) * 100
        say("S2, recall on Kerguelen 2014/15, original vs shifted -0.6 Hz:\n" + rec.round(3).unstack("cls").to_string())
    if probe_all:
        pr = pd.DataFrame(probe_all)
        pr.to_parquet(TABLES / "seeds_probes.parquet")
        base = pr[pr.probe == "base"].groupby(["seed", "cls"]).detected.mean()
        sh = pr[pr.probe == "shift"].pivot_table(index=["seed", "cls"], columns="value", values="detected")
        say("S3, shift probe detection at -0.5 / 0 Hz and drop in points:\n"
            + pd.DataFrame({"-0.5 Hz": sh[-0.5], "0 Hz": sh[0.0], "drop": (sh[0.0] - sh[-0.5]) * 100}).round(2).to_string())
        lv = pr[pr.probe == "level"].pivot_table(index=["seed", "cls"], columns="value", values="detected")
        say("S5, level probe detection at +20 dB vs base, drop in points:\n"
            + pd.DataFrame({"base": base, "+20 dB": lv[20.0], "drop": (base - lv[20.0]) * 100}).round(2).to_string())
        tr = pr[(pr.probe == "transient") & (pr.value == 30)].groupby(["seed", "cls"]).detected.mean()
        say("S6, +30 dB burst, drop in points:\n" + pd.DataFrame({"base": base, "burst": tr, "drop": (base - tr) * 100}).round(2).to_string())
    if chan_all:
        ch = pd.DataFrame(chan_all)
        ch.to_parquet(TABLES / "seeds_channels.parquet")
        m = ch.pivot_table(index="seed", columns="condition", values="peak")
        m["PCEN-only change"] = m["PCEN only"] - m["original"]
        m["magnitude share of drop"] = (m["original"] - m["magnitude and phase only"]) / (m["original"] - m["all channels"])
        say("S5, channel test (mean peak over 36 calls):\n" + m.round(3).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["run", "report", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=list(SEEDS))
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    for s in (["run", "report"] if a.step == "all" else [a.step]):
        {"run": run, "report": report}[s](a)


if __name__ == "__main__":
    main()
