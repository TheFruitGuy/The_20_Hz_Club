"""Build the Phase I tables from the exports of predict_all. Every step skips what is already there.

    python -m aivis.phase1.build all            # everything that the finished predictions allow
    python -m aivis.phase1.build labels         # label table with SNR (validation)
    python -m aivis.phase1.build sitestats      # level and noise per recording (validation + evaluation)
    python -m aivis.phase1.build grid           # frame models: events at 19 thresholds, peak score per label
    python -m aivis.phase1.build sweep          # tp/fp/fn per model, site, class, threshold, IoU
    python -m aivis.phase1.build thresholds     # default, in-sample and leave-one-site-out thresholds
    python -m aivis.phase1.build events         # the events at those thresholds, matched (official rules)

Tables go to outputs/phase1/tables/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import soundfile as sf

from .. import paths
from . import core
from .core import CLASSES, GRID, MODELS, TABLES, YOLO_GRID

VAL_SITES = paths.SITES["validation"]
EVAL_SITES = paths.SITES["evaluation"]
IOUS = (0.1, 0.3, 0.5)


def say(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def site_done(model: str, site: str) -> bool:
    return (core.ROOT / model / site / "run.json").exists()


# ----------------------------------------------------------------------------- audio

NFFT, HOP = 256, 25          # 1.024 s window, 0.1 s hop
FREQS = np.arange(NFFT // 2 + 1) * 250 / NFFT


def power_spec(x: np.ndarray) -> np.ndarray:
    """(freq, frames) power, frame t centred at t*0.1 + 0.512 s."""
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    n = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(NFFT)[None, :]
    return (np.abs(np.fft.rfft(frames, axis=1)) ** 2).T.astype(np.float32)


def label_snr(gt_file: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """SNR of each label: mean power in its frequency band during the label, against the median band
    power in the 30 s before and after (label-free frames only, 1 s margin)."""
    n = P.shape[1]
    centre = np.arange(n) * HOP / 250 + NFFT / 500
    busy = np.zeros(n, bool)
    for s, e in zip(gt_file["start_s"], gt_file["end_s"]):
        busy |= (centre >= s - 1) & (centre <= e + 1)
    out = []
    for r in gt_file.itertuples():
        lo, hi = max(r.low_frequency, 1), min(r.high_frequency, 124)
        band = (FREQS >= lo) & (FREQS <= hi)
        if not band.any():
            band[np.argmin(np.abs(FREQS - (lo + hi) / 2))] = True
        bp = P[band].mean(axis=0)
        inside = (centre >= r.start_s) & (centre <= r.end_s)
        if not inside.any():
            inside[np.clip(np.searchsorted(centre, r.start_s), 0, n - 1)] = True
        sig = bp[inside].mean()
        for w in (30, 120):
            ctx = (centre >= r.start_s - w) & (centre <= r.end_s + w) & ~busy
            if ctx.sum() >= 20:
                break
        noise = np.median(bp[ctx]) if ctx.any() else np.nan
        out.append({"label_id": r.label_id, "snr_db": 10 * np.log10(sig / noise) if noise > 0 else np.nan,
                    "band_level_db": 10 * np.log10(sig + 1e-20), "noise_db": 10 * np.log10(noise + 1e-20),
                    "peak_freq": float(FREQS[band][np.argmax(P[band][:, inside].mean(axis=1))])})
    return pd.DataFrame(out)


def step_labels(a) -> None:
    out = TABLES / "labels_validation.parquet"
    if out.exists() and not a.force:
        say("labels: done")
        return
    gt = core.load_labels("validation")
    parts_dir = TABLES / "_snr"
    parts_dir.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + a.budget_s
    for site in VAL_SITES:
        part = parts_dir / f"{site}.parquet"
        if part.exists():
            continue
        g = gt[gt["dataset"] == site]
        rows = []
        tmp = parts_dir / f"{site}.partial.parquet"
        done_files = set(pd.read_parquet(tmp)["filename"]) if tmp.exists() else set()
        prev = [pd.read_parquet(tmp)] if tmp.exists() else []
        for fn, gf in g.groupby("filename"):
            if fn in done_files:
                continue
            x, _ = sf.read(str(paths.audio_dir("validation") / site / fn), dtype="float32", always_2d=True)
            r = label_snr(gf, power_spec(x[:, 0]))
            r["filename"] = fn
            rows.append(r)
            if time.time() > t_end:
                pd.concat(prev + rows).to_parquet(tmp)
                say(f"labels: {site} paused (time budget), run again to continue")
                return
        pd.concat(prev + rows).to_parquet(part)
        tmp.unlink(missing_ok=True)
        say(f"labels: SNR for {site}")
    snr = pd.concat([pd.read_parquet(parts_dir / f"{s}.parquet") for s in VAL_SITES]).drop(columns="filename")
    gt = gt.merge(snr, on="label_id", how="left")
    gt.to_parquet(out)
    say(f"labels: {len(gt)} labels -> {out}")


def step_sitestats(a) -> None:
    """Per recording: broadband RMS level and median noise level in three bands (for C5 and C9)."""
    out = TABLES / "site_stats.parquet"
    if out.exists() and not a.force:
        say("sitestats: done")
        return
    parts_dir = TABLES / "_sitestats"
    parts_dir.mkdir(parents=True, exist_ok=True)
    bands = {"5-30Hz": (5, 30), "30-60Hz": (30, 60), "60-120Hz": (60, 120)}
    t_end = time.time() + a.budget_s
    for split, sites in (("validation", VAL_SITES), ("evaluation", EVAL_SITES)):
        for site in sites:
            part = parts_dir / f"{site}.parquet"
            if part.exists():
                continue
            tmp = parts_dir / f"{site}.partial.parquet"
            prev = pd.read_parquet(tmp) if tmp.exists() else pd.DataFrame(columns=["filename"])
            rows = []
            for w in sorted((paths.audio_dir(split) / site).glob("*.wav")):
                if w.name in set(prev["filename"]):
                    continue
                x, _ = sf.read(str(w), dtype="float32", always_2d=True)
                x = x[:, 0]
                P = power_spec(x[::1])
                row = {"dataset": site, "split": split, "filename": w.name, "duration_s": len(x) / 250,
                       "rms_db": 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12),
                       "month": core.file_start(w.name).month if hasattr(core, "file_start") else None}
                for name, (lo, hi) in bands.items():
                    bp = P[(FREQS >= lo) & (FREQS < hi)].mean(axis=0)
                    row[f"noise_{name}_db"] = 10 * np.log10(np.median(bp) + 1e-20)
                    row[f"p95_{name}_db"] = 10 * np.log10(np.percentile(bp, 95) + 1e-20)
                rows.append(row)
                if time.time() > t_end:
                    pd.concat([prev, pd.DataFrame(rows)]).to_parquet(tmp)
                    say(f"sitestats: {site} paused (time budget), run again to continue")
                    return
            pd.concat([prev, pd.DataFrame(rows)]).to_parquet(part)
            tmp.unlink(missing_ok=True)
            say(f"sitestats: {site}")
    st = pd.concat([pd.read_parquet(parts_dir / f"{s}.parquet") for s in VAL_SITES + EVAL_SITES], ignore_index=True)
    st.to_parquet(out)
    say(f"sitestats -> {out}")


# ------------------------------------------------------------------------------ grid

def label_peaks_frames(gt_file: pd.DataFrame, sm: np.ndarray) -> list[dict]:
    rows = []
    n = len(sm)
    for r in gt_file.itertuples():
        f0 = int(np.clip(np.floor(r.start_s / core.HOP_S), 0, n - 1))
        f1 = int(np.clip(np.ceil(r.end_s / core.HOP_S), f0 + 1, n))
        seg = sm[f0:f1]
        c = CLASSES.index(r.cls)
        rows.append({"label_id": r.label_id, "peak": float(seg[:, c].max()), "mean": float(seg[:, c].mean()),
                     **{f"peak_{k}": float(seg[:, i].max()) for i, k in enumerate(CLASSES)}})
    return rows


def step_grid(a) -> None:
    gt = core.load_labels("validation")
    for model in ("whalevad", "nave"):
        for site in VAL_SITES + EVAL_SITES:
            out = core.grid_path(model, site)
            peaks_out = TABLES / "peaks" / f"{model}_{site}.parquet"
            if out.exists() and not a.force:
                continue
            if not site_done(model, site):
                continue
            t0 = time.time()
            ev_parts, peak_rows = [], []
            g_site = gt[gt["dataset"] == site]
            by_file = dict(tuple(g_site.groupby("filename")))
            for fn, sm in core.iter_frame_probs(model, site):
                for ci, c in enumerate(CLASSES):
                    for thr in GRID:
                        s, e, conf = core.runs_to_events(sm[:, ci], thr)
                        if len(s):
                            ev_parts.append(pd.DataFrame({"filename": fn, "cls": c, "cls_order": ci, "thr": thr,
                                                          "start_s": s, "end_s": e, "conf": conf}))
                if fn in by_file:
                    peak_rows += label_peaks_frames(by_file[fn], sm)
            ev = pd.concat(ev_parts, ignore_index=True)
            ev.insert(0, "dataset", site)
            out.parent.mkdir(parents=True, exist_ok=True)
            ev.to_parquet(out)
            if peak_rows:
                peaks_out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(peak_rows).assign(model=model).to_parquet(peaks_out)
            say(f"grid: {model} {site}, {len(ev)} events over {len(GRID)} thresholds, {time.time() - t0:.0f} s")
            if time.time() - a.t_start > a.budget_s:
                say("grid: paused (time budget), run again to continue")
                return
    # YOLO peaks: highest confidence of a same-class box overlapping the label with IoU > 0.1
    for site in VAL_SITES:
        peaks_out = TABLES / "peaks" / f"yolo_{site}.parquet"
        if peaks_out.exists() and not a.force or not site_done("yolo", site):
            continue
        ev = core.add_abs(core.yolo_events(core.yolo_boxes(site)))
        g = gt[gt["dataset"] == site]
        rows = []
        for (fn, c), gg in g.groupby(["filename", "cls"]):
            e = ev[(ev.filename == fn) & (ev.cls == c)]
            for r in gg.itertuples():
                inter = np.clip(np.minimum(r.t1, e.t1.to_numpy()) - np.maximum(r.t0, e.t0.to_numpy()), 0, None)
                union = (r.t1 - r.t0) + (e.t1 - e.t0).to_numpy() - inter
                ok = inter / np.maximum(union, 1e-9) > 0.1
                rows.append({"label_id": r.label_id, "peak": float(e.conf.to_numpy()[ok].max()) if ok.any() else 0.0})
        peaks_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).assign(model="yolo").to_parquet(peaks_out)
        say(f"grid: yolo peaks {site}")


# ----------------------------------------------------------------------------- sweep

def model_grid(model: str):
    return YOLO_GRID if model == "yolo" else GRID


def step_sweep(a) -> None:
    out = TABLES / "sweep.parquet"
    gt = core.load_labels("validation")
    have = pd.read_parquet(out) if out.exists() and not a.force else pd.DataFrame(columns=["model", "site"])
    done = set(zip(have["model"], have["site"]))
    new = []
    for model in MODELS:
        for site in VAL_SITES:
            if (model, site) in done or not site_done(model, site):
                continue
            if model != "yolo" and not core.grid_path(model, site).exists():
                continue
            t0 = time.time()
            g = gt[gt["dataset"] == site].reset_index(drop=True)
            if model == "yolo":
                allev = core.add_abs(core.yolo_events(core.yolo_boxes(site)))
                allev["thr"] = np.nan
            else:
                allev = core.add_abs(pd.read_parquet(core.grid_path(model, site)).sort_values(
                    ["filename", "cls_order", "start_s"], kind="stable"))
            for c in CLASSES:
                gc = g[g["cls"] == c].reset_index(drop=True)
                ec = allev[allev["cls"] == c]
                for thr in model_grid(model):
                    if model == "yolo":
                        p = ec[ec["conf"] >= thr].reset_index(drop=True)
                    else:
                        p = ec[np.isclose(ec["thr"], thr)].reset_index(drop=True)
                    for iou in IOUS:
                        pm, gm = core.official_match(p, gc, iou)
                        new.append({"model": model, "site": site, "cls": c, "thr": float(thr), "iou": iou,
                                    "tp": int((gm >= 0).sum()), "fn": int((gm < 0).sum()), "fp": int((pm < 0).sum()),
                                    "n_pred": len(p)})
            say(f"sweep: {model} {site} in {time.time() - t0:.0f} s")
            pd.concat([have, pd.DataFrame(new)], ignore_index=True).to_parquet(out)
            if time.time() - a.t_start > a.budget_s:
                say("sweep: paused (time budget), run again to continue")
                return


# ------------------------------------------------------------------------ thresholds

def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def best_thr(sw: pd.DataFrame) -> float:
    """Threshold with the highest pooled F1 (ties: the lower threshold)."""
    s = sw.groupby("thr")[["tp", "fp", "fn"]].sum()
    v = f1(s.tp.to_numpy(), s.fp.to_numpy(), s.fn.to_numpy())
    return float(s.index[int(np.argmax(v))])


def nave_default() -> list[float]:
    info = json.loads((core.ROOT / "nave" / VAL_SITES[0] / "run.json").read_text())
    t = info["settings"]["thresholds"]
    return [float(GRID[np.argmin(np.abs(GRID - t[c]))]) for c in CLASSES]


def step_thresholds(a) -> None:
    sw = pd.read_parquet(TABLES / "sweep.parquet")
    res = {}
    for model in MODELS:
        m = sw[sw.model == model]
        if set(m.site) != set(VAL_SITES):
            say(f"thresholds: {model} not complete yet")
            continue
        res[model] = {"default": DEFAULT_THR.get(model) if model != "nave" else nave_default()}
        for iou in IOUS:
            mi = m[m.iou == iou]
            res[model][f"insample_iou{iou}"] = [best_thr(mi[mi.cls == c]) for c in CLASSES]
            res[model][f"loso_iou{iou}"] = {s: [best_thr(mi[(mi.cls == c) & (mi.site != s)]) for c in CLASSES]
                                            for s in VAL_SITES}
        say(f"thresholds {model}: default {res[model]['default']}, in-sample {res[model]['insample_iou0.3']}, "
            f"LOSO {res[model]['loso_iou0.3']}")
    (TABLES / "thresholds.json").write_text(json.dumps(res, indent=2))


DEFAULT_THR = core.DEFAULT_THR


def thresholds() -> dict:
    return json.loads((TABLES / "thresholds.json").read_text())


def thr_for(model: str, site: str, variant: str = "loso", iou: float = 0.3) -> list[float]:
    t = thresholds()[model]
    if variant == "default":
        return t["default"]
    if variant == "insample":
        return t[f"insample_iou{iou}"]
    if site in VAL_SITES:
        return t[f"loso_iou{iou}"][site]
    return t[f"insample_iou{iou}"]       # evaluation sites: tuned on all three dev sites


# ---------------------------------------------------------------------------- events

def events_path(model: str, variant: str = "loso") -> object:
    return TABLES / f"events_{model}_{variant}.parquet"


def step_events(a) -> None:
    """Events at default, in-sample and LOSO thresholds (IoU 0.3) for all finished sites, matched with the
    official rules on the dev sites. Evaluation sites use the in-sample (all dev) thresholds for 'loso'."""
    gt = core.load_labels("validation")
    th = thresholds()
    for model in th:
        for variant in ("default", "insample", "loso"):
            out = events_path(model, variant)
            sites = [s for s in VAL_SITES + EVAL_SITES if site_done(model, s)
                     and (model == "yolo" or core.grid_path(model, s).exists())]
            if out.exists() and not a.force and set(pd.read_parquet(out, columns=["dataset"]).dataset.unique()) >= set(sites):
                continue
            parts = []
            for site in sites:
                ev = core.events_at(model, site, thr_for(model, site, variant))
                ev["gt_match"] = -1
                if site in VAL_SITES:
                    g = gt[gt["dataset"] == site]
                    pm, _ = core.official_match(ev, g.reset_index(drop=True), 0.3)
                    ev["gt_match"] = np.where(pm >= 0, g["label_id"].to_numpy()[np.maximum(pm, 0)], -1)
                parts.append(ev)
            ev = pd.concat(parts, ignore_index=True)
            ev.insert(0, "event_id", np.arange(len(ev)))
            ev.to_parquet(out)
            say(f"events: {model} {variant}: {len(ev)} events on {len(sites)} sites")


STEPS = {"labels": step_labels, "sitestats": step_sitestats, "grid": step_grid, "sweep": step_sweep,
         "thresholds": step_thresholds, "events": step_events}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["all", *STEPS])
    p.add_argument("--force", action="store_true", help="redo even if the output exists")
    p.add_argument("--budget-s", type=float, default=1e9, help="stop after about this many seconds (resumable)")
    a = p.parse_args(argv)
    a.t_start = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in (STEPS if a.step == "all" else [a.step]):
        STEPS[name](a)
        if time.time() - a.t_start > a.budget_s:
            say("time budget used up, run again to continue")
            return


if __name__ == "__main__":
    main()
