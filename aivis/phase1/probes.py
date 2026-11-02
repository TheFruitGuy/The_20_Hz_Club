"""Ideas 3 and 4: controlled probes. Real dev calls are changed in one way at a time, then every model
runs its normal pipeline (windowing included) on the result.

    python -m aivis.phase1.probes make      # pick calls and backgrounds, write the probe clips
    python -m aivis.phase1.probes run       # run the three models on them (resumable)
    python -m aivis.phase1.probes collect   # peak score and detection per clip and model

Calls: per class, isolated dev labels (no other label within 10 s) with a clear SNR, spread over the
three dev sites. Backgrounds: 60 s stretches of every dev and evaluation site in which no label lies
and no model fires. Clips are 60 s long with the call centred at 37.5 s, a position that is inside a
window or tile of every model and away from all their edges (the position probe uses 120 s clips).

Probes (C = hypotheses of the plan)
  base        the original 60 s around the call
  level       whole clip scaled by -20 to +20 dB in 5 dB steps                         C5
  transient   0.5 s broadband burst 10, 20 or 30 s before or 10, 20 s after the call,
              10, 20 or 30 dB above the call's band power                             C4, C8
  shift       call cut out (band-pass), shifted by -3 to +3 Hz (single sideband), pasted
              into a call-free background of its own site at its own SNR             C6
  truncation  25, 50 or 75 % of the cut-out call, the start or the end kept           C2
  context     original audio kept only within +-1, 2, 5, 10, 20 s of the call, the rest
              replaced by call-free background of the same site                       C7
  density     BP only: the pulse alone vs four pulses 13 s apart                       C7
  bgswap      the cut-out call at its own SNR in background from each of the 7 sites  C9
  snr         the cut-out call at -12 to +12 dB (3 dB steps) in 4 backgrounds
              (casey2017, kerguelen2015, ddu2021, kerguelen2020)                       C4, C5, H4
  position    120 s of original audio with the call centre at 10 positions, across
              YOLO's 15 s window borders and the tile edges of NAVE and Whale-VAD     C8
SNR here is the band power of the call itself over the median band power of the background, so
0 dB means call and noise equally strong in the call's band (the label table's snr_db is
(call + noise) / noise; the conversion is in snr_to_call_ratio).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, hilbert, sosfiltfilt

from .. import paths
from . import build, core
from .core import CLASSES, MODELS, TABLES

SR = 250
PROBE_ROOT = core.ROOT / "probes"
CLIPS = PROBE_ROOT / "clips"
CLIP_S, CENTRE_S = 60.0, 37.5
POS_CLIP_S = 120.0
POSITIONS = [20, 27, 30, 33, 45, 55, 58, 60, 62, 75]     # call centre in the 120 s clip
LEVELS_DB = [-20, -15, -10, -5, 5, 10, 15, 20]
SHIFTS_HZ = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
TRANSIENT_OFFSETS = [-30, -20, -10, 10, 20]
TRANSIENT_DB = [10, 20, 30]
TRUNC = [0.25, 0.5, 0.75]
CONTEXT_S = [1, 2, 5, 10, 20]
SNR_DB = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
SNR_BG_SITES = ["casey2017", "kerguelen2015", "ddu2021", "kerguelen2020"]
BG_SITES = paths.SITES["validation"] + paths.SITES["evaluation"]
BASE_TIME = datetime(2000, 1, 1)


def say(msg):
    print(f"[probes] {msg}", flush=True)


def read(split: str, site: str, filename: str, t0: float, dur: float) -> np.ndarray:
    with sf.SoundFile(str(paths.audio_dir(split) / site / filename)) as f:
        f.seek(max(0, int(round(t0 * SR))))
        x = f.read(int(round(dur * SR)), dtype="float32", always_2d=True)[:, 0]
    if len(x) < int(round(dur * SR)):
        x = np.pad(x, (0, int(round(dur * SR)) - len(x)))
    return x.astype(np.float64)


def band_power(x: np.ndarray, lo: float, hi: float) -> float:
    """Mean power spectral density of x in [lo, hi] Hz (Hann window, independent of the length)."""
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    sel = (f >= lo) & (f <= hi)
    if not sel.any():
        sel[np.argmin(np.abs(f - (lo + hi) / 2))] = True
    return float(np.mean(np.abs(X[sel]) ** 2) / np.sum(w ** 2))


def band_noise(x: np.ndarray, lo: float, hi: float) -> float:
    """Median band power over 1 s frames (robust noise level of a background)."""
    n = SR
    parts = [band_power(x[i:i + n], lo, hi) for i in range(0, len(x) - n + 1, n // 2)]
    return float(np.median(parts))


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo, hi = max(lo - 2, 2.0), min(hi + 2, 123.0)
    sos = butter(6, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, x)


def taper(n: int, ramp: int) -> np.ndarray:
    w = np.ones(n)
    r = min(ramp, n // 2)
    if r > 0:
        w[:r] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))
        w[-r:] = w[:r][::-1]
    return w


def ssb_shift(x: np.ndarray, hz: float) -> np.ndarray:
    if hz == 0:
        return x
    t = np.arange(len(x)) / SR
    return np.real(hilbert(x) * np.exp(2j * np.pi * hz * t))


def snr_to_call_ratio(snr_db: float) -> float:
    """Label-table SNR ((call + noise) / noise) -> call / noise power ratio."""
    return max(10 ** (snr_db / 10) - 1, 0.1)


# ------------------------------------------------------------------------------- make

def pick_calls(n_per_class: int, seed: int = 0) -> pd.DataFrame:
    gt = pd.read_parquet(TABLES / "labels_validation.parquet")
    gt = gt.sort_values(["dataset", "filename", "t0"])
    # isolation: no other label (any class) within 10 s
    prev_end = gt.groupby(["dataset", "filename"]).t1.shift(1)
    next_start = gt.groupby(["dataset", "filename"]).t0.shift(-1)
    # labels overlap a lot in bouts, so use the running maximum of the previous ends
    run_end = gt.groupby(["dataset", "filename"]).t1.cummax().groupby([gt.dataset, gt.filename]).shift(1)
    iso = ((gt.t0 - run_end.fillna(-1e12)) > 10) & ((next_start.fillna(1e18) - gt.t1) > 10)
    centre = (gt.start_s + gt.end_s) / 2
    ok = iso & (centre > 105) & (centre < 3600 - 105) & gt.duration.between(0.5, 25) & (gt.snr_db >= 3)
    cand = gt[ok]
    rng = np.random.default_rng(seed)
    out = []
    for c in CLASSES:
        cc = cand[cand.cls == c]
        sites = cc.dataset.unique()
        per_site = int(np.ceil(n_per_class / max(len(sites), 1)))
        picked = []
        for s in sites:
            g = cc[cc.dataset == s]
            g = g[g.snr_db >= g.snr_db.median()]        # the clearer half
            picked.append(g.sample(min(per_site, len(g)), random_state=int(rng.integers(1 << 31))))
        p = pd.concat(picked)
        out.append(p.sample(min(n_per_class, len(p)), random_state=seed))
    calls = pd.concat(out).reset_index(drop=True)
    say("calls: " + ", ".join(f"{c} {n}" for c, n in calls.cls.value_counts().items())
        + " | sites " + ", ".join(f"{s} {n}" for s, n in calls.dataset.value_counts().items()))
    return calls


def pick_backgrounds(n_per_site: int, seed: int = 0) -> pd.DataFrame:
    """60 s stretches with no label within 5 s and no event of any model (LOSO / in-sample thresholds),
    and no YOLO box at confidence 0.05 or more."""
    gt = core.load_labels("validation")
    busy = []
    for m in MODELS:
        ev = pd.read_parquet(build.events_path(m, "loso"), columns=["dataset", "filename", "start_s", "end_s"])
        busy.append(ev)
    sites = [s for s in BG_SITES if all(build.site_done(m, s) for m in MODELS)]
    for site in sites:
        b = core.yolo_events(core.yolo_boxes(site))
        busy.append(b[b.conf >= 0.05][["dataset", "filename", "start_s", "end_s"]])
    busy.append(gt[["dataset", "filename", "start_s", "end_s"]])
    busy = pd.concat(busy)
    st = pd.read_parquet(TABLES / "site_stats.parquet")
    rng = np.random.default_rng(seed)
    rows = []
    for site in sites:
        files = st[st.dataset == site]
        b_site = busy[busy.dataset == site]
        by_file = dict(tuple(b_site.groupby("filename")))
        tries = 0
        got = 0
        while got < n_per_site and tries < 5000:
            tries += 1
            f = files.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
            if f.duration_s < 200:
                continue
            t0 = float(rng.uniform(30, f.duration_s - 100))
            b = by_file.get(f.filename)
            if b is not None and ((b.start_s < t0 + 65) & (b.end_s > t0 - 5)).any():
                continue
            rows.append({"bg_id": len(rows), "dataset": site, "split": core.split_of(site), "filename": f.filename,
                         "t0": t0})
            got += 1
        say(f"backgrounds {site}: {got} (after {tries} tries)")
    return pd.DataFrame(rows)


class Writer:
    def __init__(self, folder, clip_s):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.rows = []
        self.clip_s = clip_s

    def add(self, x: np.ndarray, **meta):
        name = (BASE_TIME + timedelta(seconds=self.n * 1000)).strftime("%Y-%m-%dT%H-%M-%S") + "_000.wav"
        sf.write(str(self.folder / name), x.astype(np.float32), SR, subtype="FLOAT")
        self.rows.append({"filename": name, **meta})
        self.n += 1


def make(a) -> None:
    if (PROBE_ROOT / "manifest.parquet").exists() and not a.force:
        say("make: done")
        return
    calls = pick_calls(a.calls_per_class)
    bgs = pick_backgrounds(a.bg_per_site)
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(PROBE_ROOT / "calls.parquet")
    bgs.to_parquet(PROBE_ROOT / "backgrounds.parquet")
    rng = np.random.default_rng(1)
    w60 = Writer(CLIPS / "probe60", CLIP_S)
    w120 = Writer(CLIPS / "probe120", POS_CLIP_S)
    bg_cache = {}

    def bg(site, k=0):
        cand = bgs[bgs.dataset == site]
        r = cand.iloc[(k + int(rng.integers(len(cand)))) % len(cand)]
        key = r.bg_id
        if key not in bg_cache:
            bg_cache[key] = read(r.split, r.dataset, r.filename, r.t0, CLIP_S)
        return bg_cache[key].copy(), int(key)

    n_slots = int(CLIP_S * SR)
    t_start = time.time()
    for ci, call in enumerate(calls.itertuples()):
        lo, hi = call.low_frequency, call.high_frequency
        dur = call.end_s - call.start_s
        centre = (call.start_s + call.end_s) / 2
        c0, c1 = CENTRE_S - dur / 2, CENTRE_S + dur / 2           # call position in a 60 s clip
        meta = {"call_id": int(call.label_id), "cls": call.cls, "fine": call.fine, "site": call.dataset,
                "call_t0": c0, "call_t1": c1, "orig_snr_db": float(call.snr_db)}
        ctx = read("validation", call.dataset, call.filename, centre - CENTRE_S, CLIP_S)
        w60.add(ctx, probe="base", value=0.0, bg_site=call.dataset, **meta)
        for g in LEVELS_DB:
            w60.add(ctx * 10 ** (g / 20), probe="level", value=float(g), bg_site=call.dataset, **meta)
        # transient: decaying white-noise burst, band power relative to the call's band power in ctx
        i0, i1 = int(c0 * SR), int(c1 * SR)
        call_bp = band_power(ctx[i0:i1], lo, hi)
        for off in TRANSIENT_OFFSETS:
            for L in TRANSIENT_DB:
                burst = rng.standard_normal(int(0.5 * SR)) * np.exp(-np.linspace(0, 5, int(0.5 * SR)))
                burst *= np.sqrt(call_bp * 10 ** (L / 10) / max(band_power(burst, lo, hi), 1e-30))
                x = ctx.copy()
                s = int((CENTRE_S + off - 0.25) * SR)
                x[s:s + len(burst)] += burst
                w60.add(x, probe="transient", value=float(L), offset=float(off), bg_site=call.dataset, **meta)
        # the cut-out call
        pad = 0.5
        seg = read("validation", call.dataset, call.filename, call.start_s - pad, dur + 2 * pad)
        cut = bandpass(seg, lo, hi) * taper(len(seg), int(0.3 * SR))
        n_noise = band_noise(read("validation", call.dataset, call.filename, call.start_s - 20, 15), lo, hi)
        p_call = max(band_power(cut[int(pad * SR):-int(pad * SR) or None], lo, hi) - n_noise, 1e-30)
        ratio = snr_to_call_ratio(call.snr_db)

        def paste(background, sig, snr_ratio, at=CENTRE_S - dur / 2 - pad, nb=None):
            nb = band_noise(background, lo, hi) if nb is None else nb
            gain = np.sqrt(snr_ratio * nb / p_call)
            x = background.copy()
            s = int(round(at * SR))
            e = min(s + len(sig), len(x))
            x[s:e] += gain * sig[:e - s]
            return x

        own_bg, own_id = bg(call.dataset)
        for hz in SHIFTS_HZ:
            w60.add(paste(own_bg, ssb_shift(cut, hz), ratio), probe="shift", value=float(hz), bg_site=call.dataset,
                    bg_id=own_id, **meta)
        n_cut, n_pad = len(cut), int(pad * SR)
        core_len = n_cut - 2 * n_pad
        for frac in TRUNC:
            keep = max(int(core_len * frac), 2)
            for side in ("start", "end"):
                sig = cut.copy()
                if side == "start":     # keep the start of the call
                    sig[n_pad + keep:] *= 0
                    sig[n_pad + keep - min(25, keep):n_pad + keep] *= np.linspace(1, 0, min(25, keep))
                else:
                    sig[:n_cut - n_pad - keep] *= 0
                    k = n_cut - n_pad - keep
                    sig[k:k + min(25, keep)] *= np.linspace(0, 1, min(25, keep))
                m2 = dict(meta)
                if side == "start":
                    m2["call_t1"] = c0 + dur * frac
                else:
                    m2["call_t0"] = c1 - dur * frac
                w60.add(paste(own_bg, sig, ratio), probe=f"trunc_{side}", value=frac, bg_site=call.dataset,
                        bg_id=own_id, **m2)
        for k in CONTEXT_S:
            other, oid = bg(call.dataset, 1)
            keep = np.zeros(n_slots)
            keep[max(0, int((c0 - k) * SR)):int((c1 + k) * SR)] = 1
            keep = np.convolve(keep, np.ones(50) / 50, mode="same")      # 0.2 s cross-fade
            w60.add(ctx * keep + other * (1 - keep), probe="context", value=float(k), bg_site=call.dataset,
                    bg_id=oid, **meta)
        if call.cls == "bp":
            x = own_bg.copy()
            for off in (-26, -13, 0, 13):
                x = paste(x, cut, ratio, at=CENTRE_S - dur / 2 - pad + off, nb=band_noise(own_bg, lo, hi))
            w60.add(x, probe="density", value=4.0, bg_site=call.dataset, bg_id=own_id, **meta)
        for site in sorted(set(bgs.dataset)):
            b, bid = bg(site, 2)
            w60.add(paste(b, cut, ratio), probe="bgswap", value=0.0, bg_site=site, bg_id=bid, **meta)
        for site in [s for s in SNR_BG_SITES if s in set(bgs.dataset)]:
            b, bid = bg(site, 3)
            for snr in SNR_DB:
                w60.add(paste(b, cut, 10 ** (snr / 10)), probe="snr", value=float(snr), bg_site=site, bg_id=bid, **meta)
        for pos in POSITIONS:
            x = read("validation", call.dataset, call.filename, centre - pos, POS_CLIP_S)
            m2 = dict(meta, call_t0=pos - dur / 2, call_t1=pos + dur / 2)
            w120.add(x, probe="position", value=float(pos), bg_site=call.dataset, **m2)
        if ci % 20 == 0:
            say(f"make: {ci + 1}/{len(calls)} calls, {w60.n + w120.n} clips, {time.time() - t_start:.0f} s")
    man = pd.concat([pd.DataFrame(w60.rows).assign(dataset="probe60"),
                     pd.DataFrame(w120.rows).assign(dataset="probe120")], ignore_index=True)
    man.to_parquet(PROBE_ROOT / "manifest.parquet")
    say(f"make: {len(man)} clips ({w60.n} x 60 s, {w120.n} x 120 s) -> {CLIPS}")


# -------------------------------------------------------------------------------- run

def run(a) -> None:
    from .. import predict
    for ds in ("probe60", "probe120"):
        for m in MODELS:
            out = PROBE_ROOT / "runs" / m / ds
            if (out / "run.json").exists():
                say(f"run: {m} {ds} done")
                continue
            say(f"run: {m} {ds} ...")
            args = [m, "--audio-dir", str(CLIPS / ds), "--dataset", ds, "--out", str(out), "--overwrite",
                    "--tag", "probes", "--device", a.device]
            if m == "yolo":
                args += ["--conf", "0.01"]
            predict.main(args)


# ---------------------------------------------------------------------------- collect

def collect(a) -> None:
    man = pd.read_parquet(PROBE_ROOT / "manifest.parquet")
    th = build.thresholds()
    rows = []
    for m in MODELS:
        thr = dict(zip(CLASSES, th[m]["insample_iou0.3"]))
        for ds, mg in man.groupby("dataset"):
            out = PROBE_ROOT / "runs" / m / ds
            if not (out / "run.json").exists():
                continue
            if m == "yolo":
                ev = core.yolo_events(pd.read_parquet(out / "boxes_raw.parquet"))
                by_file = dict(tuple(ev.groupby("filename")))
            for r in mg.itertuples():
                c = r.cls
                res = {"filename": r.filename, "dataset": ds, "model": m}
                if m == "yolo":
                    e = by_file.get(r.filename, ev.iloc[:0])
                    inter = np.clip(np.minimum(r.call_t1, e.end_s) - np.maximum(r.call_t0, e.start_s), 0, None)
                    iou = inter / ((r.call_t1 - r.call_t0) + (e.end_s - e.start_s) - inter)
                    same = (e.cls == c).to_numpy()
                    ok = same & (iou.to_numpy() > 0.1)
                    res["peak"] = float(e.conf.to_numpy()[ok].max()) if ok.any() else 0.0
                    det = same & (iou.to_numpy() > 0.3) & (e.conf.to_numpy() >= thr[c])
                    res["detected"] = bool(det.any())
                    fa = (e.conf.to_numpy() >= e.cls.map(thr).to_numpy()) & (inter.to_numpy() <= 0)
                    res["false_alarms"] = int(fa.sum())
                    oth = (~same) & (iou.to_numpy() > 0.3) & (e.conf.to_numpy() >= e.cls.map(thr).to_numpy())
                    res["other_class"] = bool(oth.any())
                else:
                    with np.load(out / "frame_probs" / ds / r.filename.replace(".wav", ".npz")) as z:
                        sm = core.smooth(z["probs"].astype(np.float32))
                    f0 = int(max(0, np.floor(r.call_t0 / core.HOP_S)))
                    f1 = int(min(len(sm), np.ceil(r.call_t1 / core.HOP_S)))
                    ci = CLASSES.index(c)
                    res["peak"] = float(sm[f0:max(f1, f0 + 1), ci].max())
                    det, fa, oth = False, 0, False
                    for k, cl in enumerate(CLASSES):
                        s, e_, _ = core.runs_to_events(sm[:, k], float(core.GRID[np.argmin(np.abs(core.GRID - thr[cl]))]))
                        inter = np.clip(np.minimum(r.call_t1, e_) - np.maximum(r.call_t0, s), 0, None)
                        iou = inter / ((r.call_t1 - r.call_t0) + (e_ - s) - inter) if len(s) else np.zeros(0)
                        if cl == c:
                            det = bool((iou > 0.3).any())
                        elif (iou > 0.3).any():
                            oth = True
                        fa += int((inter <= 0).sum())
                    res.update(detected=det, false_alarms=fa, other_class=oth)
                rows.append(res)
        say(f"collect: {m}")
    out = man.merge(pd.DataFrame(rows), on=["filename", "dataset"])
    out.to_parquet(TABLES / "probe_results.parquet")
    say(f"collect: {len(out)} rows -> {TABLES / 'probe_results.parquet'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["make", "run", "collect", "all"])
    p.add_argument("--calls-per-class", type=int, default=40)
    p.add_argument("--bg-per-site", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    for step in (["make", "run", "collect"] if a.step == "all" else [a.step]):
        {"make": make, "run": run, "collect": collect}[step](a)


if __name__ == "__main__":
    main()
