"""Event decoders for NAVE's frame probabilities, scored on the five seed checkpoints (pre-registered D1 to D9).

    python -m aivis.phase1.decode run [--models nave AUG] [--seeds 0 1 ...] [--pipelines 2s centre] [--workers 10]
    python -m aivis.phase1.decode predict --models AUG --device cuda      # centre-stitched outputs (GPU)
    python -m aivis.phase1.decode smoke --prob-root outputs/nave/validation_firstlook/frame_probs

Models: nave = the five NAVE seed checkpoints (outputs/phase1/seeds); AUG, NOMAG, LOGMAG = the retrained
variants of aivis.phase1.ablate (outputs/phase1/ablate/<model>_seed<s>). Their 2 s outputs come from
`ablate run`; `predict` adds the centre-stitched ones. For the 2 s pipeline, the report also scores the
Kerguelen 2014/15 audio shifted by -0.6 Hz (frequency drift, follow-up F1) with each held-out fold's
parameters, so the drift sensitivity can be compared across decoders and models.

`run` reads the frame probabilities that `aivis.phase1.seeds run` saved (outputs/phase1/seeds/seed*/val_2s and
val_centre), so no model is run and no GPU is needed. It writes decode_scores, decode_classes, decode_fine
(.parquet), decode_params.json and decode_report.log to outputs/phase1/tables. `smoke` runs the same code on any
folder laid out as <site>/*.npz (for example the six first-look files) to check that everything works; its
scores mean nothing.

Decoders (every parameter tuned per class and per fold: chosen on the two other dev sites by pooled F1 with
the official matching at IoU 0.3, then applied to the held-out site, like the thresholds everywhere else):
    base        the published pipeline: 500 ms median filter, threshold, merge gaps up to 0.5 s, keep 0.5 to 30 s.
                Must reproduce seeds_scores (checked and printed).
    caps        base with the merge gap (0.5 / 1 / 2 s) and the minimum (0.5 to 2 s) and maximum event duration
                (4 to 30 s) tuned per class. Includes base as one grid point.
    smooth      caps with the median filter length tuned per class too (off / 0.5 / 1 / 2 s). Includes caps.
    csebb       change-point Sound Event Bounding Boxes (Ebbers et al., Interspeech 2024; package sebbs 1.0.2):
                a step filter on the raw probabilities proposes onsets and offsets, neighbouring candidates are
                merged when the gap between them is shallow, each box gets the mean probability inside as its
                confidence, and the threshold is applied to that confidence. Extent and confidence are decoupled.
    csebb_caps  csebb with the duration limits of caps tuned on top (step filter and merging fixed at the
                fold's csebb choice).
    hyst        duration-aware hysteresis, for classes that mix call types of different length (D = blue whale
                D-calls, median 2.8 s, plus fin whale 40 Hz downsweeps, median 1.0 s): events are the runs above a
                low threshold (smoothing, merge gap and maximum duration from the fold's smooth choice, minimum
                0.5 s), and an event is kept if its peak probability exceeds one threshold when it is shorter than
                a split length and another when it is longer. A min-duration cap can only drop all short events;
                this can keep the confident short ones.
    route       per class, the decoder among smooth, csebb_caps and hyst with the best pooled F1 on the fold's
                two tuning sites.

No NMS decoder: one threshold on one class never gives overlapping events, so NMS has nothing to remove from
a frame model. Its job (one prediction per call) is what the merge gap, the duration limits and the SEBB
merging do here.

Besides macro B and per-class F1, the report gives the gain per held-out site, the recall per fine call type
(bma, bmb, bmz, bmd, bpd, bp20, bp20plus; precision cannot be split by fine type because false alarms have none)
and the parameters most often chosen.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .. import paths
from . import core
from .core import CLASSES, GRID, HOP_S, TABLES

VAL = paths.SITES["validation"]
SEEDS = (0, 1, 42, 666, 1337)
PIPELINES = {"2s": "val_2s", "centre": "val_centre"}
MODELS = ("nave", "AUG", "NOMAG", "LOGMAG", "FINE")
SHIFTED = ("kerguelen2014", "kerguelen2015")
DECODERS = ("base", "caps", "smooth", "csebb", "csebb_caps", "hyst", "route")
ROUTE_FROM = ("smooth", "csebb_caps", "hyst")
IOU = 0.3
SMOOTH_S = (0.02, 0.5, 1.0, 2.0)            # median filter length; 0.02 s = one frame = off. 0.5 s is NAVE's
SMOOTH_BASE = 0.5
BASE = (0.5, 0.5, 30.0)                     # merge gap, min and max duration of the published pipeline
GAPS = (0.5, 1.0, 2.0)
MIN_DUR = (0.5, 1.0, 1.5, 2.0)
MAX_DUR = (4.0, 8.0, 15.0, 30.0)
CAPS = [(g, lo, hi) for g, lo, hi in product(GAPS, MIN_DUR, MAX_DUR)]
STEP_FILTER_S = (0.25, 0.5, 1.0, 2.0)       # sebbs' default grid is 0.32 to 0.64 s for DESED; Z-calls last ~8 s
MERGE = [(a, np.inf) for a in (0.15, 0.2, 0.3)] + [(np.inf, r) for r in (1.5, 2.0, 3.0)]   # (abs, rel) as sebbs.tune
SEBB_DUR = [(0.0, np.inf)] + [(lo, hi) for lo, hi in product(MIN_DUR, MAX_DUR)]
HYST_LO = (0.1, 0.2, 0.3)                   # plus the fold's smooth threshold
HYST_HI = tuple(np.round(np.arange(0.1, 0.91, 0.1), 2))
HYST_SPLIT = (1.5, 2.5)


def say(msg: str) -> None:
    print(f"[decode] {msg}", flush=True)


def smooth(p: np.ndarray, seconds: float) -> np.ndarray:
    """Median filter per class; at 0.5 s exactly core.smooth (25 frames, scipy's default mode)."""
    n = max(1, int(round(seconds / HOP_S)))
    if n == 1:
        return np.asarray(p, dtype=np.float32)
    out = np.zeros_like(p, dtype=np.float32)
    for c in range(p.shape[1]):
        out[:, c] = median_filter(p[:, c], size=n)
    return out


# ------------------------------------------------------------------------------------------- events

def threshold_runs(p: np.ndarray, thr: float):
    """Runs of p > thr as (start_frame, end_frame, mean probability, peak probability), before merging
    (start, end and mean exactly as core.runs_to_events)."""
    active = p > thr
    d = np.diff(active.astype(np.int8), prepend=0, append=0)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if len(s) == 0:
        return s, e, np.empty(0), np.empty(0)
    cs = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    peak = np.maximum.reduceat(np.append(p, 0.0), np.column_stack([s, e]).ravel())[::2]
    return s, e, (cs[e] - cs[s]) / (e - s), peak.astype(np.float64)


class Events:
    """Candidate events of one site and class: file index, start and end (s, file-relative), confidence, peak."""

    def __init__(self, fidx, st, en, conf, peak=None):
        self.fidx, self.st, self.en, self.conf = (np.asarray(fidx, np.int64), np.asarray(st, np.float64),
                                                  np.asarray(en, np.float64), np.asarray(conf, np.float64))
        self.peak = self.conf if peak is None else np.asarray(peak, np.float64)

    @staticmethod
    def empty() -> "Events":
        return Events([], [], [], [], [])

    @staticmethod
    def concat(parts: list) -> "Events":
        if not parts:
            return Events.empty()
        return Events(*(np.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))))

    def select(self, m) -> "Events":
        return Events(self.fidx[m], self.st[m], self.en[m], self.conf[m], self.peak[m])

    def merge(self, gap: float) -> "Events":
        """Merge events of the same file whose gap is at most `gap` seconds; confidence and peak of the merged
        event are the maxima (same arithmetic as core.runs_to_events). Sorted by (file, start)."""
        if len(self.st) == 0:
            return self
        new = np.r_[True, (self.fidx[1:] != self.fidx[:-1]) | (self.st[1:] - self.en[:-1] > gap)]
        a = np.flatnonzero(new)
        b = np.r_[a[1:], len(self.st)] - 1
        return Events(self.fidx[a], self.st[a], self.en[b], np.maximum.reduceat(self.conf, a), np.maximum.reduceat(self.peak, a))

    @property
    def dur(self) -> np.ndarray:
        return self.en - self.st


def merge_segments(seg_bounds, mean_s, min_s, max_s, thr_abs: float, thr_rel: float):
    """Vectorised equivalent of sebbs.csebbs._merge_segments (checked against it in `smoke`): merge two event
    candidates when the gap between them is shallow relative to both (absolute and relative criterion)."""
    if len(seg_bounds) >= 6:
        ev = max_s[1:-1:2] + 1e-3
        gp = min_s[2:-2:2] + 1e-3
        merge = ((ev[:-1] / gp) < thr_rel) & ((ev[1:] / gp) < thr_rel) & ((ev[:-1] - gp) < thr_abs) & ((ev[1:] - gp) < thr_abs)
        g = np.flatnonzero(~merge)
        idx = np.concatenate(([0, 1], np.stack((2 + 2 * g, 3 + 2 * g), axis=1).ravel(), [len(seg_bounds) - 2, len(seg_bounds) - 1])).astype(int)
        lens = seg_bounds[1:] - seg_bounds[:-1]
        st = idx[:-1]
        tot = np.add.reduceat(lens, st)
        wsum = np.add.reduceat(lens * mean_s, st)
        zero = tot == 0.0
        mean_s = np.where(zero, 0.0, wsum / (tot + 1e-6))
        min_s = np.where(zero, 0.0, np.minimum.reduceat(min_s, st))
        max_s = np.where(zero, 0.0, np.maximum.reduceat(max_s, st))
        seg_bounds = seg_bounds[idx]
    return seg_bounds[1:-2:2], seg_bounds[2:-1:2], mean_s[1:-1:2]


# ----------------------------------------------------------------------------------------- matching

class Labels:
    """Labels of one site and class, for the exact single-class equivalent of core.official_match."""

    def __init__(self, t0: np.ndarray, t1: np.ndarray):
        self.n = len(t0)
        self.t0, self.t1 = t0, t1
        self.order = np.argsort(t0, kind="stable")
        self.sorted_t0 = t0[self.order]
        self.max_dur = float((t1 - t0).max()) if self.n else 0.0


def count(ev: Events, lab: Labels, epoch: np.ndarray) -> tuple[int, int, int]:
    """tp, fp, fn for one site and class, matching exactly like core.official_match on that class alone.

    Why the shortcut is exact: the scorer walks files in sorted order and rows in csv order (here: by file,
    then start), and each prediction takes the first label (in label order) with the highest IoU among the
    labels not matched yet, if that IoU is above 0.3. Its +-2 min window around a file's predictions always
    contains every label that overlaps one of them (labels last under 40 s), and a label without overlap has
    IoU 0 and can never be taken, so it is enough to look at the labels that overlap each prediction."""
    n_pred = len(ev.st)
    if n_pred == 0 or lab.n == 0:
        return 0, n_pred, lab.n
    p0 = epoch[ev.fidx] + ev.st
    p1 = epoch[ev.fidx] + ev.en
    lo = np.searchsorted(lab.sorted_t0, p0 - lab.max_dur - 1e-6, "left")
    hi = np.searchsorted(lab.sorted_t0, p1, "left")
    cand = np.flatnonzero(hi > lo)
    if len(cand) == 0:
        return 0, n_pred, lab.n
    taken = np.zeros(lab.n, bool)
    order, t0, t1 = lab.order, lab.t0, lab.t1
    tp = 0
    for r in cand:
        js = np.sort(order[lo[r]:hi[r]])            # back to label order
        js = js[~taken[js]]
        if len(js) == 0:
            continue
        a, b = p0[r], p1[r]
        inter = np.clip(np.minimum(b, t1[js]) - np.maximum(a, t0[js]), 0, None)
        iou = inter / ((b - a) + (t1[js] - t0[js]) - inter)
        k = int(np.argmax(iou))
        if iou[k] > IOU:
            taken[js[k]] = True
            tp += 1
    return tp, n_pred - tp, lab.n - tp


def f1(tp, fp, fn) -> np.ndarray:
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(2 * tp / (2 * tp + fp + fn))


# -------------------------------------------------------------------------------------------- a site

class Site:
    """Frame probabilities of one site turned into all candidate event sets the decoders need."""

    def __init__(self, name: str, prob_dir: Path, gt: pd.DataFrame, smooths=SMOOTH_S, with_sebb: bool = True):
        self.name = name
        files = sorted(prob_dir.glob("*.npz"))
        if not files:
            raise SystemExit(f"no frame probabilities in {prob_dir}")
        self.files = [f.stem + ".wav" for f in files]
        self.epoch = core.epochs(pd.Series(self.files))
        g = gt[gt.dataset == name]
        self.labels = {c: Labels(g[g.cls == c].t0.to_numpy(), g[g.cls == c].t1.to_numpy()) for c in CLASSES}
        assert set(HYST_LO) <= set(map(float, GRID))
        self.thresholds = [float(t) for t in GRID]
        runs = {(c, sm, t): [] for c in CLASSES for sm in smooths for t in self.thresholds}
        sebb = {(c, L, m): [] for c in CLASSES for L in (STEP_FILTER_S if with_sebb else ()) for m in range(len(MERGE))}
        from sebbs.change_detection import change_detection
        for fi, f in enumerate(files):
            with np.load(f) as z:
                raw = z["probs"].astype(np.float32)
            for sm in smooths:
                p = core.smooth(raw) if sm == SMOOTH_BASE else smooth(raw, sm)
                for ci, c in enumerate(CLASSES):
                    for t in self.thresholds:
                        s, e, cf, pk = threshold_runs(p[:, ci], t)
                        runs[(c, sm, t)].append((np.full(len(s), fi), s * HOP_S, e * HOP_S, cf, pk))
            ts = np.arange(len(raw) + 1) * HOP_S
            for L in (STEP_FILTER_S if with_sebb else ()):
                cd = change_detection(raw.astype(np.float64), ts, L)
                for ci, c in enumerate(CLASSES):
                    sb, mean_s, min_s, max_s = cd[ci]
                    for m, (ta, tr) in enumerate(MERGE):
                        on, off, cf = merge_segments(np.asarray(sb), np.asarray(mean_s), np.asarray(min_s), np.asarray(max_s), ta, tr)
                        sebb[(c, L, m)].append((np.full(len(on), fi), on, off, cf))
        self.runs = {k: Events.concat(v) for k, v in runs.items()}          # thresholded, not merged
        self.sebb = {k: Events.concat(v) for k, v in sebb.items()}          # all boxes, any confidence
        self._merged = {}

    def merged(self, c: str, sm: float, thr: float, gap: float) -> Events:
        k = (c, sm, thr, gap)
        if k not in self._merged:
            self._merged[k] = self.runs[(c, sm, thr)].merge(gap)
        return self._merged[k]

    def caps_events(self, c: str, thr: float, gap: float, lo: float, hi: float, sm: float = SMOOTH_BASE) -> Events:
        e = self.merged(c, sm, float(thr), gap)
        d = e.dur
        return e.select((d >= lo) & (d <= hi))

    def hyst_events(self, c: str, sm: float, gap: float, hi: float, t_lo: float, split: float, t_short: float, t_long: float) -> Events:
        e = self.caps_events(c, t_lo, gap, 0.5, hi, sm)
        d = e.dur
        return e.select(e.peak > np.where(d < split, t_short, t_long))

    def sebb_events(self, c: str, L: float, m: int, thr: float, lo: float = 0.0, hi: float = np.inf) -> Events:
        e = self.sebb[(c, L, m)]
        d = e.dur
        return e.select((e.conf > thr) & (d >= lo) & (d <= hi))

    def count(self, c: str, ev: Events):
        return count(ev, self.labels[c], self.epoch)

    def frame(self, per_class: dict[str, Events]) -> pd.DataFrame:
        """Events of all classes as a detections table in the scorer's row order (file, class, start)."""
        parts = []
        for ci, c in enumerate(CLASSES):
            e = per_class[c]
            parts.append(pd.DataFrame({"dataset": self.name, "fidx": e.fidx, "cls": c, "cls_order": ci,
                                       "start_s": e.st, "end_s": e.en, "conf": e.conf}))
        ev = pd.concat(parts, ignore_index=True).sort_values(["fidx", "cls_order", "start_s"], kind="stable")
        ev["filename"] = np.array(self.files)[ev.fidx.to_numpy()]
        ev = ev.drop(columns=["fidx"]).reset_index(drop=True)
        return core.add_abs(ev)


# ------------------------------------------------------------------------------------------- tuning

def needs(decoders) -> set:
    need = set(decoders) | (set(ROUTE_FROM) if "route" in decoders else set())
    if "hyst" in need:
        need.add("smooth")
    if "csebb_caps" in need:
        need.add("csebb")
    return need


def site_options(decoders) -> dict:
    need = needs(decoders)
    return {"smooths": SMOOTH_S if "smooth" in need else (SMOOTH_BASE,), "with_sebb": "csebb" in need}


COLS = ["family", "site", "cls", "thr", "smooth", "gap", "min_dur", "max_dur", "step_s", "merge", "tp", "fp", "fn"]


def sweep(sites: dict[str, Site], decoders=DECODERS) -> pd.DataFrame:
    """tp, fp, fn of every grid point of base/caps/smooth and csebb (as far as the decoders need them),
    per site and class."""
    opt = site_options(decoders)
    rows = []
    for s, site in sites.items():
        for c in CLASSES:
            for t in GRID:
                t = float(t)
                for sm in opt["smooths"]:
                    for gap, lo, hi in CAPS:
                        rows.append(("caps", s, c, t, sm, gap, lo, hi, np.nan, -1, *site.count(c, site.caps_events(c, t, gap, lo, hi, sm))))
                for L in (STEP_FILTER_S if opt["with_sebb"] else ()):
                    for m in range(len(MERGE)):
                        rows.append(("csebb", s, c, t, np.nan, np.nan, 0.0, np.inf, L, m, *site.count(c, site.sebb_events(c, L, m, t))))
    return pd.DataFrame(rows, columns=COLS)


def best(sw: pd.DataFrame, keys: list[str]) -> tuple[dict, float]:
    """Grid point with the highest pooled F1 over the rows given (first in grid order on ties), and that F1."""
    s = sw.groupby(keys, sort=False, dropna=False)[["tp", "fp", "fn"]].sum()
    v = f1(s.tp, s.fp, s.fn)
    i = int(np.argmax(v))
    return dict(zip(keys, s.index[i] if len(keys) > 1 else (s.index[i],))), float(v[i])


def pooled_best(rows: list[tuple], keys: list[str]) -> tuple[dict, float]:
    return best(pd.DataFrame(rows, columns=["site", *keys, "tp", "fp", "fn"]), keys)


def tune_and_apply(sites: dict[str, Site], sw: pd.DataFrame, decoders) -> tuple[dict, dict]:
    """Nested LOSO: per held-out site and class, parameters from the other sites; returns the held-out events
    as detections tables per decoder, and the chosen parameters."""
    need = needs(decoders)
    out = {d: [] for d in decoders}
    params = {d: {} for d in decoders}
    for held in sites:
        train = [s for s in sites if s != held]
        per = {d: {} for d in need}
        prm, score = {d: {} for d in need}, {d: {} for d in need}
        H = sites[held]
        for c in CLASSES:
            caps = sw[(sw.family == "caps") & (sw.cls == c) & sw.site.isin(train)]
            cs = sw[(sw.family == "csebb") & (sw.cls == c) & sw.site.isin(train)]
            if "base" in need:
                b = caps[(caps.smooth == SMOOTH_BASE) & (caps.gap == BASE[0]) & (caps.min_dur == BASE[1]) & (caps.max_dur == BASE[2])]
                p, score["base"][c] = best(b, ["thr"])
                per["base"][c], prm["base"][c] = H.caps_events(c, p["thr"], *BASE), p
            if "caps" in need:
                p, score["caps"][c] = best(caps[caps.smooth == SMOOTH_BASE], ["gap", "min_dur", "max_dur", "thr"])
                per["caps"][c], prm["caps"][c] = H.caps_events(c, p["thr"], p["gap"], p["min_dur"], p["max_dur"]), p
            if "smooth" in need:
                p, score["smooth"][c] = best(caps, ["smooth", "gap", "min_dur", "max_dur", "thr"])
                per["smooth"][c] = H.caps_events(c, p["thr"], p["gap"], p["min_dur"], p["max_dur"], p["smooth"])
                prm["smooth"][c] = p
            if "csebb" in need:
                p, score["csebb"][c] = best(cs, ["step_s", "merge", "thr"])
                L, m = p["step_s"], int(p["merge"])
                per["csebb"][c] = H.sebb_events(c, L, m, p["thr"])
                prm["csebb"][c] = {**p, "merge": list(MERGE[m])}
            if "csebb_caps" in need:
                rows = [(s, float(t), lo, hi, *sites[s].count(c, sites[s].sebb_events(c, L, m, t, lo, hi)))
                        for s in train for t in GRID for lo, hi in SEBB_DUR]
                q, score["csebb_caps"][c] = pooled_best(rows, ["thr", "min_dur", "max_dur"])
                per["csebb_caps"][c] = H.sebb_events(c, L, m, q["thr"], q["min_dur"], q["max_dur"])
                prm["csebb_caps"][c] = {"step_s": L, "merge": list(MERGE[m]), **q}
            if "hyst" in need:
                ps = prm["smooth"][c]
                sm, gap, hi = ps["smooth"], ps["gap"], ps["max_dur"]
                lows = sorted(set(HYST_LO) | {float(ps["thr"])})
                rows = []
                for s in train:
                    for t_lo in lows:
                        e = sites[s].caps_events(c, t_lo, gap, 0.5, hi, sm)
                        d = e.dur
                        his = sorted({float(h) for h in HYST_HI if h >= t_lo} | {t_lo})   # t_lo itself = no extra filter
                        for split in HYST_SPLIT:
                            short = d < split
                            for t_s, t_l in product(his, his):
                                ev = e.select(e.peak > np.where(short, t_s, t_l))
                                rows.append((s, t_lo, split, t_s, t_l, *sites[s].count(c, ev)))
                q, score["hyst"][c] = pooled_best(rows, ["t_lo", "split", "t_short", "t_long"])
                per["hyst"][c] = H.hyst_events(c, sm, gap, hi, q["t_lo"], q["split"], q["t_short"], q["t_long"])
                prm["hyst"][c] = {"smooth": sm, "gap": gap, "max_dur": hi, **q}
            if "route" in decoders:
                pick = max(ROUTE_FROM, key=lambda d: score[d][c])        # first of ROUTE_FROM on ties
                per.setdefault("route", {})[c] = per[pick][c]
                prm.setdefault("route", {})[c] = {"decoder": pick, "train_f1": score[pick][c]}
        for d in decoders:
            out[d].append(H.frame(per[d]))
            for c in CLASSES:
                params[d][f"{held}/{c}"] = prm[d][c]
    return {d: pd.concat(v, ignore_index=True) for d, v in out.items()}, params


def score_all(events: dict[str, pd.DataFrame], gt: pd.DataFrame, tag: dict):
    mac_rows, cls_rows, fine_rows = [], [], []
    for d, ev in events.items():
        t, mac = core.score(ev, gt, IOU)
        for r in mac.itertuples():
            mac_rows.append({**tag, "decoder": d, "dataset": r.dataset, "macro_A": r.macro_A, "macro_B": r.macro_B})
        for r in t[t.dataset == "joined"].itertuples():
            cls_rows.append({**tag, "decoder": d, "cls": r.cls, "tp": r.tp, "fp": r.fp, "fn": r.fn,
                             "precision": r.precision, "recall": r.recall, "f1": r.f1})
        for ds, g in gt.groupby("dataset", sort=False):
            g = g.reset_index(drop=True)
            _, gm = core.official_match(ev[ev.dataset == ds].reset_index(drop=True), g, IOU)
            for fine, n in g.groupby("fine").size().items():
                fine_rows.append({**tag, "decoder": d, "dataset": ds, "fine": fine, "n": int(n),
                                  "tp": int((gm[(g.fine == fine).to_numpy()] >= 0).sum())})
    return mac_rows, cls_rows, fine_rows


SHIFT_DECODERS = ("base", "caps", "smooth", "hyst")


def events_from(site: Site, d: str, c: str, p: dict) -> Events:
    """Events of one class at a held-out fold's chosen parameters (threshold-based decoders)."""
    if d == "base":
        return site.caps_events(c, p["thr"], *BASE)
    if d == "caps":
        return site.caps_events(c, p["thr"], p["gap"], p["min_dur"], p["max_dur"])
    if d == "smooth":
        return site.caps_events(c, p["thr"], p["gap"], p["min_dur"], p["max_dur"], p["smooth"])
    if d == "hyst":
        return site.hyst_events(c, p["smooth"], p["gap"], p["max_dur"], p["t_lo"], p["split"], p["t_short"], p["t_long"])
    raise ValueError(d)


def shift_rows(sites: dict[str, Site], shift_dirs: dict[str, Path], gt, params, decoders, tag) -> list[dict]:
    """Recall on the original and on the -0.6 Hz shifted audio of the Kerguelen sites, each at the
    parameters chosen with that site held out."""
    rows = []
    dec = [d for d in decoders if d in SHIFT_DECODERS]
    smooths = tuple(sorted({params[d][f"{s}/{c}"].get("smooth", SMOOTH_BASE) for d in dec for s in shift_dirs for c in CLASSES}))
    for s, sd in shift_dirs.items():
        sh = Site(s, sd, gt, smooths=smooths, with_sebb=False)
        for d in dec:
            for c in CLASSES:
                p = params[d][f"{s}/{c}"]
                for cond, site in (("original", sites[s]), ("shifted", sh)):
                    tp, _, fn = site.count(c, events_from(site, d, c, p))
                    rows.append({**tag, "decoder": d, "site": s, "condition": cond, "cls": c, "tp": tp, "n": tp + fn})
    return rows


def job(model, seed, pipeline, prob_dirs: dict[str, Path], decoders, shift_dirs=None, gt_sites=None):
    t0 = time.time()
    name = f"{model} seed {seed} {pipeline}"
    gt = core.load_labels("validation")
    if gt_sites is not None:                                   # smoke: only the files we have
        gt = gt[gt.filename.isin(gt_sites)].reset_index(drop=True)
    sites = {s: Site(s, prob_dirs[s], gt, **site_options(decoders)) for s in VAL}
    say(f"{name}: candidates ready ({time.time() - t0:.0f} s)")
    sw = sweep(sites, decoders)
    say(f"{name}: sweep done ({time.time() - t0:.0f} s)")
    events, params = tune_and_apply(sites, sw, decoders)
    tag = {"model": model, "seed": seed, "pipeline": pipeline}
    mac, cls, fine = score_all(events, gt, tag)
    shift = shift_rows(sites, shift_dirs, gt, params, decoders, tag) if shift_dirs else []
    say(f"{name}: done ({time.time() - t0:.0f} s)")
    return mac, cls, fine, shift, {f"{model}/{seed}/{pipeline}": params}


# ------------------------------------------------------------------------------------------- report

def ms(x: pd.Series) -> str:
    return f"{x.mean():.3f} ({x.std(ddof=1):.3f})" if len(x) > 1 else f"{x.mean():.3f}"


def label(df: pd.DataFrame, multi: bool) -> pd.DataFrame:
    """With several models, 'pipeline' becomes '<model> <pipeline>' for the tables of the report."""
    if df is None or df.empty or not multi:
        return df
    return df.assign(pipeline=df["model"].astype(str) + " " + df["pipeline"].astype(str))


def compare_models(mac: pd.DataFrame, shift: pd.DataFrame) -> list[str]:
    """Paired by seed: every other model minus nave, per pipeline and decoder; drift per model."""
    lines = []
    j = mac[mac.dataset == "joined"]
    models = list(dict.fromkeys(j.model))
    if "nave" in models and len(models) > 1:
        ref = j[j.model == "nave"].set_index(["pipeline", "decoder", "seed"]).macro_B
        lines.append("\nmodel against nave, macro B difference paired by seed (mean (sd), seeds where the model is ahead):")
        for m in [x for x in models if x != "nave"]:
            mm = j[j.model == m].set_index(["pipeline", "decoder", "seed"]).macro_B
            dif = (mm - ref).dropna()
            for (pl, d), g in dif.groupby(level=["pipeline", "decoder"], sort=False):
                lines.append(f"  {m:6s} {pl:6s} {d:10s} {ms(g)}, {int((g > 0).sum())} of {len(g)}")
    if shift is not None and not shift.empty:
        r = shift.groupby(["model", "decoder", "seed", "cls", "condition"], sort=False)[["tp", "n"]].sum()
        r = (r.tp / r.n).unstack("condition")
        r["change"] = (r["shifted"] - r["original"]) * 100
        lines.append("\nfrequency drift, Kerguelen 2014/15 shifted by -0.6 Hz (2 s pipeline): recall original, shifted, change in points (mean over seeds; change: mean (sd) and range):")
        g = r.groupby(["model", "decoder", "cls"], sort=False)
        for (m, d, c), x in g:
            sd = f" ({x.change.std(ddof=1):.1f})" if len(x) > 1 else ""
            lines.append(f"  {m:6s} {d:7s} {c:5s} {x.original.mean():.3f}  {x.shifted.mean():.3f}  {x.change.mean():+.1f}{sd}"
                         f"  [{x.change.min():+.1f} to {x.change.max():+.1f}]")
    return lines


def report(mac: pd.DataFrame, cls: pd.DataFrame, fine: pd.DataFrame, params: dict, shift: pd.DataFrame | None = None) -> str:
    extra = compare_models(mac, shift) if "model" in mac else []
    multi = "model" in mac and (mac.model.nunique() > 1 or set(mac.model) != {"nave"})
    mac, cls, fine = (label(x, multi) for x in (mac, cls, fine))
    j = mac[mac.dataset == "joined"]
    dec = [d for d in DECODERS if d in set(j.decoder)]
    pls = list(dict.fromkeys(j.sort_values("pipeline", key=lambda s: s.map(lambda v: (v.split(" ")[0] != "nave", v)), kind="stable").pipeline))
    cols = [(p, d) for p in pls for d in dec]
    lines = ["LOSO macro B (dev joined) per seed:"]
    piv = j.pivot_table(index="seed", columns=["pipeline", "decoder"], values="macro_B")
    lines.append(piv.reindex(columns=cols).round(3).to_string())
    lines.append("\nmean (sd) over seeds, macro B | macro A | gain in macro B over base of the same pipeline (mean (sd), seeds with a gain):")
    for pl in pls:
        base = j[(j.pipeline == pl) & (j.decoder == "base")].set_index("seed").macro_B
        for d in dec:
            g = j[(j.pipeline == pl) & (j.decoder == d)]
            gain = g.set_index("seed").macro_B - base
            lines.append(f"  {pl:11s} {d:10s} {ms(g.macro_B)} | {ms(g.macro_A)} | {ms(gain)}, {int((gain > 0).sum())} of {len(gain)}")
    s = mac[mac.dataset != "joined"]
    keys = ["seed", "pipeline", "dataset"]
    b = s[s.decoder == "base"][keys + ["macro_B"]].rename(columns={"macro_B": "base_B"})
    gains = s.merge(b, on=keys, how="left").assign(gain=lambda x: x.macro_B - x.base_B)
    lines.append("\ngain in macro B over base per held-out site (mean over seeds):")
    lines.append(gains.pivot_table(index=["pipeline", "decoder"], columns="dataset", values="gain", sort=False)
                 .reindex(cols).dropna(how="all").round(3).to_string())
    lines += extra
    lines.append("\nper class F1, P, R (dev joined), mean over seeds:")
    c = cls.groupby(["pipeline", "decoder", "cls"], sort=False)[["f1", "precision", "recall"]].mean().unstack("cls")
    lines.append(c.round(3).to_string())
    fr = fine.groupby(["pipeline", "decoder", "seed", "fine"], sort=False)[["tp", "n"]].sum()
    fr = (fr.tp / fr.n).groupby(["pipeline", "decoder", "fine"], sort=False).mean().unstack("fine")
    order = [f for f in ("bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus") if f in fr.columns]
    lines.append("\nrecall per fine call type (dev joined, mean over seeds):")
    lines.append(fr[order].round(3).to_string())
    lines.append("\nparameters chosen most often (over seeds, pipelines and folds), per decoder and class:")
    for d in dec:
        for c in CLASSES:
            ps = [v for run in params.values() for k, v in run.get(d, {}).items() if k.endswith("/" + c)]
            if not ps:
                continue
            keys = [k for k in ps[0] if k != "train_f1"]
            common = {k: Counter(json.dumps(p.get(k), default=float) for p in ps).most_common(1)[0] for k in keys}
            lines.append(f"  {d:10s} {c:5s} " + ", ".join(f"{k} {v} ({n}/{len(ps)})" for k, (v, n) in common.items()))
    return "\n".join(lines)


def check_base(mac: pd.DataFrame) -> str:
    """base must reproduce the earlier scoring: seeds_scores for nave, ablate_scores (2 s) for the variants."""
    out = []
    mine = mac[(mac.dataset == "joined") & (mac.decoder == "base")]
    for model, g in mine.groupby("model", sort=False):
        if model == "nave":
            f = TABLES / "seeds_scores.parquet"
            if not f.exists():
                out.append("base check nave: no seeds_scores.parquet")
                continue
            ref = pd.read_parquet(f)
            ref = ref[ref.dataset == "joined"].set_index(["seed", "pipeline"]).macro_B
            m = g.set_index(["seed", "pipeline"]).macro_B
        else:
            f = TABLES / "ablate_scores.parquet"
            if not f.exists():
                out.append(f"base check {model}: no ablate_scores.parquet")
                continue
            ref = pd.read_parquet(f)
            ref = ref[ref.rung == model].set_index("seed").macro_B
            m = g[g.pipeline == "2s"].set_index("seed").macro_B
        both = pd.concat({"earlier": ref, "decode base": m}, axis=1).dropna()
        if both.empty:
            out.append(f"base check {model}: nothing to compare")
        elif np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9):
            out.append(f"base check {model}: reproduces the earlier scores exactly ({len(both)} runs)")
        else:
            out.append(f"base check {model}: MISMATCH\n" + both.round(4).to_string())
    return "\n".join(out) or "base check: nothing to compare"


# ---------------------------------------------------------------------------------------------- cli

def collect(results):
    mac, cls, fine, shift, params = [], [], [], [], {}
    for m, c, f, s, p in results:
        mac += m
        cls += c
        fine += f
        shift += s
        params.update(p)
    return pd.DataFrame(mac), pd.DataFrame(cls), pd.DataFrame(fine), pd.DataFrame(shift), params


def model_root(model: str, seed: int) -> Path:
    return core.ROOT / ("seeds" if model == "nave" else "ablate") / (f"seed{seed}" if model == "nave" else f"{model}_seed{seed}")


def shift_root(model: str, seed: int) -> Path:
    if model == "nave":
        from . import seeds as sd
        return sd.run_dirs(seed)["shift"]
    return model_root(model, seed) / "shift"


def predict(a) -> None:
    """Centre-stitched dev outputs (30 s overlap) for the models given; resumable, skips finished sites."""
    from . import ablate, seeds as sd
    for model in a.models:
        for seed in a.seeds:
            w = sd.ckpt(seed) if model == "nave" else ablate.ckpt(model, seed)
            if w is None:
                say(f"{model} seed {seed}: no checkpoint, skipped")
                continue
            out = model_root(model, seed) / PIPELINES["centre"]
            for site in VAL:
                sd.go(["nave", "--weights", str(w), "--device", a.device, "--split", "validation", "--sites", site,
                       "--overlap-s", "30", "--stitch", "centre"], out / site)
    say("predict: done")


def run(a) -> None:
    jobs = []
    for model in a.models:
        for seed in a.seeds:
            for pl in a.pipelines:
                dirs = {s: model_root(model, seed) / PIPELINES[pl] / s / "frame_probs" / s for s in VAL}
                missing = [str(d) for d in dirs.values() if not d.is_dir()]
                if missing:
                    say(f"{model} seed {seed} {pl}: missing {missing[0]}, skipped")
                    continue
                shift = None
                if pl == "2s":
                    sdirs = {s: shift_root(model, seed) / s / "frame_probs" / s for s in SHIFTED}
                    shift = sdirs if all(d.is_dir() for d in sdirs.values()) else None
                jobs.append((model, seed, pl, dirs, a.decoders, shift))
    if not jobs:
        raise SystemExit("nothing to do")
    say(f"{len(jobs)} jobs on {a.workers} workers")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        mac, cls, fine, shift, params = collect(ex.map(job, *zip(*jobs)))
    sfx = f"_{a.tag}" if a.tag else ""
    mac.to_parquet(TABLES / f"decode_scores{sfx}.parquet")
    cls.to_parquet(TABLES / f"decode_classes{sfx}.parquet")
    fine.to_parquet(TABLES / f"decode_fine{sfx}.parquet")
    if not shift.empty:
        shift.to_parquet(TABLES / f"decode_shift{sfx}.parquet")
    (TABLES / f"decode_params{sfx}.json").write_text(json.dumps(params, indent=1, default=float))
    text = check_base(mac) + "\n\n" + report(mac, cls, fine, params, shift)
    (TABLES / f"decode_report{sfx}.log").write_text(text + "\n")
    say("\n" + text)


def smoke(a) -> None:
    from sebbs.csebbs import _merge_segments
    from sebbs.change_detection import change_detection
    # 1. the vectorised merge equals sebbs' own
    f = sorted(a.prob_root.glob("*/*.npz"))[0]
    with np.load(f) as z:
        raw = z["probs"].astype(np.float64)
    ts = np.arange(len(raw) + 1) * HOP_S
    n_ok = 0
    for L in STEP_FILTER_S:
        cd = change_detection(raw, ts, L)
        for ci in range(len(CLASSES)):
            sb, mn, lo, hi = (np.asarray(x) for x in cd[ci])
            for ta, tr in MERGE:
                ref = _merge_segments(sb, mn, lo, hi, threshold_abs=ta, threshold_rel=tr)
                mine = merge_segments(sb, mn, lo, hi, ta, tr)
                assert all(np.allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-9) for x, y in zip(ref, mine)), (L, ci, ta, tr)
                n_ok += 1
    say(f"smoke: merge_segments equals sebbs' _merge_segments in {n_ok} cases")
    # 2. events at the published settings equal core.runs_to_events, peaks equal the frame maximum
    sm = core.smooth(raw.astype(np.float32))
    for ci in range(len(CLASSES)):
        for t in GRID[::4]:
            s, e, cf, pk = threshold_runs(sm[:, ci], float(t))
            ev = Events(np.zeros(len(s), int), s * HOP_S, e * HOP_S, cf, pk).merge(BASE[0])
            ev = ev.select((ev.dur >= BASE[1]) & (ev.dur <= BASE[2]))
            ref = core.runs_to_events(sm[:, ci], float(t))
            assert all(np.array_equal(x, y) for x, y in zip(ref, (ev.st, ev.en, ev.conf))), (ci, t)
            assert np.allclose(pk, [sm[a:b, ci].max() for a, b in zip(s, e)]), (ci, t)
    say("smoke: base events equal core.runs_to_events, peaks equal the frame maximum")
    # 3. the matcher equals core.official_match
    files = {d.name: d for d in sorted(a.prob_root.iterdir()) if d.is_dir()}
    names = {p.stem + ".wav" for d in files.values() for p in d.glob("*.npz")}
    gt = core.load_labels("validation")
    gt = gt[gt.filename.isin(names)].reset_index(drop=True)
    n_ok = 0
    for s, d in files.items():
        site = Site(s, d, gt)
        g = gt[gt.dataset == s]
        for c in CLASSES:
            gc = g[g.cls == c].reset_index(drop=True)
            for t in GRID[::3]:
                for gap, lo, hi in CAPS[::5]:
                    for ev in (site.caps_events(c, t, gap, lo, hi, 1.0), site.sebb_events(c, 0.5, 0, t),
                               site.hyst_events(c, SMOOTH_BASE, gap, hi, 0.2, 1.5, float(t), 0.5)):
                        df = site.frame({k: ev if k == c else Events.empty() for k in CLASSES})
                        pm, gm = core.official_match(df, gc, IOU)
                        ref = (int((gm >= 0).sum()), int((pm < 0).sum()), int((gm < 0).sum()))
                        assert site.count(c, ev) == ref, (s, c, t, gap, lo, hi, site.count(c, ev), ref)
                        n_ok += 1
    say(f"smoke: matcher equals core.official_match in {n_ok} cases")
    # 4. the whole pipeline on these files
    shift = {s: files[s] for s in SHIFTED if s in files}       # stands in for the shifted audio
    res = [job(m, "smoke", "2s", files, a.decoders, shift_dirs=shift, gt_sites=names) for m in ("nave", "AUG")]
    mac, cls, fine, sh, params = collect(res)
    say("smoke: pipeline ran (scores on so few files mean nothing)\n" + report(mac, cls, fine, params, sh))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["run", "predict", "smoke"])
    p.add_argument("--models", nargs="+", default=["nave"], choices=MODELS)
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--pipelines", nargs="+", default=list(PIPELINES), choices=list(PIPELINES))
    p.add_argument("--decoders", nargs="+", default=list(DECODERS), choices=list(DECODERS))
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--tag", default="", help="suffix for the output tables")
    p.add_argument("--prob-root", type=Path, help="smoke: folder with <site>/*.npz")
    a = p.parse_args(argv)
    if a.step == "smoke":
        if a.prob_root is None:
            raise SystemExit("smoke needs --prob-root")
        smoke(a)
    elif a.step == "predict":
        predict(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
