"""Shared pieces for Phase I: labels, events from the saved model outputs, and the official matching.

Times: every table carries file-relative seconds (start_s, end_s) and absolute UTC epoch seconds
(t0, t1). The matching works on t0/t1, like the official scorer works on datetimes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .. import paths
from ..recordings import file_start

ROOT = paths.OUTPUTS / "phase1"
TABLES = ROOT / "tables"
MODELS = ("yolo", "whalevad", "nave")
MODEL_NAMES = {"yolo": "YOLO", "whalevad": "Whale-VAD", "nave": "NAVE"}
CLASSES = paths.CLASSES                      # bmabz, d, bp
HOP_S = 0.02
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)          # frame-model thresholds
YOLO_GRID = np.round(np.r_[0.01, 0.02, 0.03, 0.05, 0.075, np.arange(0.1, 0.951, 0.05)], 3)
SMOOTH_FRAMES = 25          # NAVE: 500 ms median filter at 20 ms frames
MERGE_GAP_S, MIN_DUR_S, MAX_DUR_S = 0.5, 0.5, 30.0
DEFAULT_THR = {"yolo": [0.1, 0.1, 0.1], "whalevad": [0.5, 0.5, 0.5]}   # NAVE: from its run.json


def split_of(site: str) -> str:
    for split, sites in paths.SITES.items():
        if site in sites:
            return split
    raise KeyError(site)


def epoch(filename: str) -> float:
    return file_start(filename).timestamp()


_EPOCH: dict[str, float] = {}


def epochs(filenames: pd.Series) -> np.ndarray:
    out = np.empty(len(filenames))
    for i, f in enumerate(filenames):
        v = _EPOCH.get(f)
        if v is None:
            v = _EPOCH[f] = epoch(f)
        out[i] = v
    return out


# ----------------------------------------------------------------------------- labels

def load_labels(split: str = "validation", sites=None) -> pd.DataFrame:
    """Annotations with class, fine label, file-relative and absolute times, in file order
    (sites in config order, rows as in the csv, the order the official scorer sees them)."""
    sites = sites or paths.SITES[split]
    parts = []
    for s in sites:
        f = paths.annotations_dir(split) / f"{s}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    gt = pd.concat(parts, ignore_index=True)
    gt = gt[gt["annotation"].notna()].reset_index(drop=True)
    gt = gt.rename(columns={"annotation": "fine"})
    gt["cls"] = gt["fine"].map(paths.FINE_TO_CLASS)
    t0 = pd.to_datetime(gt["start_datetime"], format="ISO8601", utc=True)
    t1 = pd.to_datetime(gt["end_datetime"], format="ISO8601", utc=True)
    gt["t0"] = t0.astype("int64") / 1e9
    gt["t1"] = t1.astype("int64") / 1e9
    fe = epochs(gt["filename"])
    gt["start_s"] = gt["t0"] - fe
    gt["end_s"] = gt["t1"] - fe
    gt["duration"] = gt["t1"] - gt["t0"]
    gt["month"] = t0.dt.month
    gt["hour"] = t0.dt.hour
    gt.insert(0, "label_id", np.arange(len(gt)))
    return gt[["label_id", "dataset", "filename", "cls", "fine", "annotator", "start_s", "end_s", "t0", "t1",
               "duration", "low_frequency", "high_frequency", "month", "hour"]]


# ------------------------------------------------------------------ frame-model events

def smooth(probs: np.ndarray) -> np.ndarray:
    """Same as nave_train.smooth_probabilities (scipy median filter, 25 frames, mode reflect)."""
    probs = np.asarray(probs, dtype=np.float32)
    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = median_filter(probs[:, c], size=SMOOTH_FRAMES)
    return out


def runs_to_events(p: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One class of smoothed probabilities -> (start_s, end_s, confidence), exactly like
    threshold_to_detections + merge_and_filter of NAVE (same float arithmetic)."""
    active = p > thr
    d = np.diff(active.astype(np.int8), prepend=0, append=0)
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    if len(s) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    cs = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    conf = (cs[e] - cs[s]) / (e - s)
    st, en = s * HOP_S, e * HOP_S
    # merge: a new group starts where the gap to the running end is above MERGE_GAP_S
    out_s, out_e, out_c = [st[0]], [en[0]], [conf[0]]
    for a, b, c in zip(st[1:], en[1:], conf[1:]):
        if a - out_e[-1] <= MERGE_GAP_S:
            out_e[-1] = max(out_e[-1], b)
            out_c[-1] = max(out_c[-1], c)
        else:
            out_s.append(a)
            out_e.append(b)
            out_c.append(c)
    out_s, out_e, out_c = np.array(out_s), np.array(out_e), np.array(out_c)
    dur = out_e - out_s
    keep = (dur >= MIN_DUR_S) & (dur <= MAX_DUR_S)
    return out_s[keep], out_e[keep], out_c[keep]


def frame_probs_dir(model: str, site: str) -> Path:
    return ROOT / model / site / "frame_probs" / site


def iter_frame_probs(model: str, site: str):
    """(filename, smoothed probs (n_frames, 3)) for every recording of a site."""
    for f in sorted(frame_probs_dir(model, site).glob("*.npz")):
        with np.load(f) as z:
            probs = z["probs"].astype(np.float32)
        yield f.stem + ".wav", smooth(probs)


# ----------------------------------------------------------------------- YOLO events

def yolo_boxes(site: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "yolo" / site / "boxes_raw.parquet")


def yolo_events(boxes: pd.DataFrame, win_s: float = 30.0, f_top: float = 125 - 5) -> pd.DataFrame:
    """Boxes -> events with the baseline's arithmetic (same as aivis.models.yolo.boxes_to_rows)."""
    b = boxes
    start = (b.x - b.w / 2 + b.window) * win_s
    end = b.w * win_s + start
    return pd.DataFrame({"dataset": b.dataset.values, "filename": b.filename.values, "cls": b.name.values,
                         "start_s": start.values, "end_s": end.values, "conf": b.conf.values,
                         "low_frequency": ((1 - (b.y + b.h / 2)) * f_top).clip(lower=0).values,
                         "high_frequency": ((1 - (b.y - b.h / 2)) * f_top).clip(upper=f_top).values,
                         "window": b.window.values})


def temporal_nms(ev: pd.DataFrame, iou: float = 0.5) -> pd.DataFrame:
    """Drop events that overlap a higher-confidence event of the same class and file by more than iou
    (YOLO's duplicates from the overlapping windows). Keeps the original row order."""
    ev = ev.reset_index(drop=True)
    keep = np.ones(len(ev), bool)
    for _, g in ev.groupby(["dataset", "filename", "cls"], sort=False):
        idx = g.index.to_numpy()
        order = idx[np.argsort(-g["conf"].to_numpy(), kind="stable")]
        s, e = ev.loc[order, "start_s"].to_numpy(), ev.loc[order, "end_s"].to_numpy()
        alive = np.ones(len(order), bool)
        for i in range(len(order)):
            if not alive[i]:
                continue
            inter = np.clip(np.minimum(e[i], e[i + 1:]) - np.maximum(s[i], s[i + 1:]), 0, None)
            union = (e[i] - s[i]) + (e[i + 1:] - s[i + 1:]) - inter
            alive[i + 1:] &= ~(inter / np.maximum(union, 1e-9) > iou)
        keep[order[~alive]] = False
    return ev[keep]


# ---------------------------------------------------------------------- event grids

def grid_path(model: str, site: str) -> Path:
    return TABLES / "grid" / f"{model}_{site}.parquet"


def add_abs(ev: pd.DataFrame) -> pd.DataFrame:
    fe = epochs(ev["filename"])
    ev["t0"] = fe + ev["start_s"].to_numpy()
    ev["t1"] = fe + ev["end_s"].to_numpy()
    return ev


def events_at(model: str, site: str, thresholds, nms: bool = False) -> pd.DataFrame:
    """Events of one model and site at per-class thresholds [bmabz, d, bp], in the row order the
    detections.csv of that run would have. Frame models need thresholds on the grid."""
    thr = dict(zip(CLASSES, thresholds))
    if model == "yolo":
        ev = yolo_events(yolo_boxes(site))
        ev = ev[ev["conf"] >= ev["cls"].map(thr)].reset_index(drop=True)
        if nms:
            ev = temporal_nms(ev).reset_index(drop=True)
    else:
        g = pd.read_parquet(grid_path(model, site))
        parts = []
        for c, t in thr.items():
            t = float(GRID[np.argmin(np.abs(GRID - t))])
            parts.append(g[(g["cls"] == c) & (np.isclose(g["thr"], t))])
        ev = pd.concat(parts)
        # postprocess.events orders by (file, class in first-seen order of the raw runs, start)
        ev = ev.sort_values(["filename", "cls_order", "start_s"], kind="stable").reset_index(drop=True)
        ev = ev.drop(columns=["thr", "cls_order"])
    ev["model"] = model
    return add_abs(ev)


# -------------------------------------------------------------------- official matching

def official_match(pred: pd.DataFrame, gt: pd.DataFrame, iou_thr: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Same result as evaluation_submissions.run of the task repo, much faster.

    pred: dataset, filename, cls, t0, t1 in the order of the csv rows. gt: cls, t0, t1 in file order.
    Returns (gt index matched by each prediction or -1, prediction index that matched each label or -1).
    Rules copied from the scorer: files in sorted (dataset, filename) order; labels whose start lies
    within 2 min of the file's predictions (any site); per class, each prediction in row order takes the
    first label with the highest IoU among the labels not detected yet, if that IoU is above iou_thr.
    """
    pred_match = np.full(len(pred), -1, dtype=np.int64)
    gt_match = np.full(len(gt), -1, dtype=np.int64)
    if len(pred) == 0 or len(gt) == 0:
        return pred_match, gt_match
    g_t0, g_t1, g_cls = gt["t0"].to_numpy(), gt["t1"].to_numpy(), gt["cls"].to_numpy()
    order = np.argsort(g_t0, kind="stable")
    sorted_t0 = g_t0[order]
    p_t0, p_t1, p_cls = pred["t0"].to_numpy(), pred["t1"].to_numpy(), pred["cls"].to_numpy()
    keys = pred["dataset"].astype(str) + "\x00" + pred["filename"].astype(str)
    for _, rows in pd.Series(np.arange(len(pred))).groupby(keys.to_numpy(), sort=True):
        rows = rows.to_numpy()
        lo = p_t0[rows].min() - 120.0
        hi = p_t1[rows].max() + 120.0
        a, b = np.searchsorted(sorted_t0, lo, "left"), np.searchsorted(sorted_t0, hi, "right")
        cand = np.sort(order[a:b])                      # back to gt order
        if len(cand) == 0:
            continue
        for c in np.unique(p_cls[rows]):
            cc = cand[(g_cls[cand] == c) & (gt_match[cand] < 0)]
            if len(cc) == 0:
                continue
            free = np.ones(len(cc), bool)
            gs, ge = g_t0[cc], g_t1[cc]
            for r in rows[p_cls[rows] == c]:
                if not free.any():
                    break
                inter = np.clip(np.minimum(p_t1[r], ge) - np.maximum(p_t0[r], gs), 0, None)
                union = (p_t1[r] - p_t0[r]) + (ge - gs) - inter
                iou = np.where(free, inter / union, -np.inf)
                j = int(np.argmax(iou))
                if iou[j] > iou_thr:
                    free[j] = False
                    pred_match[r] = cc[j]
                    gt_match[cc[j]] = r
    return pred_match, gt_match


def counts(pred: pd.DataFrame, gt: pd.DataFrame, pm: np.ndarray, gm: np.ndarray, by=("dataset", "cls")) -> pd.DataFrame:
    """tp, fp, fn per site and class, the way the scorer counts them (tp from labels, fp from predictions)."""
    by = list(by)
    g = gt.assign(tp=(gm >= 0).astype(int), fn=(gm < 0).astype(int)).groupby(by)[["tp", "fn"]].sum()
    p = pred.assign(fp=(pm < 0).astype(int)).groupby(by)[["fp"]].sum()
    t = g.join(p, how="outer").fillna(0).astype(int).reset_index()
    return t


def rates(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["precision"] = t.tp / (t.tp + t.fp)
        t["recall"] = t.tp / (t.tp + t.fn)
        t["f1"] = (2 * t.precision * t.recall / (t.precision + t.recall)).fillna(0)
    return t


def macro(t: pd.DataFrame, by="dataset") -> pd.DataFrame:
    """macro A (mean class F1) and macro B (F1 of mean P and R, classes without detections skipped in P)."""
    rows = []
    for k, g in t.groupby(by, sort=False):
        p, r = g.precision.mean(skipna=True), g.recall.mean(skipna=True)
        p, r = (0.0 if np.isnan(p) else p), (0.0 if np.isnan(r) else r)
        rows.append({by: k, "macro_A": g.f1.fillna(0).mean(), "macro_B": 0.0 if not p + r else 2 * p * r / (p + r)})
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame, gt: pd.DataFrame, iou_thr: float = 0.3, per_site: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-site-and-class table plus macro per site and for all sites joined, like aivis.evaluate --strict."""
    parts = []
    for ds, g in gt.groupby("dataset", sort=False):
        p = pred[pred["dataset"] == ds].reset_index(drop=True)
        g = g.reset_index(drop=True)
        pm, gm = official_match(p, g, iou_thr)
        parts.append(counts(p, g, pm, gm))
    t = pd.concat(parts, ignore_index=True)
    t = t.set_index(["dataset", "cls"]).reindex(
        pd.MultiIndex.from_product([gt["dataset"].unique(), CLASSES], names=["dataset", "cls"]), fill_value=0).reset_index()
    j = t.groupby("cls", sort=False)[["tp", "fp", "fn"]].sum().reset_index().assign(dataset="joined")
    t = rates(pd.concat([t, j[t.columns]], ignore_index=True))
    return t, macro(t)
