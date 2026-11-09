"""Idea 1, error anatomy: an outcome for every label and every event of every model, plus the regions
where models fire (for the UpSet views and the agreement analysis of idea 2).

    python -m aivis.phase1.anatomy [--variant loso]

Label outcomes (per model, first rule that applies):
    hit            matched by the official rules (IoU > 0.3, greedy)
    fragmented     two or more events of the class overlap it and together reach IoU > 0.3
    merged         an overlapping event of the class also covers another label of the class, or was
                   matched to another label
    near_miss      best IoU with one event of the class between 0.1 and 0.3
    wrong_class    no event of its class, but one of another class with IoU > 0.3
    missed_almost  peak score inside the label at least half the threshold (frame models: smoothed
                   probability; YOLO: best box confidence with IoU > 0.1)
    missed_silent  everything else
Event outcomes:
    tp             matched
    duplicate      IoU > 0.3 with a label of its class that another event of the same model already hit
    boundary       overlaps a label of its class, but not enough (fragments, merges, shifted edges)
    cross_class    overlaps a label of another class
    near_call      no overlap, but a label of its class within 10 s (inside a bout or sequence)
    background     none of these
Regions: per file and class, the union of events (gaps under 1 s closed) with the models that fire in it.
On dev sites only false-alarm events (not tp) count, on evaluation sites all events.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import paths
from . import build, core
from .core import CLASSES, MODELS, TABLES

LABEL_OUTCOMES = ["hit", "fragmented", "merged", "near_miss", "wrong_class", "missed_almost", "missed_silent"]
EVENT_OUTCOMES = ["tp", "duplicate", "boundary", "cross_class", "near_call", "background"]
NEAR_S = 10.0
REGION_GAP_S = 1.0


def overlap(a0, a1, b0, b1):
    """inter and IoU of every a against every b (2-D arrays)."""
    inter = np.clip(np.minimum(a1[:, None], b1[None, :]) - np.maximum(a0[:, None], b0[None, :]), 0, None)
    union = (a1 - a0)[:, None] + (b1 - b0)[None, :] - inter
    return inter, inter / np.maximum(union, 1e-9)


def union_iou(l0, l1, e0, e1) -> float:
    """IoU of a label with the union of some events."""
    if len(e0) == 0:
        return 0.0
    o = np.argsort(e0)
    s, e = e0[o], e1[o]
    ms, me = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= me[-1]:
            me[-1] = max(me[-1], b)
        else:
            ms.append(a)
            me.append(b)
    ms, me = np.array(ms), np.array(me)
    inter = np.clip(np.minimum(l1, me) - np.maximum(l0, ms), 0, None).sum()
    union = (l1 - l0) + (me - ms).sum() - inter
    return float(inter / union)


def load_peaks(model: str) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted((TABLES / "peaks").glob(f"{model}_*.parquet"))]
    return pd.concat(parts)[["label_id", "peak"]] if parts else pd.DataFrame(columns=["label_id", "peak"])


def analyse_model(model: str, gt: pd.DataFrame, ev: pd.DataFrame, variant: str):
    """Label outcomes and event outcomes of one model on the dev sites."""
    peaks = load_peaks(model).set_index("label_id")["peak"]
    lab_rows, ev_rows = [], []
    ev = ev[ev["dataset"].isin(gt["dataset"].unique())]
    hit_by = dict(zip(ev.loc[ev.gt_match >= 0, "gt_match"], ev.loc[ev.gt_match >= 0, "event_id"]))
    for (ds, fn), g in gt.groupby(["dataset", "filename"], sort=False):
        thr = dict(zip(CLASSES, build.thr_for(model, ds, variant)))
        e_file = ev[(ev.dataset == ds) & (ev.filename == fn)]
        # labels of this file and of the neighbours (events can reach a little beyond the file)
        g_near = gt[(gt.dataset == ds) & (gt.t1 > g.t0.min() - 60) & (gt.t0 < g.t1.max() + 60)]
        for c in CLASSES:
            gc = g[g.cls == c]
            if gc.empty:
                continue
            ec = e_file[e_file.cls == c]
            eo = e_file[e_file.cls != c]
            l0, l1 = gc.t0.to_numpy(), gc.t1.to_numpy()
            inter, iou = overlap(l0, l1, ec.t0.to_numpy(), ec.t1.to_numpy())
            _, iou_o = overlap(l0, l1, eo.t0.to_numpy(), eo.t1.to_numpy())
            # same-class labels nearby, to tell merges
            gnc = g_near[g_near.cls == c]
            inter_eg, _ = overlap(ec.t0.to_numpy(), ec.t1.to_numpy(), gnc.t0.to_numpy(), gnc.t1.to_numpy())
            n_labels_per_event = (inter_eg > 0).sum(axis=1)
            ec_match = ec.gt_match.to_numpy()
            for i, r in enumerate(gc.itertuples()):
                ov = inter[i] > 0
                best = float(iou[i].max()) if iou.shape[1] else 0.0
                peak = float(peaks.get(r.label_id, np.nan))
                if r.label_id in hit_by:
                    out = "hit"
                elif ov.sum() >= 2 and union_iou(r.t0, r.t1, ec.t0.to_numpy()[ov], ec.t1.to_numpy()[ov]) > 0.3:
                    out = "fragmented"
                elif ov.any() and ((n_labels_per_event[ov] >= 2).any() or ((ec_match[ov] >= 0) & (ec_match[ov] != r.label_id)).any()):
                    out = "merged"
                elif best >= 0.1:
                    out = "near_miss"
                elif iou_o.shape[1] and iou_o[i].max() > 0.3:
                    out = "wrong_class"
                elif peak >= thr[c] / 2:
                    out = "missed_almost"
                else:
                    out = "missed_silent"
                j = int(np.argmax(iou[i])) if iou.shape[1] else -1
                on = off = np.nan
                if j >= 0 and best > 0:
                    on = float(ec.t0.iloc[j] - r.t0)
                    off = float(ec.t1.iloc[j] - r.t1)
                lab_rows.append({"label_id": r.label_id, "model": model, "outcome": out, "best_iou": best,
                                 "n_overlap": int(ov.sum()), "peak": peak, "onset_err": on, "offset_err": off,
                                 "matched_event": hit_by.get(r.label_id, -1)})
    # events
    gt_by = {k: v for k, v in gt.groupby(["dataset", "filename"])}
    for (ds, fn), e in ev.groupby(["dataset", "filename"], sort=False):
        g = gt[(gt.dataset == ds) & (gt.t1 > e.t0.min() - 60) & (gt.t0 < e.t1.max() + 60)]
        for c, ec in e.groupby("cls"):
            gc, go = g[g.cls == c], g[g.cls != c]
            inter, iou = overlap(ec.t0.to_numpy(), ec.t1.to_numpy(), gc.t0.to_numpy(), gc.t1.to_numpy())
            inter_o, _ = overlap(ec.t0.to_numpy(), ec.t1.to_numpy(), go.t0.to_numpy(), go.t1.to_numpy())
            gap = np.full(len(ec), np.inf)
            if len(gc):
                gap = np.maximum(0, np.maximum(gc.t0.to_numpy()[None, :] - ec.t1.to_numpy()[:, None],
                                               ec.t0.to_numpy()[:, None] - gc.t1.to_numpy()[None, :])).min(axis=1)
            hit_labels = np.array([lid in hit_by for lid in gc.label_id]) if len(gc) else np.zeros(0, bool)
            for i, r in enumerate(ec.itertuples()):
                if r.gt_match >= 0:
                    out = "tp"
                elif len(gc) and ((iou[i] > 0.3) & hit_labels).any():
                    out = "duplicate"
                elif len(gc) and (inter[i] > 0).any():
                    out = "boundary"
                elif inter_o.shape[1] and (inter_o[i] > 0).any():
                    out = "cross_class"
                elif gap[i] <= NEAR_S:
                    out = "near_call"
                else:
                    out = "background"
                ev_rows.append({"event_id": r.event_id, "model": model, "outcome": out,
                                "best_iou": float(iou[i].max()) if iou.shape[1] else 0.0})
    return pd.DataFrame(lab_rows), pd.DataFrame(ev_rows)


def regions(evs: dict[str, pd.DataFrame], dev_sites: set[str]) -> pd.DataFrame:
    """Union of events per file and class (gaps under 1 s closed), with the models that fire in it.
    On dev sites only events that are not true positives."""
    parts = []
    for m, ev in evs.items():
        e = ev[~(ev.dataset.isin(dev_sites) & (ev.gt_match >= 0))]
        parts.append(e[["dataset", "filename", "cls", "t0", "t1", "conf"]].assign(model=m))
    allev = pd.concat(parts, ignore_index=True).sort_values(["dataset", "filename", "cls", "t0"])
    rows = []
    for (ds, fn, c), g in allev.groupby(["dataset", "filename", "cls"], sort=False):
        t0, t1, mm, cf = g.t0.to_numpy(), g.t1.to_numpy(), g.model.to_numpy(), g.conf.to_numpy()
        start, end, members = t0[0], t1[0], {mm[0]: cf[0]}
        for a, b, m, k in zip(t0[1:], t1[1:], mm[1:], cf[1:]):
            if a - end < REGION_GAP_S:
                end = max(end, b)
                members[m] = max(members.get(m, 0), k)
            else:
                rows.append((ds, fn, c, start, end, members))
                start, end, members = a, b, {m: k}
        rows.append((ds, fn, c, start, end, members))
    out = pd.DataFrame(rows, columns=["dataset", "filename", "cls", "t0", "t1", "members"])
    for m in evs:
        out[m] = out.members.map(lambda d, m=m: m in d)
        out[f"conf_{m}"] = out.members.map(lambda d, m=m: d.get(m, np.nan))
    out["combo"] = out[list(evs)].apply(lambda r: "+".join(k for k in evs if r[k]), axis=1)
    out["kind"] = np.where(out.dataset.isin(dev_sites), "false_alarm", "any")
    out["duration"] = out.t1 - out.t0
    return out.drop(columns="members").reset_index(drop=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="loso", choices=["default", "insample", "loso"])
    a = p.parse_args(argv)
    gt = pd.read_parquet(TABLES / "labels_validation.parquet") if (TABLES / "labels_validation.parquet").exists() \
        else core.load_labels("validation")
    evs = {m: pd.read_parquet(build.events_path(m, a.variant)) for m in MODELS if build.events_path(m, a.variant).exists()}
    labs, evouts = [], []
    for m, ev in evs.items():
        sites = set(ev.dataset) & set(paths.SITES["validation"])
        lo, eo = analyse_model(m, gt[gt.dataset.isin(sites)], ev, a.variant)
        labs.append(lo)
        evouts.append(eo.merge(ev.drop(columns=["model"]), on="event_id"))
        print(f"[anatomy] {m}: " + ", ".join(f"{k} {v}" for k, v in lo.outcome.value_counts().items()))
    lab = pd.concat(labs, ignore_index=True)
    lab.to_parquet(TABLES / f"label_outcomes_{a.variant}.parquet")
    pd.concat(evouts, ignore_index=True).to_parquet(TABLES / f"event_outcomes_{a.variant}.parquet")
    wide = lab.pivot(index="label_id", columns="model", values="outcome")
    hits = (wide == "hit").add_prefix("hit_")
    wide = gt.merge(wide.add_prefix("out_").join(hits), left_on="label_id", right_index=True)
    wide["hit_combo"] = wide[[f"hit_{m}" for m in evs]].apply(
        lambda r: "+".join(m for m in evs if r[f"hit_{m}"]) or "none", axis=1)
    wide.to_parquet(TABLES / f"labels_wide_{a.variant}.parquet")
    reg = regions(evs, set(paths.SITES["validation"]))
    reg.to_parquet(TABLES / f"regions_{a.variant}.parquet")
    print(f"[anatomy] labels by hit combination:\n{wide.hit_combo.value_counts().to_string()}")
    print(f"[anatomy] {len(reg)} regions -> {TABLES}")


if __name__ == "__main__":
    main()
