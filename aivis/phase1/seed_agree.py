"""Exploratory follow-up to the five-seed analysis (S1 to S6): do the seeds miss the same calls as the other
architectures, and is agreement between seeds a usable confidence signal on a new site?

    python -m aivis.phase1.seed_agree [--workers 3]      # CPU only, needs outputs/phase1/seeds/seed*/val_2s

For every NAVE seed: events at that seed's own LOSO thresholds (published 2 s pipeline, as in S1), matched
with the official rules.

shared misses   per dev label, how many of the five seeds hit it; for the labels all five seeds miss, the
                share that YOLO and Whale-VAD (Phase I, LOSO) also miss, against the same share over all
                labels. A high overlap points at the labels or the data rather than at one architecture
                (C11) and gives the sample for the listening check.
agreement       for every event of the released seed (666), how many of the other four seeds have an
                overlapping event of the same class. Precision per agreement level, per class and site.
transfer        precision per agreement level learnt on two dev sites, used to estimate the precision of the
                third site from its agreement counts alone; error against the real precision. Phase I did
                the same with agreement between the three models and was off by up to 0.4.
Tables: seed_agree_labels, seed_agree_shared, seed_agree_precision, seed_agree_transfer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import core, seeds
from .core import CLASSES, TABLES

VAL = seeds.VAL
ANCHOR = 666


def say(m):
    print(f"[seed_agree] {m}", flush=True)


def seed_events(seed: int) -> pd.DataFrame:
    """Events of one seed at its LOSO thresholds on the three dev sites."""
    gt = core.load_labels("validation")
    d = seeds.run_dirs(seed)["2s"]
    evs = {s: seeds.grid_events(d / s / "frame_probs" / s, s) for s in VAL}
    loso, _ = seeds.tune(seeds.sweep(evs, gt))
    ev = pd.concat([seeds.at(evs[s], loso[s]) for s in VAL], ignore_index=True)
    return ev.assign(seed=seed)


def match(ev: pd.DataFrame, gt: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """gt_match per event (label_id or -1) and hit per label_id."""
    em = np.full(len(ev), -1)
    hit = pd.Series(False, index=gt.label_id)
    for s in VAL:
        for c in CLASSES:
            ie = np.flatnonzero(((ev.dataset == s) & (ev.cls == c)).to_numpy())
            g = gt[(gt.dataset == s) & (gt.cls == c)].reset_index(drop=True)
            p = ev.iloc[ie].reset_index(drop=True)
            pm, gm = core.official_match(p, g, 0.3)
            em[ie] = np.where(pm >= 0, g.label_id.to_numpy()[np.maximum(pm, 0)], -1)
            hit.loc[g.label_id[gm >= 0]] = True
    return em, hit


def agreement(anchor: pd.DataFrame, evs: dict[int, pd.DataFrame], anchor_seed: int = ANCHOR) -> np.ndarray:
    """For every anchor event, how many of the other seeds have an overlapping event of the same class."""
    agree = np.zeros(len(anchor), int)
    for s, o in evs.items():
        if s == anchor_seed:
            continue
        by = {k: g for k, g in o.groupby(["dataset", "filename", "cls"])}
        for i, r in enumerate(anchor.itertuples()):
            g = by.get((r.dataset, r.filename, r.cls))
            if g is not None and ((g.t0 < r.t1) & (g.t1 > r.t0)).any():
                agree[i] += 1
    return agree


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args(argv)
    gt = core.load_labels("validation")
    seed_list = list(seeds.SEEDS)
    with ProcessPoolExecutor(min(a.workers, len(seed_list))) as ex:
        evs = dict(zip(seed_list, ex.map(seed_events, seed_list)))
    say("events of all seeds at their LOSO thresholds")
    hits, gmatch = {}, {}
    for s, ev in evs.items():
        gmatch[s], hits[s] = match(ev, gt)
    H = pd.DataFrame({f"hit_s{s}": hits[s] for s in seed_list})
    H["n_seeds_hit"] = H.sum(axis=1)
    lab = pd.read_parquet(TABLES / "labels_wide_loso.parquet").set_index("label_id")
    H = H.join(lab[["dataset", "cls", "fine", "snr_db", "hit_yolo", "hit_whalevad", "hit_nave"]])
    H.reset_index().rename(columns={"index": "label_id"}).to_parquet(TABLES / "seed_agree_labels.parquet")

    # 4: shared misses
    rows = []
    for c in CLASSES + ["all"]:
        h = H if c == "all" else H[H.cls == c]
        miss5 = h.n_seeds_hit == 0
        p_miss = 1 - h[[f"hit_s{s}" for s in seed_list]].mean()
        other_miss = ~h.hit_yolo.astype(bool) & ~h.hit_whalevad.astype(bool)
        rows.append({"cls": c, "labels": len(h), "missed_by_all_5_seeds": float(miss5.mean()),
                     "if_independent": float(np.prod(p_miss)),
                     "yolo_and_wvad_miss_given_5_seed_miss": float(other_miss[miss5].mean()) if miss5.any() else np.nan,
                     "yolo_and_wvad_miss_overall": float(other_miss.mean()),
                     "yolo_and_wvad_miss_given_5_seed_hit": float(other_miss[h.n_seeds_hit == 5].mean()) if (h.n_seeds_hit == 5).any() else np.nan,
                     "missed_by_all_7_models": float((miss5 & other_miss).mean()),
                     "median_snr_5_seed_miss": float(h.snr_db[miss5].median()),
                     "median_snr_all": float(h.snr_db.median())})
    S = pd.DataFrame(rows)
    S.to_parquet(TABLES / "seed_agree_shared.parquet")
    pd.set_option("display.width", 220)
    say("4) shared misses:\n" + S.round(3).to_string(index=False))

    # 5: agreement between seeds as confidence
    anchor = evs[ANCHOR].reset_index(drop=True)
    anchor["tp"] = gmatch[ANCHOR] >= 0
    anchor["agree"] = agreement(anchor, evs)
    P = anchor.groupby(["cls", "agree"]).agg(events=("tp", "size"), precision=("tp", "mean")).reset_index()
    P_site = anchor.groupby(["dataset", "cls", "agree"]).agg(events=("tp", "size"), precision=("tp", "mean")).reset_index()
    P_site.to_parquet(TABLES / "seed_agree_precision.parquet")
    say("5) precision of seed 666's events by the number of other seeds that agree:\n"
        + P.pivot(index="agree", columns="cls", values="precision").round(3).to_string()
        + "\nevents:\n" + P.pivot(index="agree", columns="cls", values="events").to_string())
    T = []
    for s in VAL:
        for c in CLASSES:
            tr = anchor[(anchor.dataset != s) & (anchor.cls == c)].groupby("agree").tp.mean()
            te = anchor[(anchor.dataset == s) & (anchor.cls == c)]
            if te.empty:
                continue
            est = float(te.agree.map(tr).fillna(tr.mean()).mean())
            T.append({"held_out": s, "cls": c, "events": len(te), "estimated_precision": est,
                      "real_precision": float(te.tp.mean()), "error": est - float(te.tp.mean())})
    T = pd.DataFrame(T)
    T.to_parquet(TABLES / "seed_agree_transfer.parquet")
    say("5) precision of a held-out site estimated from seed agreement learnt on the other two:\n"
        + T.round(3).to_string(index=False) + f"\nmean absolute error {T.error.abs().mean():.3f}, largest {T.error.abs().max():.3f}")


if __name__ == "__main__":
    main()
