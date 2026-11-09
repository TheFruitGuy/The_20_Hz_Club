"""Ideas 2 and 5: ranking under scoring variants, and agreement of the models on the new sites.

    python -m aivis.phase1.summaries ranking     # macro A/B per model under every scoring variant (C1)
    python -m aivis.phase1.summaries agreement   # regions of all events, dev precision per model combination,
                                                 # detections per hour, estimated precision on the eval sites
    python -m aivis.phase1.summaries audit       # blind audit sample (idea 2), with a separate key
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import paths
from . import anatomy, build, core
from .core import CLASSES, MODELS, TABLES

VAL = paths.SITES["validation"]
EVAL = paths.SITES["evaluation"]


def say(msg):
    print(f"[summaries] {msg}", flush=True)


# ---------------------------------------------------------------------------- ranking

def ranking(a) -> None:
    """12 variants (IoU 0.1/0.3/0.5 x macro A/B x in-sample/LOSO thresholds) plus the published default
    thresholds, and YOLO with temporal NMS. Thresholds are tuned at the IoU used for scoring."""
    gt = core.load_labels("validation")
    th = build.thresholds()
    rows = []
    for model in th:
        for nms in ([False, True] if model == "yolo" else [False]):
            name = model + ("_nms" if nms else "")
            for iou in build.IOUS:
                for variant in ("default", "insample", "loso"):
                    parts = []
                    for site in VAL:
                        t = th[model]["default"] if variant == "default" else (
                            th[model][f"insample_iou{iou}"] if variant == "insample" else th[model][f"loso_iou{iou}"][site])
                        parts.append(core.events_at(model, site, t, nms=nms))
                    ev = pd.concat(parts, ignore_index=True)
                    table, mac = core.score(ev, gt, iou)
                    for r in mac.itertuples():
                        rows.append({"model": name, "iou": iou, "thresholds": variant, "dataset": r.dataset,
                                     "macro_A": r.macro_A, "macro_B": r.macro_B})
                    for r in table.itertuples():
                        rows.append({"model": name, "iou": iou, "thresholds": variant, "dataset": r.dataset,
                                     "cls": r.cls, "tp": r.tp, "fp": r.fp, "fn": r.fn, "precision": r.precision,
                                     "recall": r.recall, "f1": r.f1})
            say(f"ranking: {name}")
    out = pd.DataFrame(rows)
    out.to_parquet(TABLES / "ranking.parquet")
    m = out[out.cls.isna() & (out.dataset == "joined")]
    say("macro B (joined dev sites):\n" + m.pivot_table(index=["iou", "thresholds"], columns="model",
                                                        values="macro_B").round(3).to_string())


# -------------------------------------------------------------------------- agreement

def agreement(a) -> None:
    gt = core.load_labels("validation")
    evs = {m: pd.read_parquet(build.events_path(m, "loso")) for m in MODELS if build.events_path(m, "loso").exists()}
    # regions of ALL events (dev and eval), so dev and eval are built the same way
    allev = {m: e.assign(gt_match=-1) for m, e in evs.items()}
    reg = anatomy.regions(allev, set())
    reg["kind"] = "all"
    # on dev: is there a label of the region's class inside it?
    reg["real"] = np.nan
    for (ds, c), g in reg[reg.dataset.isin(VAL)].groupby(["dataset", "cls"]):
        lab = gt[(gt.dataset == ds) & (gt.cls == c)].sort_values("t0")
        lt0, lt1 = lab.t0.to_numpy(), np.maximum.accumulate(lab.t1.to_numpy())
        # any label overlapping [t0, t1]: the last label starting before t1 must end after t0
        k = np.searchsorted(lt0, g.t1.to_numpy(), "left") - 1
        real = (k >= 0) & (lt1[np.maximum(k, 0)] > g.t0.to_numpy())
        reg.loc[g.index, "real"] = real.astype(float)
    reg["site_group"] = np.where(reg.dataset.str.startswith("ddu"), "DDU",
                                 np.where(reg.dataset.str.startswith("casey"), "Casey", "Kerguelen"))
    reg["day"] = pd.to_datetime(reg.t0, unit="s", utc=True).dt.floor("D")
    reg.to_parquet(TABLES / "regions_all.parquet")
    # dev precision per combination and class
    dev = reg[reg.dataset.isin(VAL)]
    cal = dev.groupby(["cls", "combo"]).real.agg(["mean", "size"]).rename(columns={"mean": "precision_dev", "size": "n_dev"})
    cal.reset_index().to_parquet(TABLES / "combo_precision_dev.parquet")
    say("dev precision by combination:\n" + cal.round(3).to_string())
    # detections per hour, share confirmed by another model, estimated precision on eval
    st = pd.read_parquet(TABLES / "site_stats.parquet")
    hours = st.groupby("dataset").duration_s.sum() / 3600
    rows = []
    for m in evs:
        r = reg[reg[m]]
        for (ds, c), g in r.groupby(["dataset", "cls"]):
            est = g.merge(cal.reset_index(), on=["cls", "combo"], how="left")["precision_dev"]
            rows.append({"model": m, "dataset": ds, "cls": c, "regions": len(g), "per_hour": len(g) / hours.get(ds, np.nan),
                         "confirmed_share": float((g.combo != m).mean()),
                         "all_three_share": float((g.combo == "+".join(evs)).mean()),
                         "est_precision": float(est.mean()), "dev_precision": float(g.real.mean()) if ds in VAL else np.nan})
    out = pd.DataFrame(rows)
    out.to_parquet(TABLES / "agreement.parquet")
    say("agreement:\n" + out.pivot_table(index=["dataset"], columns=["model"], values="est_precision").round(2).to_string())


# ------------------------------------------------------------------------------ audit

def audit(a) -> None:
    """15 regions per combination (all three, each model alone) x class x site group (DDU, Kerguelen) on
    the eval sites. The sheet has no model information; the key is a separate file."""
    reg = pd.read_parquet(TABLES / "regions_all.parquet")
    ev = reg[reg.dataset.isin(EVAL)]
    combos = ["+".join(MODELS)] + list(MODELS)
    rng = np.random.default_rng(42)
    picks = []
    for combo in combos:
        for c in CLASSES:
            for grp in ("DDU", "Kerguelen"):
                g = ev[(ev.combo == combo) & (ev.cls == c) & (ev.site_group == grp)]
                if len(g):
                    picks.append(g.sample(min(a.n, len(g)), random_state=int(rng.integers(1 << 31))))
    s = pd.concat(picks).sample(frac=1, random_state=7).reset_index(drop=True)
    s.insert(0, "audit_id", [f"A{i:03d}" for i in range(len(s))])
    s["start_s"] = s.t0 - core.epochs(s.filename)
    s["end_s"] = s.t1 - core.epochs(s.filename)
    sheet = s[["audit_id", "dataset", "filename", "cls", "start_s", "end_s"]].assign(
        call_present="", call_type="", confidence_1to3="", annotator="", note="")
    out = core.ROOT / "audit"
    out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out / "audit_sheet.csv", index=False)
    s[["audit_id", "combo", "site_group"] + [f"conf_{m}" for m in MODELS]].to_csv(out / "audit_key_DO_NOT_OPEN.csv", index=False)
    say(f"audit: {len(s)} regions -> {out / 'audit_sheet.csv'} (key in audit_key_DO_NOT_OPEN.csv)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["ranking", "agreement", "audit", "all"])
    p.add_argument("--n", type=int, default=15, help="audit: regions per cell")
    a = p.parse_args(argv)
    for s in (["ranking", "agreement", "audit"] if a.step == "all" else [a.step]):
        {"ranking": ranking, "agreement": agreement, "audit": audit}[s](a)


if __name__ == "__main__":
    main()
