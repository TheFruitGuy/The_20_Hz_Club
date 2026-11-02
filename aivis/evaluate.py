"""Score detections with the official leaderboard scorer (evaluation/evaluation_submissions.py of the
task repo, not evaluation.py, which silently drops the D class).

    python -m aivis.evaluate outputs/nave/validation
    python -m aivis.evaluate some/detections.csv --split validation --sites casey2017

Prints tp, fp, fn, precision, recall and F1 per site and class, plus
    macro A  mean of the three per-class F1 scores
    macro B  F1 of the mean precision and mean recall (what the leaderboard ranks by; like the
             leaderboard, a class without any detection is left out of the mean precision)
and writes metrics.csv and metrics.json next to the detections.

Scorer rules worth knowing: IoU above 0.3, greedy matching in the order of the prediction rows,
labels are looked up in a +-2 min window around each file's predictions, and a second detection of
the same call counts as a false positive. The window is not restricted to the same site, so if two
sites overlap in time, --strict scores every site on its own.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

_OFFICIAL = None


def official():
    global _OFFICIAL
    if _OFFICIAL is None:
        f = paths.repo("task2") / "evaluation" / "evaluation_submissions.py"
        spec = importlib.util.spec_from_file_location("task2_evaluation_submissions", f)
        _OFFICIAL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OFFICIAL)
    return _OFFICIAL


def load_labels(split: str, sites: list[str], recordings: list[str] | None = None) -> pd.DataFrame:
    folder = paths.annotations_dir(split)
    if not folder.is_dir():
        raise SystemExit(f"no labels for split {split!r} ({folder} does not exist)")
    parts = [pd.read_csv(folder / f"{s}.csv") for s in sites if (folder / f"{s}.csv").exists()]
    if not parts:
        raise SystemExit(f"no label files for {sites} in {folder}")
    gt = pd.concat(parts, ignore_index=True)
    if recordings:  # a run on some files only: keep the labels of those files
        keep = {tuple(r.split("/", 1)) for r in recordings}
        gt = gt[[(d, f) in keep for d, f in zip(gt["dataset"], gt["filename"])]]
    return gt


def run_official(pred: pd.DataFrame, gt: pd.DataFrame, iou: float, conf_threshold) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        pp, gp = Path(tmp) / "pred.csv", Path(tmp) / "gt.csv"
        pred.to_csv(pp, index=False)
        gt.to_csv(gp, index=False)
        with contextlib.redirect_stdout(io.StringIO()):   # the scorer prints a debug line
            res = official().run(pp, gp, iou_threshold=iou, conf_threshold=conf_threshold)
    res = res.reset_index(drop=True)
    res["class"] = ["bmabz", "d", "bp"] * (len(res) // 3)   # the scorer's row order: bmabz, fm (= d), bp
    for c in ("tp", "fp", "fn"):
        res[c] = res[c].astype(int)
    return res[["dataset", "class", "tp", "fp", "fn"]]


def add_rates(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["precision"] = t.tp / (t.tp + t.fp)
        t["recall"] = t.tp / (t.tp + t.fn)
        t["f1"] = 2 * t.precision * t.recall / (t.precision + t.recall)
    return t


def macro(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in t.groupby("dataset", sort=False):
        p, r = g.precision.mean(skipna=True), g.recall.mean(skipna=True)
        p, r = (0.0 if np.isnan(p) else p), (0.0 if np.isnan(r) else r)   # no detections at all
        rows.append({"dataset": ds, "macro_A": g.f1.fillna(0).mean(),
                     "macro_B": 0.0 if not (p + r) else 2 * p * r / (p + r),
                     "classes_without_detections": ",".join(g[(g.tp + g.fp) == 0]["class"])})
    return pd.DataFrame(rows)


def overlapping_sites(gt: pd.DataFrame) -> list[tuple[str, str]]:
    t = pd.to_datetime(gt["start_datetime"], format="ISO8601", utc=True)
    span = t.groupby(gt["dataset"]).agg(["min", "max"])
    names = list(span.index)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if span.loc[a, "min"] <= span.loc[b, "max"] and span.loc[b, "min"] <= span.loc[a, "max"]]


def evaluate(pred: pd.DataFrame, gt: pd.DataFrame, iou: float = 0.3, conf_threshold=None,
             strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strict:
        parts = [run_official(pred[pred.dataset == ds], g, iou, conf_threshold).query("dataset != 'joined'")
                 for ds, g in gt.groupby("dataset")]
        t = pd.concat(parts, ignore_index=True)
        joined = t.groupby("class", sort=False)[["tp", "fp", "fn"]].sum().reset_index().assign(dataset="joined")
        t = pd.concat([t, joined[t.columns]], ignore_index=True)
    else:
        t = run_official(pred, gt, iou, conf_threshold)
    t = add_rates(t)
    return t, macro(t)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("detections", type=Path, help="a run folder (with detections.csv) or a csv file")
    p.add_argument("--split", choices=("train", "validation"), help="labels to score against (default from run.json)")
    p.add_argument("--sites", nargs="+", help="score only these sites")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--conf-threshold", type=float, help="only for labels with a relative_confidence column")
    p.add_argument("--strict", action="store_true", help="score each site separately (no matching across sites)")
    p.add_argument("--name", help="file name stem for the results (default metrics, metrics_loso, ...)")
    a = p.parse_args(argv)

    csv = a.detections / "detections.csv" if a.detections.is_dir() else a.detections
    run_info = {}
    if (csv.parent / "run.json").exists():
        run_info = json.loads((csv.parent / "run.json").read_text())
    a.name = a.name or "metrics" + csv.stem.removeprefix("detections")
    pred = pd.read_csv(csv)
    split = a.split or run_info.get("split")
    if split not in ("train", "validation"):
        raise SystemExit("give --split train or --split validation (the evaluation split has no labels)")
    sites = a.sites or run_info.get("sites") or sorted(pred["dataset"].unique())
    gt = load_labels(split, sites, run_info.get("recordings"))
    pred = pred[pred["dataset"].isin(sites)]

    clash = overlapping_sites(gt)
    if clash and not a.strict:
        print(f"Note: these sites overlap in time, so the scorer may match across them: {clash}. "
              f"Use --strict to score each site on its own.")
    table, summary = evaluate(pred, gt, a.iou, a.conf_threshold, a.strict)

    pd.set_option("display.width", 120)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = csv.parent
    table.to_csv(out / f"{a.name}.csv", index=False)
    result = {"detections": str(csv), "split": split, "sites": sites, "iou": a.iou,
              "conf_threshold": a.conf_threshold, "strict": a.strict, "n_labels": int(len(gt)),
              "n_detections": int(len(pred)), "macro": summary.to_dict("records")}
    (out / f"{a.name}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n-> {out / (a.name + '.csv')}, {out / (a.name + '.json')}")
    return result


if __name__ == "__main__":
    main()
