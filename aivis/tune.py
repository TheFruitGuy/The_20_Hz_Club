"""Tune the per-class thresholds of a Whale-VAD or NAVE run on labelled sites with NAVE's tuner,
two ways:
    in-sample  tune and score on all sites together (what most papers report, optimistic)
    loso       for each site, thresholds tuned on the other sites only (leave-one-site-out)

    python -m aivis.tune outputs/whalevad/validation
    python -m aivis.evaluate outputs/whalevad/validation/detections_loso.csv
    python -m aivis.evaluate outputs/whalevad/validation/detections_insample.csv

Writes thresholds.json, detections_insample.csv and detections_loso.csv into the run folder.
Note: NAVE's tuner optimises NAVE's own event matching (each label takes its best unmatched
detection, IoU >= 0.3), which is close to but not the same as the leaderboard scorer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate, io, paths, postprocess
from .recordings import file_start


def label_events(split: str, sites: list[str], recordings: list[str] | None):
    nt = postprocess.nave_train()
    gt = evaluate.load_labels(split, sites, recordings)
    start = pd.to_datetime(gt["start_datetime"], format="ISO8601", utc=True)
    end = pd.to_datetime(gt["end_datetime"], format="ISO8601", utc=True)
    t0 = pd.to_datetime(gt["filename"].map(file_start), utc=True)
    out = []
    for ds, fn, lab, s, e in zip(gt["dataset"], gt["filename"], gt["annotation"],
                                 (start - t0).dt.total_seconds(), (end - t0).dt.total_seconds()):
        out.append(nt.Detection(dataset=ds, filename=fn, label=paths.FINE_TO_CLASS.get(lab, lab),
                                start_s=float(s), end_s=float(e)))
    return out


def load_probs(run_dir: Path) -> dict:
    probs = {}
    for f in sorted((run_dir / "frame_probs").glob("*/*.npz")):
        probs[(f.parent.name, f.stem + ".wav", 0)] = io.load_frame_probs(f)["probs"].astype(np.float32)
    if not probs:
        raise SystemExit(f"no frame probabilities in {run_dir / 'frame_probs'} (run predict without --no-frame-probs)")
    return probs


def detections(probs: dict, thresholds_for) -> pd.DataFrame:
    rows = []
    for (ds, fn, _), p in probs.items():
        rows += postprocess.events(p, thresholds_for(ds), ds, fn)
    return io.detections_frame(rows)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run folder of whalevad or nave on a labelled split")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args(argv)
    info = json.loads((a.run / "run.json").read_text())
    if info["model"] not in ("whalevad", "nave"):
        raise SystemExit("tune works on Whale-VAD and NAVE runs")
    split, sites = info["split"], info["sites"]
    nt = postprocess.nave_train()
    probs = load_probs(a.run)
    gt = label_events(split, sites, info.get("recordings"))
    print(f"{len(probs)} recordings, {len(gt)} labels, sites {sites}")

    thr_all = [float(t) for t in nt.tune_thresholds_per_class(probs, gt, workers=a.workers)]
    print("in-sample thresholds", dict(zip(paths.CLASSES, thr_all)))
    loso = {}
    if len(sites) > 1:
        for s in sites:
            tr_p = {k: v for k, v in probs.items() if k[0] != s}
            tr_g = [g for g in gt if g.dataset != s]
            loso[s] = [float(t) for t in nt.tune_thresholds_per_class(tr_p, tr_g, workers=a.workers)]
            print(f"held out {s}: thresholds", dict(zip(paths.CLASSES, loso[s])))

    io.write_detections(detections(probs, lambda ds: thr_all), a.run / "detections_insample.csv")
    if loso:
        io.write_detections(detections(probs, lambda ds: loso[ds]), a.run / "detections_loso.csv")
    result = {"in_sample": dict(zip(paths.CLASSES, thr_all)),
              "loso": {s: dict(zip(paths.CLASSES, t)) for s, t in loso.items()},
              "tuner": "nave_train.tune_thresholds_per_class (NAVE's matching, IoU >= 0.3)",
              "postprocess": postprocess.settings()}
    (a.run / "thresholds.json").write_text(json.dumps(result, indent=2))
    print(f"-> {a.run / 'thresholds.json'}; score with python -m aivis.evaluate {a.run / 'detections_loso.csv'}")
    return result


if __name__ == "__main__":
    main()
