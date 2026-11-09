"""All three models on every validation and evaluation recording, one run folder per model and site.

    python -m aivis.phase1.predict_all                       # validation first, then evaluation
    python -m aivis.phase1.predict_all --splits validation   # only the labelled dev sites
    python -m aivis.phase1.predict_all --status              # what is done

Safe to stop and start again: a site counts as done once its run.json exists, an unfinished site
is redone. Output: outputs/phase1/<model>/<site>/ (detections.csv, frame_probs/ or boxes_raw.parquet,
run.json). Progress goes to outputs/phase1/progress.log.

Settings for the Phase I exports:
  YOLO      confidence 0.01 instead of 0.1, so thresholds can be swept afterwards; the baseline
            output is the subset with confidence >= 0.1
  Whale-VAD frame probabilities saved (all 7 outputs too), events at 0.5 (re-made at tuned thresholds later)
  NAVE      frame probabilities saved, events at the checkpoint thresholds
On a Mac, run it under caffeinate so the machine does not sleep:  caffeinate -i python -m aivis.phase1.predict_all
"""
from __future__ import annotations

import argparse
import shutil
import time
import traceback
from datetime import datetime

from .. import paths, predict

ROOT = paths.OUTPUTS / "phase1"
MODELS = ("yolo", "whalevad", "nave")
YOLO_CONF = 0.01


def run_dir(model: str, site: str):
    return ROOT / model / site


def done(model: str, site: str) -> bool:
    return (run_dir(model, site) / "run.json").exists()


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def jobs(splits, models, sites=None):
    for split in splits:
        for site in paths.SITES[split]:
            if sites and site not in sites:
                continue
            for model in models:
                yield split, site, model


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["validation", "evaluation"], choices=["validation", "evaluation", "train"])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--sites", nargs="+", help="only these sites")
    p.add_argument("--device", default="auto", help="auto (YOLO on MPS/CUDA, frame models on CUDA or CPU), cpu, cuda, mps")
    p.add_argument("--status", action="store_true", help="only show what is done")
    a = p.parse_args(argv)

    todo = list(jobs(a.splits, a.models, a.sites))
    if a.status:
        for split, site, model in todo:
            print(f"{split:10} {site:20} {model:9} {'done' if done(model, site) else '-'}")
        return
    free = shutil.disk_usage(paths.OUTPUTS.parent if not paths.OUTPUTS.exists() else paths.OUTPUTS).free / 1e9
    log(f"start: {len(todo)} jobs, {sum(not done(m, s) for _, s, m in todo)} to do, {free:.0f} GB free")
    if free < 15:
        log("WARNING: less than 15 GB free, the frame probabilities need about 1 GB per model and 100 h of audio")
    failed = []
    for split, site, model in todo:
        if done(model, site):
            continue
        t0 = time.time()
        log(f"{model} {site} ({split}) ...")
        args = [model, "--split", split, "--sites", site, "--out", str(run_dir(model, site)), "--overwrite",
                "--tag", "phase1", "--device", a.device]
        if model == "yolo":
            args += ["--conf", str(YOLO_CONF)]
        try:
            predict.main(args)
            log(f"{model} {site} done in {(time.time() - t0) / 60:.1f} min")
        except Exception:  # noqa: BLE001  keep going with the next site
            traceback.print_exc()
            log(f"{model} {site} FAILED, see the traceback above; continuing")
            failed.append(f"{model}/{site}")
    log("finished" + (f", failed: {', '.join(failed)}" if failed else ", all done"))


if __name__ == "__main__":
    main()
