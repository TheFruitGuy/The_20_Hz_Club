"""F2d (ai-interactions/preregistration.md, 06.11.2026): the 2019/20 eval audio shifted DOWN by 0.6 Hz, the control for F2.

    python -m aivis.phase1.f2_down --device cuda:0                 # Whale-VAD only
    python -m aivis.phase1.f2_down --device cuda:0 --models whalevad nave yolo

Reuses the F2 code: single-sideband shift of whole recordings, same thresholds (tuned on all three dev sites).
Prints BMABZ detections per hour per site: as recorded, shifted up (F2 run) and shifted down (this run).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .. import paths, predict
from . import core
from . import followups as fu

SITES = ["kerguelen2019", "kerguelen2020"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["whalevad"], choices=core.MODELS)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    jobs = []
    for site in SITES:
        out = fu.ROOT / "audio" / f"{site}_down"
        out.mkdir(parents=True, exist_ok=True)
        src = paths.audio_dir(core.split_of(site)) / site
        jobs += [(w, out / w.name, -fu.SHIFT_HZ) for w in sorted(src.glob("*.wav"))]
    todo = [j for j in jobs if not j[1].exists()]
    fu.say(f"f2d: {len(todo)} of {len(jobs)} files to shift down")
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(fu.shift_file, todo, chunksize=4))

    for site in SITES:
        for m in a.models:
            out = fu.ROOT / "runs" / m / f"{site}_down"
            if (out / "run.json").exists():
                continue
            fu.say(f"f2d: {m} {site} shifted down")
            predict.main([m, "--audio-dir", str(fu.ROOT / "audio" / f"{site}_down"), "--dataset", site, "--out", str(out),
                          "--overwrite", "--tag", "followup", "--no-frame-probs", "--device", a.device,
                          *fu.thr_arg(m, site)])

    hours = pd.read_parquet(core.TABLES / "site_stats.parquet").groupby("dataset").duration_s.sum() / 3600
    rows = []
    for m in a.models:
        for site in SITES:
            runs = {"as recorded": fu.baseline(m, site), "shifted down": fu.load_run(m, f"{site}_down", site)}
            if (fu.ROOT / "runs" / m / f"{site}_shift" / "run.json").exists():
                runs["shifted up"] = fu.load_run(m, f"{site}_shift", site)
            for cond, ev in runs.items():
                for c in core.CLASSES:
                    rows.append({"model": m, "site": site, "condition": cond, "cls": c,
                                 "per_hour": float((ev.cls == c).sum() / hours[site])})
    t = pd.DataFrame(rows)
    t.to_parquet(core.TABLES / "f2d_eval_shift_down.parquet")
    piv = t[t.cls == "bmabz"].pivot_table(index=["model", "site"], columns="condition", values="per_hour").round(1)
    fu.say("f2d, BMABZ detections per hour:\n" + piv.to_string())
    tot = t[t.cls == "bmabz"].pivot_table(index="model", columns="condition", values="per_hour", aggfunc="sum")
    fu.say("f2d, factor against as recorded (both sites):\n" + tot.div(tot["as recorded"], axis=0).round(2).to_string())


if __name__ == "__main__":
    main()
