"""Retrain NAVE (full recipe, kernel 153) with another input representation, for tests M1 to M4.

    python -m aivis.phase1.train_input --variant nomag  --seed 0 [--workers 8]
    python -m aivis.phase1.train_input --variant logmag --seed 0

nomag   without the linear magnitude channel (cos, sin of the phase, PCEN)
logmag  log magnitude, mean-normalized per frequency bin over each tile, instead of the linear magnitude
Everything else is NAVE's own training (external/NAVE-SED/nave_train.py, unchanged). The variant is stored in
the checkpoint's config ("input"), so aivis.models.nave.load rebuilds the right front end.
Runs go to outputs/phase1/input_runs/nave_s<seed>_pcen1_fdy1_k153_<variant>_<time>/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .. import nave_input, paths
from . import core

INPUT_ROOT = core.ROOT / "input_runs"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=["nomag", "logmag"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=8, help="data loader workers (NAVE's default is 16)")
    p.add_argument("--epochs", type=int, help="override NAVE's 40 epochs (only for a quick test run)")
    p.add_argument("--out", help="other output folder (for a quick test run)")
    a = p.parse_args(argv)

    os.environ.setdefault("NAVE_DATA_ROOT", str(paths.DEV))
    paths.use_repo("nave")
    import nave_config as cfg     # noqa: E402
    import nave_train as nt       # noqa: E402

    nave_input.apply(a.variant)
    cfg.OUTPUT_DIR = Path(a.out) if a.out else INPUT_ROOT
    cfg.NUM_WORKERS = a.workers
    if a.epochs:
        cfg.EPOCHS = a.epochs
    orig_tag, orig_conf = cfg.ablation_tag, cfg.ablation_config
    cfg.ablation_tag = lambda: f"{orig_tag()}_{a.variant}"
    cfg.ablation_config = lambda: {**orig_conf(), "input": a.variant}
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[train_input] seed {a.seed}, input {a.variant}, runs in {cfg.OUTPUT_DIR}", flush=True)
    sys.argv = ["nave_train", "--seed", str(a.seed)]
    nt.main()


if __name__ == "__main__":
    main()
