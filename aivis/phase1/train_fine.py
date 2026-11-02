"""Retrain NAVE with 5 outputs per call type (aivis.nave_heads "fine5") plus the frequency-shift augmentation,
for test F1 to F4.

    python -m aivis.phase1.train_fine --seed 0 [--workers 8]
    python -m aivis.phase1.train_fine --seed 0 --epochs 1 --out /tmp/fine_test      # quick check

Everything else is the AUG recipe of aivis.phase1.train_aug (NAVE's own training, kernel 153, single-sideband
shifts of -1.5 to +1.5 Hz on 80 % of the training segments), so FINE differs from AUG only in the output head.
The 5 outputs are trained with their own class weights (w_c = N / P_c over the fine labels); validation,
threshold tuning and checkpoint selection merge them back to the 3 official classes (maximum within a class).

Runs go to outputs/phase1/fine_runs/nave_s<seed>_pcen1_fdy1_k153_fine5_fshift1.5_<time>/ (nave_best.pt etc.).
The checkpoint's config records heads = "fine5".
"""
from __future__ import annotations

import argparse
import os

from .. import nave_heads, paths
from . import core, train_aug

FINE_ROOT = core.ROOT / "fine_runs"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-shift-hz", type=float, default=1.5)
    p.add_argument("--p", type=float, default=0.8, help="share of training segments that get shifted")
    p.add_argument("--workers", type=int, default=8, help="data loader workers (NAVE's default is 16)")
    p.add_argument("--epochs", type=int, help="override NAVE's 40 epochs (only for a quick test run)")
    p.add_argument("--out", help="other output folder (for a quick test run)")
    a = p.parse_args(argv)

    os.environ.setdefault("NAVE_DATA_ROOT", str(paths.DEV))
    paths.use_repo("nave")
    import nave_config as cfg     # noqa: E402

    nave_heads.patch_training()
    orig_tag, orig_conf = cfg.ablation_tag, cfg.ablation_config
    cfg.ablation_tag = lambda: orig_tag() + "_fine5"
    cfg.ablation_config = lambda: {**orig_conf(), "heads": "fine5"}
    print(f"[train_fine] seed {a.seed}, outputs {nave_heads.FINE5}", flush=True)

    args = ["--seed", str(a.seed), "--max-shift-hz", str(a.max_shift_hz), "--p", str(a.p),
            "--workers", str(a.workers), "--out", str(a.out or FINE_ROOT)]
    if a.epochs:
        args += ["--epochs", str(a.epochs)]
    train_aug.main(args)


if __name__ == "__main__":
    main()
