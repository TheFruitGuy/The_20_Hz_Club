"""Everything for Phase I in one go, resumable: predictions, tables, anatomy, ranking, agreement,
audit sample and probes.

    caffeinate -i python -m aivis.phase1.run_all          # on a Mac, so it does not sleep
    python -m aivis.phase1.run_all --skip-probes          # without the probe suite (about 2 h less)
"""
from __future__ import annotations

import argparse
import time

from . import anatomy, build, predict_all, probes, summaries


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--skip-probes", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--calls-per-class", type=int, default=40)
    a = p.parse_args(argv)
    t0 = time.time()
    if not a.skip_predict:
        predict_all.main(["--device", a.device])
    build.main(["all"])
    anatomy.main([])
    summaries.main(["all"])
    if not a.skip_probes:
        probes.main(["all", "--device", a.device, "--calls-per-class", str(a.calls_per_class)])
    predict_all.log(f"run_all finished in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
