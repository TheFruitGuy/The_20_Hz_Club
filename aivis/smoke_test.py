"""Quick check that all three models run end to end, on a synthetic 3 min recording (no data needed).

    python -m aivis.smoke_test           # synthetic recording, CPU
    python -m aivis.smoke_test --real    # also the first file of casey2017 (needs the dev data)

The scores on the synthetic file mean nothing; this only tests that nothing crashes and that the
outputs have the expected format.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import evaluate, paths, predict

FS = 250


def _sweep(f0, f1, dur):
    t = np.arange(int(dur * FS)) / FS
    env = np.minimum(1, np.minimum(t, dur - t) / 0.15)
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)) * env


def synthetic(folder: Path) -> pd.DataFrame:
    """Writes audio/synthetic/2020-01-01T00-00-00_000.wav and returns its labels."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, 180 * FS)
    calls = [("bmz", 20.0, np.concatenate([_sweep(27, 27, 8), _sweep(27, 19, 2), _sweep(19, 19, 8)])),
             ("bmd", 70.0, _sweep(90, 30, 2.5)), ("bp20", 110.0, _sweep(28, 15, 1.0)),
             ("bp20", 123.0, _sweep(28, 15, 1.0))]
    rows = []
    for lab, t0, c in calls:
        i = int(t0 * FS)
        x[i:i + len(c)] += 0.1 * c
        s = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(t0, unit="s")
        e = s + pd.to_timedelta(len(c) / FS, unit="s")
        rows.append({"dataset": "synthetic", "filename": "2020-01-01T00-00-00_000.wav", "annotation": lab,
                     "start_datetime": s.isoformat(), "end_datetime": e.isoformat()})
    wav = folder / "audio" / "synthetic" / "2020-01-01T00-00-00_000.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), (x * 32767).astype(np.int16), FS, subtype="PCM_16")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    os.environ["AIVIS_NO_REGISTRY"] = "1"
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labels = synthetic(tmp)
        jobs = [(m, ["--audio-dir", str(tmp / "audio" / "synthetic"), "--dataset", "synthetic"], labels)
                for m in predict.MODELS]
        if a.real:
            casey = paths.audio_dir("validation") / "casey2017"
            if casey.is_dir():
                jobs += [(m, ["--split", "validation", "--sites", "casey2017", "--limit-files", "1"], None)
                         for m in predict.MODELS]
            else:
                print(f"--real: {casey} not found, skipped")
        for k, (model, extra, gt) in enumerate(jobs):
            weights = paths.weights({"yolo": "yolo_baseline"}.get(model, model))
            if not weights.exists():
                print(f"[skip] {model}: {weights.name} missing")
                failed.append(f"{model} (no weights)")
                continue
            out = tmp / "out" / f"{k}_{model}"
            t0 = time.time()
            try:
                run_dir = predict.main([model, *extra, "--out", str(out), "--device", a.device])
                det = pd.read_csv(run_dir / "detections.csv")
                assert list(det.columns[:5]) == ["dataset", "filename", "annotation", "start_datetime", "end_datetime"]
                if gt is not None:
                    table, summary = evaluate.evaluate(det, gt)
                    score = f"synthetic macro B {summary.iloc[-1].macro_B:.2f}"
                else:
                    score = "real file"
                print(f"[ok]   {model}: {len(det)} detections, {score}, {time.time() - t0:.0f} s")
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                print(f"[FAIL] {model}")
                failed.append(model)
    if failed:
        print("Smoke test problems:", ", ".join(failed))
        return 1
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
