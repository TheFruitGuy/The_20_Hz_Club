"""Run one of the models on a split, some sites or any folder of 250 Hz wav files.

    python -m aivis.predict nave --split validation
    python -m aivis.predict whalevad --split validation --sites casey2017 --limit-files 2
    python -m aivis.predict yolo --split evaluation --device cuda
    python -m aivis.predict whalevad --audio-dir /some/folder --dataset mysite

Writes outputs/<model>/<split>[_<tag>]/ with
    detections.csv       one row per detected event, challenge columns first (the scorer reads these)
    frame_probs/         Whale-VAD and NAVE: frame probabilities per recording (.npz)
    boxes_raw.parquet    YOLO: every box of every window before conversion
    run.json             settings, thresholds, weights hash, commits, package versions
and adds a line to outputs/runs.csv.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from . import io, paths
from .recordings import list_recordings

MODELS = ("yolo", "whalevad", "nave")


def pick_device(name: str, model: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if model == "yolo" and mps is not None and mps.is_available():
        return "mps"   # the frame models use complex STFT ops, safer on CPU
    return "cpu"


def parse_thresholds(text: str | None):
    if not text:
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) == 1:
        vals *= 3
    if len(vals) != 3:
        raise SystemExit("--thresholds needs one value or three (bmabz,d,bp)")
    return vals


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=MODELS)
    src = p.add_argument_group("input")
    src.add_argument("--split", choices=paths.SPLITS, default="validation")
    src.add_argument("--sites", nargs="+", help="only these sites of the split")
    src.add_argument("--audio-dir", type=Path, help="any folder of wav files instead of a split")
    src.add_argument("--dataset", help="site name to use with --audio-dir")
    src.add_argument("--limit-files", type=int, help="first N files per site, for quick tests")
    src.add_argument("--files", nargs="+", help="only these file names")
    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, help="output folder (default outputs/<model>/<split>[_<tag>])")
    out.add_argument("--tag", help="name of this run, added to the folder name")
    out.add_argument("--overwrite", action="store_true")
    out.add_argument("--no-frame-probs", action="store_true", help="frame models: do not save probabilities")
    run = p.add_argument_group("model")
    run.add_argument("--weights", type=Path, help="other weights than the default in weights/")
    run.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    run.add_argument("--batch-size", type=int)
    run.add_argument("--thresholds", help="frame models: bmabz,d,bp (one value for all three)")
    run.add_argument("--segment-s", type=float, help="frame models: tile length (Whale-VAD 30, NAVE 60)")
    run.add_argument("--overlap-s", type=float, default=2.0, help="frame models: tile overlap")
    run.add_argument("--cover-tail", action="store_true", help="frame models: extra tile at the file end")
    run.add_argument("--stitch", choices=["mean", "centre"], default="mean",
                     help="frame models: average overlapping tiles (default) or take each frame from its most central tile")
    run.add_argument("--conf", type=float, default=0.1, help="YOLO: confidence threshold (baseline 0.1)")
    run.add_argument("--imgsz", type=int, default=640, help="YOLO: image size (baseline 640)")
    run.add_argument("--save-images", action="store_true", help="YOLO: also write the spectrogram PNGs")
    a = p.parse_args(argv)

    recs = list_recordings(split=None if a.audio_dir else a.split, sites=a.sites, audio_dir=a.audio_dir,
                           dataset=a.dataset, limit_files=a.limit_files, files=a.files)
    if not recs:
        raise SystemExit("no wav files found")
    where = a.dataset or (a.audio_dir.name if a.audio_dir else a.split)
    out_dir = a.out or paths.OUTPUTS / a.model / (where + (f"_{a.tag}" if a.tag else ""))
    device = pick_device(a.device, a.model)
    args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    r = io.Run(out_dir, a.model, where, args, overwrite=a.overwrite)

    kw = {"weights": a.weights, "save_probs": not a.no_frame_probs, "thresholds": parse_thresholds(a.thresholds),
          "overlap_s": a.overlap_s, "cover_tail": a.cover_tail, "stitch": a.stitch, "conf": a.conf, "imgsz": a.imgsz,
          "save_images": a.save_images}
    if a.batch_size:
        kw["batch_size"] = a.batch_size
    if a.segment_s:
        kw["segment_s"] = a.segment_s
    if a.model == "yolo":
        from .models import yolo as m
    elif a.model == "whalevad":
        from .models import whalevad as m
    else:
        from .models import nave as m

    print(f"{a.model} on {len(recs)} recordings from {where}, device {device}, output {out_dir}")
    t0 = time.time()
    rows, settings = m.predict(recs, r, device, **kw)
    det = io.detections_frame(rows)
    io.write_detections(det, r.dir / "detections.csv")
    sites = sorted({x.dataset for x in recs})
    r.finish(device=device, sites=sites, n_files=len(recs), recordings=[f"{x.dataset}/{x.filename}" for x in recs],
             audio_hours=round(sum(x.duration_s for x in recs) / 3600, 2), n_detections=len(det),
             seconds=round(time.time() - t0, 1), weights=settings.get("weights"),
             weights_sha256=settings.get("weights_sha256"), settings=settings)
    counts = det["annotation"].value_counts().to_dict() if len(det) else {}
    print(f"{len(det)} detections {counts} in {time.time() - t0:.0f} s -> {r.dir / 'detections.csv'}")
    return r.dir


if __name__ == "__main__":
    main()
