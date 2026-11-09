"""Common output formats and the run record (outputs/runs.csv, made on the first run)."""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

# detections.csv: the challenge columns first (what the official scorer reads), then ours
DETECTION_COLUMNS = ["dataset", "filename", "annotation", "start_datetime", "end_datetime",
                     "start_s", "end_s", "confidence", "low_frequency", "high_frequency"]

HOP_S = 0.02            # frame step of Whale-VAD and NAVE
FRAME_LEN_S = 1.024     # STFT window (256 samples at 250 Hz)


def detections_frame(rows: list[dict]) -> pd.DataFrame:
    """rows need dataset, filename, annotation, start_s, end_s, confidence (+ optional frequencies)."""
    from .recordings import file_start
    df = pd.DataFrame(rows, columns=["dataset", "filename", "annotation", "start_s", "end_s", "confidence",
                                     "low_frequency", "high_frequency"])
    if df.empty:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    starts = df["filename"].map(file_start)
    df["start_datetime"] = [(t + timedelta(seconds=float(s))).isoformat(timespec="microseconds")
                            for t, s in zip(starts, df["start_s"])]
    df["end_datetime"] = [(t + timedelta(seconds=float(s))).isoformat(timespec="microseconds")
                          for t, s in zip(starts, df["end_s"])]
    return df[DETECTION_COLUMNS]


def write_detections(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.4f")


def read_detections(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_frame_probs(path: Path, probs: np.ndarray, classes: list[str], **extra: np.ndarray) -> None:
    """Per-recording frame probabilities. Frame f covers [f*HOP_S, f*HOP_S + FRAME_LEN_S) of the file;
    the time convention used for events is the frame start (same as NAVE's pipeline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probs=probs.astype(np.float16), classes=np.array(classes),
                        hop_s=HOP_S, frame_len_s=FRAME_LEN_S,
                        **{k: v.astype(np.float16) for k, v in extra.items()})


def load_frame_probs(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sha256(path: Path, n: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def _git(*args: str, cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                              capture_output=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return ""


def provenance() -> dict:
    ext = {}
    for key in paths.CONFIG["repos"]:
        p = paths.repo(key)
        if p.exists():
            ext[key] = _git("rev-parse", "HEAD", cwd=p)
    return {
        "aivis_commit": _git("rev-parse", "HEAD", cwd=paths.ROOT),
        "aivis_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", cwd=paths.ROOT)),
        "external_commits": ext,
        "packages": {p: _version(p) for p in ("torch", "torchaudio", "ultralytics", "whalevad",
                                              "numpy", "pandas", "scipy", "soundfile")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }


class Run:
    """One output folder per run, with run.json and a line in outputs/runs.csv."""

    def __init__(self, out_dir: Path, model: str, split: str, args: dict, overwrite: bool = False):
        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"{out_dir} already has results. Use --tag NAME for a new run "
                                  f"or --overwrite to replace it.")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        self.info = {"run_id": out_dir.name, "model": model, "split": split, "args": args,
                     "started_utc": now(), **provenance()}

    def finish(self, **summary) -> None:
        self.info.update(summary)
        self.info["finished_utc"] = now()
        (self.dir / "run.json").write_text(json.dumps(self.info, indent=2, default=str))
        register(self.info, self.dir)


REGISTRY_COLUMNS = ["run_id", "started_utc", "user", "host", "model", "split", "sites", "n_files",
                    "n_detections", "weights", "weights_sha256", "aivis_commit", "aivis_dirty",
                    "out_dir", "tag"]


def register(info: dict, out_dir: Path) -> None:
    if os.environ.get("AIVIS_NO_REGISTRY"):   # set by the smoke test
        return
    reg = paths.OUTPUTS / "runs.csv"
    reg.parent.mkdir(parents=True, exist_ok=True)
    new = not reg.exists() or reg.stat().st_size == 0
    try:
        rel = str(Path(out_dir).resolve().relative_to(paths.ROOT))
    except ValueError:
        rel = str(out_dir)
    row = {k: info.get(k, "") for k in REGISTRY_COLUMNS}
    w_path = str(info.get("weights") or "")
    try:
        w_path = str(Path(w_path).resolve().relative_to(paths.ROOT)) if w_path else ""
    except ValueError:
        pass
    row.update(out_dir=rel, weights=w_path, aivis_commit=str(info.get("aivis_commit", ""))[:7],
               sites=" ".join(info.get("sites", [])), tag=info.get("args", {}).get("tag") or "")
    with open(reg, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
