"""Where everything lives. All paths come from here, so nobody hardcodes their own."""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "config" / "external.toml").read_text(encoding="utf-8"))
_LOCAL = ROOT / "config" / "local.toml"   # written by `aivis_setup.py all --data-dir`, not in git
LOCAL = tomllib.loads(_LOCAL.read_text(encoding="utf-8")) if _LOCAL.exists() else {}

# data folder: AIVIS_DATA, else data_dir from config/local.toml, else data/ in the repo
DATA = Path(os.environ.get("AIVIS_DATA") or LOCAL.get("data_dir") or ROOT / "data")
DEV = DATA / "dev"            # train/ and validation/, each with audio/<site>/*.wav and annotations/<site>.csv
EVAL = DATA / "eval"          # audio/<site>/*.wav, no labels
WEIGHTS = ROOT / "weights"
OUTPUTS = Path(os.environ.get("AIVIS_OUTPUTS", ROOT / "outputs"))
LOGS = ROOT / "logs"

SITES: dict[str, list[str]] = CONFIG["sites"]
SPLITS = ("train", "validation", "evaluation")
CLASSES = ["bmabz", "d", "bp"]
FINE_TO_CLASS = {"bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
                 "bmd": "d", "bpd": "d", "bp20": "bp", "bp20plus": "bp"}


def repo(key: str) -> Path:
    """Path of an external repo (task2, whalevad, nave, deepvoice)."""
    return ROOT / CONFIG["repos"][key]["path"]


def weights(key: str) -> Path:
    """Default weights file of a model (yolo_baseline, whalevad, nave)."""
    return WEIGHTS / CONFIG["weights"][key]["file"]


def split_dir(split: str) -> Path:
    if split in ("train", "validation"):
        return DEV / split
    if split == "evaluation":
        return EVAL
    raise ValueError(f"unknown split {split!r}, use one of {SPLITS}")


def audio_dir(split: str) -> Path:
    return split_dir(split) / "audio"


def annotations_dir(split: str) -> Path:
    """Folder with <site>.csv label files. Does not exist for the evaluation split."""
    return split_dir(split) / "annotations"


def split_of_site(site: str) -> str:
    for split, sites in SITES.items():
        if site in sites:
            return split
    raise ValueError(f"unknown site {site!r}")


def use_repo(key: str, sub: str = "") -> Path:
    """Put an external repo (or a folder in it) on sys.path and return its path."""
    p = repo(key) / sub
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing, run `python scripts/aivis_setup.py all` first")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
