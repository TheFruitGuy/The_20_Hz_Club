"""Finding recordings and reading their start time from the file name."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from . import paths

SAMPLE_RATE = 250


@dataclass(frozen=True)
class Recording:
    dataset: str          # site-year, e.g. casey2017
    filename: str         # e.g. 2017-02-04T05-00-00_000.wav
    path: Path
    duration_s: float
    sample_rate: int

    @property
    def start(self) -> datetime:
        return file_start(self.filename)


def file_start(filename: str) -> datetime:
    """UTC start of a recording, from names like 2014-02-18T21-00-00_000.wav (same rule as the baseline)."""
    stem = Path(filename).stem
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S_%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.strptime(stem.split("_")[0], "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)


def list_recordings(split: str | None = None, sites: list[str] | None = None,
                    audio_dir: Path | None = None, dataset: str | None = None,
                    limit_files: int | None = None, files: list[str] | None = None) -> list[Recording]:
    """Recordings of a split (all its sites, or the given ones), or of one folder of wav files.

    split="validation"                      -> the three dev sites
    split="validation", sites=["casey2017"] -> one site
    audio_dir=..., dataset="mysite"         -> any folder of 250 Hz wav files
    limit_files                             -> first N files per site, for quick tests
    files                                   -> only these file names (e.g. 2017-01-03T05-00-00_000.wav)
    """
    folders: list[tuple[str, Path]] = []
    if audio_dir is not None:
        audio_dir = Path(audio_dir)
        folders.append((dataset or audio_dir.name, audio_dir))
    else:
        if split is None:
            raise ValueError("give a split or an audio_dir")
        for site in sites or paths.SITES[split]:
            folders.append((site, paths.audio_dir(split) / site))
    recs: list[Recording] = []
    for ds, folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} not found (is the data downloaded? python scripts/aivis_setup.py data)")
        wavs = sorted(folder.glob("*.wav"))
        if files:
            wavs = [w for w in wavs if w.name in set(files)]
        if limit_files:
            wavs = wavs[:limit_files]
        for w in wavs:
            info = sf.info(str(w))
            recs.append(Recording(ds, w.name, w, info.frames / info.samplerate, info.samplerate))
    return recs
