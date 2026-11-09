"""Shared post-processing for the frame-wise models (Whale-VAD and NAVE).

We use NAVE's pipeline for both, imported from the pinned NAVE repo, so the two are treated exactly
the same: 500 ms median filter, per-class threshold, merge gaps up to 0.5 s, keep events of 0.5 to 30 s.
"""
from __future__ import annotations

import os

import numpy as np

from . import paths

_NT = None


def nave_train():
    """The nave_train module of the pinned NAVE repo (post-processing, metrics, threshold tuner)."""
    global _NT
    if _NT is None:
        os.environ.setdefault("NAVE_DATA_ROOT", str(paths.DEV))
        paths.use_repo("nave")
        import nave_train  # noqa: E402  (from external/NAVE-SED)
        _NT = nave_train
    return _NT


def settings() -> dict:
    cfg = nave_train().cfg
    return {"median_filter_ms": cfg.SMOOTH_KERNEL_MS, "merge_gap_s": cfg.MERGE_GAP_S,
            "min_duration_s": cfg.POST_MIN_DUR_S, "max_duration_s": cfg.POST_MAX_DUR_S,
            "time_reference": "frame start (frame f at f * 0.02 s)"}


def events(probs: np.ndarray, thresholds, dataset: str, filename: str) -> list[dict]:
    """Frame probabilities (n_frames, 3) of one recording -> list of event dicts (file-relative seconds)."""
    nt = nave_train()
    smooth = nt.smooth_probabilities(np.asarray(probs, dtype=np.float32))
    dets = nt.threshold_to_detections(smooth, np.asarray(thresholds, dtype=np.float64), dataset, filename)
    return [{"dataset": d.dataset, "filename": d.filename, "annotation": d.label,
             "start_s": d.start_s, "end_s": d.end_s, "confidence": d.confidence}
            for d in nt.merge_and_filter(dets)]
