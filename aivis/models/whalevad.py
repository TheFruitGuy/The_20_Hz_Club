"""Whale-VAD (CNN + BiLSTM) with the released v0.1.0 weights.

The released head has 7 outputs although the model was trained on the 3 joined classes. The weight
vectors fall into three tight groups (outputs 0-2, 3-4, 5-6), and synthetic calls give the order
bmabz, d, bp, so we take the maximum per group. Check this on real dev calls before relying on it.
There are no published thresholds or post-processing, so we use NAVE's post-processing and fixed
thresholds unless you pass tuned ones (python -m aivis.tune).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .. import frames, io, paths, postprocess

GROUPS = {"bmabz": [0, 1, 2], "d": [3, 4], "bp": [5, 6]}
DEFAULT_THRESHOLDS = (0.5, 0.5, 0.5)


def load(weights: Path, device: str):
    from whalevad import WhaleVAD_Weights, WhaleVADClassifier
    from whalevad.model import WhaleVADModel
    spec = WhaleVAD_Weights.DEFAULT
    clf = WhaleVADClassifier(**spec.model_config)
    clf.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return WhaleVADModel(clf, spec.transform).eval().to(device)


def to_classes(p7: np.ndarray) -> np.ndarray:
    return np.stack([p7[:, g].max(axis=1) for g in GROUPS.values()], axis=1)


def predict(recs, run: io.Run, device: str, weights: Path | None = None, segment_s: float = 30.0,
            overlap_s: float = 2.0, batch_size: int = 32, thresholds=None, save_probs: bool = True,
            cover_tail: bool = False, stitch: str = "mean", **_):
    weights = Path(weights or paths.weights("whalevad"))
    source = "given on the command line" if thresholds else "fixed default 0.5, untuned"
    thresholds = list(thresholds or DEFAULT_THRESHOLDS)
    model = load(weights, device)

    def forward(x: torch.Tensor) -> np.ndarray:
        _, prob, _ = model(x.to(device).unsqueeze(1))      # (batch, channel=1, samples)
        return prob.float().cpu().numpy()                  # (batch, frames, 7)

    rows = []
    for rec in tqdm(recs, desc="whalevad", unit="file"):
        p7 = frames.run_recording(rec, forward, segment_s, overlap_s, batch_size, cover_tail, stitch)
        p3 = to_classes(p7)
        if save_probs:
            io.save_frame_probs(run.dir / "frame_probs" / rec.dataset / f"{Path(rec.filename).stem}.npz",
                                p3, paths.CLASSES, probs7=p7)
        rows += postprocess.events(p3, thresholds, rec.dataset, rec.filename)
    return rows, {"weights": str(weights), "weights_sha256": io.sha256(weights, 12),
                  "thresholds": dict(zip(paths.CLASSES, thresholds)),
                  "threshold_source": source,
                  "output_groups": GROUPS, "segment_s": segment_s, "overlap_s": overlap_s,
                  "cover_tail": cover_tail, "stitch": stitch, "postprocess": postprocess.settings()}
