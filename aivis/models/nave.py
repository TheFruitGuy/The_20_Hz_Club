"""NAVE (CNN with FDY + Conformer) with the checkpoint from the NAVE-SED repo.

Defaults follow the reported setup: 60 s tiles with 2 s overlap, NAVE's post-processing and the
per-class thresholds stored in the checkpoint. Those thresholds were tuned on all three dev sites,
so dev scores with them are in-sample (optimistic). For LOSO thresholds use python -m aivis.tune.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .. import frames, io, paths, postprocess


def load(weights: Path, device: str):
    os.environ.setdefault("NAVE_DATA_ROOT", str(paths.DEV))
    paths.use_repo("nave")
    from nave_evaluate import build_nave_from_ckpt   # noqa: E402  (from external/NAVE-SED)
    from nave_features import NAVEFeatureExtractor   # noqa: E402
    from .. import nave_heads, nave_input
    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    conf = (ck.get("config") or {}) if isinstance(ck, dict) else {}
    nave_input.apply(conf.get("input"))
    nave_heads.apply(conf.get("heads"))     # output width (3, or 5 for fine5); set on every load
    model, stored_thr = build_nave_from_ckpt(str(weights), torch.device(device))
    model.heads = conf.get("heads")
    return model, NAVEFeatureExtractor().to(device), stored_thr


def predict(recs, run: io.Run, device: str, weights: Path | None = None, segment_s: float = 60.0,
            overlap_s: float = 2.0, batch_size: int | None = None, thresholds=None, save_probs: bool = True,
            cover_tail: bool = False, stitch: str = "mean", **_):
    weights = Path(weights or paths.weights("nave"))
    batch_size = batch_size or (16 if device == "cuda" else 4)   # attention over 3000 frames needs memory
    model, spec, stored = load(weights, device)
    if thresholds:
        source = "given on the command line"
    elif stored is not None:
        thresholds, source = [float(t) for t in stored], "checkpoint (tuned on all three dev sites, in-sample)"
    else:
        thresholds, source = [0.3, 0.3, 0.3], "fixed 0.3 (checkpoint has none)"

    from .. import nave_heads
    fine = getattr(model, "heads", None) == "fine5"

    def forward(x: torch.Tensor) -> np.ndarray:
        p = torch.sigmoid(model(spec(x.to(device)))).float().cpu().numpy()   # (batch, frames, 3 or 5)
        # fine5: merge to the 3 official classes per tile (maximum within a class, as NAVE's validation does
        # before stitching) and carry the 5 fine outputs along as extra columns: [3 merged | 5 fine]
        return np.concatenate([nave_heads.collapse(p), p], axis=-1) if fine else p

    rows = []
    for rec in tqdm(recs, desc="nave", unit="file"):
        p = frames.run_recording(rec, forward, segment_s, overlap_s, batch_size, cover_tail, stitch)
        p3 = p[:, :3]
        if save_probs:
            extra = {"probs_fine": p[:, 3:]} if fine else {}      # columns nave_heads.FINE5
            io.save_frame_probs(run.dir / "frame_probs" / rec.dataset / f"{Path(rec.filename).stem}.npz",
                                p3, paths.CLASSES, **extra)
        rows += postprocess.events(p3, thresholds, rec.dataset, rec.filename)
    import nave_config as cfg   # noqa: E402
    return rows, {"weights": str(weights), "weights_sha256": io.sha256(weights, 12),
                  "thresholds": dict(zip(paths.CLASSES, thresholds)), "threshold_source": source,
                  "ablation": cfg.ablation_tag(), "heads": getattr(model, "heads", None) or "coarse3",
                  "segment_s": segment_s, "overlap_s": overlap_s,
                  "cover_tail": cover_tail, "stitch": stitch, "postprocess": postprocess.settings()}
