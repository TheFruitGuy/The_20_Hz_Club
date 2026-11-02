"""Tiling and stitching for the frame-wise models, the same way NAVE's evaluation does it:
fixed-length tiles with a small overlap, probabilities averaged where tiles overlap."""
from __future__ import annotations

from typing import Callable

import numpy as np
import soundfile as sf
import torch

from .recordings import Recording

HOP = 5  # samples per frame (20 ms at 250 Hz)


def tiles(n_samples: int, sr: int, segment_s: float, overlap_s: float,
          cover_tail: bool = False) -> list[tuple[int, int]]:
    """Start/end samples of the tiles. Same rule as NAVE: tiles of segment_s every segment_s - overlap_s
    seconds while they fit, so the last few seconds of a file can stay uncovered (cover_tail=True adds
    one tile that ends at the file end). A file shorter than one tile becomes a single shorter tile."""
    seg = int(round(segment_s * sr))
    step = int(round((segment_s - overlap_s) * sr))
    if seg <= 0 or step <= 0:   # step 0 would never end (and fill the memory)
        raise ValueError(f"overlap ({overlap_s} s) must be shorter than the tile ({segment_s} s)")
    out, t = [], 0
    while t + seg <= n_samples:
        out.append((t, t + seg))
        t += step
    if not out:
        return [(0, n_samples)]
    if cover_tail and out[-1][1] < n_samples:
        out.append((n_samples - seg, n_samples))
    return out


@torch.no_grad()
def run_recording(rec: Recording, forward: Callable[[torch.Tensor], np.ndarray], segment_s: float,
                  overlap_s: float, batch_size: int, cover_tail: bool = False, stitch: str = "mean") -> np.ndarray:
    """Frame probabilities (n_frames, n_out) for one recording. forward maps a (B, n_samples) float tensor
    to (B, frames, n_out) probabilities.

    stitch="mean"   average where tiles overlap (NAVE's stitch_segments, the default)
    stitch="centre" each frame from the tile in which it lies furthest from an edge (no averaging)"""
    audio, sr = sf.read(str(rec.path), dtype="float32", always_2d=True)
    audio = audio[:, 0]
    spans = tiles(len(audio), sr, segment_s, overlap_s, cover_tail)
    outs: list[tuple[int, np.ndarray]] = []
    by_len: dict[int, list[tuple[int, int]]] = {}
    for s, e in spans:
        by_len.setdefault(e - s, []).append((s, e))
    for _, group in by_len.items():
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            x = torch.from_numpy(np.stack([audio[s:e] for s, e in chunk]))
            p = forward(x)
            for (s, e), pj in zip(chunk, p):
                n = min((e - s) // HOP, pj.shape[0])
                outs.append((s, pj[:n]))
    if stitch == "centre":
        max_end = max(s + p.shape[0] * HOP for s, p in outs)
        total = max_end // HOP + 1
        out = np.zeros((total, outs[0][1].shape[1]), dtype=np.float32)
        best = np.full(total, -1, dtype=np.int64)
        for s, p in outs:
            f0, n = s // HOP, len(p)
            k = np.arange(n)
            dist = np.minimum(k, n - 1 - k)              # frames to the nearer tile edge
            sl = slice(f0, f0 + n)
            take = dist > best[sl]
            out[sl][take] = p[take]
            best[sl][take] = dist[take]
        return out
    if stitch != "mean":
        raise ValueError(f"unknown stitch {stitch!r}")
    # stitch like NAVE's stitch_segments: average where tiles overlap
    max_end = max(s + p.shape[0] * HOP for s, p in outs)
    total = max_end // HOP + 1
    n_out = outs[0][1].shape[1]
    acc = np.zeros((total, n_out), dtype=np.float64)
    cnt = np.zeros(total, dtype=np.float64)
    for s, p in outs:
        f0 = s // HOP
        acc[f0:f0 + len(p)] += p
        cnt[f0:f0 + len(p)] += 1
    return (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
