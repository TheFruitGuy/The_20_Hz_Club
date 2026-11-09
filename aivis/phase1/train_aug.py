"""Retrain NAVE (full recipe, kernel 153) with frequency-shift augmentation, for test A1 to A4.

    python -m aivis.phase1.train_aug --seed 0 [--max-shift-hz 1.5] [--p 0.8] [--workers 8]

Everything is NAVE's own training (external/NAVE-SED/nave_train.py, unchanged, same data, epochs,
validation and model selection). The only change: before the feature extractor, each training segment
is, with probability p, shifted in frequency by a value drawn uniformly from -max..+max Hz (single
sideband, so the whole spectrum moves by the same number of hertz, like the call drift). Validation
and threshold tuning see the original audio.

Runs go to outputs/phase1/aug_runs/nave_s<seed>_pcen1_fdy1_k153_fshift<max>_<time>/ (nave_best.pt etc.).
The checkpoint's config records fshift_hz and fshift_p.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from .. import paths
from . import core

AUG_ROOT = core.ROOT / "aug_runs"


def ssb_shift(x: torch.Tensor, hz: torch.Tensor, sr: int = 250) -> torch.Tensor:
    """Shift every row of x (batch, samples) by hz[row] Hz (analytic signal via the FFT)."""
    n = x.shape[-1]
    X = torch.fft.fft(x.float(), dim=-1)
    h = torch.zeros(n, device=x.device)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    analytic = torch.fft.ifft(X * h, dim=-1)
    t = torch.arange(n, device=x.device, dtype=torch.float32) / sr
    phase = torch.exp(2j * torch.pi * hz[:, None].float() * t[None, :])
    return torch.real(analytic * phase).to(x.dtype)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-shift-hz", type=float, default=1.5)
    p.add_argument("--p", type=float, default=0.8, help="share of training segments that get shifted")
    p.add_argument("--workers", type=int, default=8, help="data loader workers (NAVE's default is 16)")
    p.add_argument("--epochs", type=int, help="override NAVE's 40 epochs (only for a quick test run)")
    p.add_argument("--out", help="other output folder (for a quick test run)")
    a = p.parse_args(argv)

    os.environ.setdefault("NAVE_DATA_ROOT", str(paths.DEV))
    paths.use_repo("nave")
    import nave_config as cfg     # noqa: E402
    import nave_train as nt       # noqa: E402

    cfg.OUTPUT_DIR = Path(a.out) if a.out else AUG_ROOT
    cfg.NUM_WORKERS = a.workers
    if a.epochs:
        cfg.EPOCHS = a.epochs
    tag = f"_fshift{a.max_shift_hz:g}"
    orig_tag, orig_conf = cfg.ablation_tag, cfg.ablation_config
    cfg.ablation_tag = lambda: orig_tag() + tag
    cfg.ablation_config = lambda: {**orig_conf(), "fshift_hz": a.max_shift_hz, "fshift_p": a.p}

    gen = torch.Generator().manual_seed(10_000 + a.seed)

    class ShiftThenSpec(torch.nn.Module):
        """Wraps NAVE's feature extractor during training only (model.training decides)."""
        def __init__(self, spec, model_ref):
            super().__init__()
            self.spec, self.model_ref = spec, model_ref

        @property
        def hop_length(self):
            return self.spec.hop_length

        def forward(self, audio):
            if self.model_ref[0] is not None and self.model_ref[0].training:
                b = audio.shape[0]
                on = torch.rand(b, generator=gen) < a.p
                hz = (torch.rand(b, generator=gen) * 2 - 1) * a.max_shift_hz * on
                if on.any():
                    audio = ssb_shift(audio, hz.to(audio.device))
            return self.spec(audio)

    model_ref = [None]
    orig_train_epoch = nt._train_epoch

    def train_epoch(model, spec_extractor, *args, **kw):
        model_ref[0] = model
        return orig_train_epoch(model, ShiftThenSpec(spec_extractor, model_ref), *args, **kw)

    nt._train_epoch = train_epoch
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[train_aug] seed {a.seed}, shift +-{a.max_shift_hz} Hz on {a.p:.0%} of training segments, "
          f"runs in {cfg.OUTPUT_DIR}", flush=True)
    sys.argv = ["nave_train", "--seed", str(a.seed)]
    nt.main()


if __name__ == "__main__":
    main()
