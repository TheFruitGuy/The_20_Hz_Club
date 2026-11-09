"""Input variants of NAVE's front end (tests M1 to M4), applied as a reversible patch of NAVE-SED's code.

    variant None      NAVE as published: |demeaned STFT|, cos and sin of its phase, fixed PCEN (4 channels)
    variant "nomag"   without the linear magnitude channel: cos, sin, PCEN (3 channels)
    variant "logmag"  channel 0 = log(|STFT| + 1e-6) minus its mean over the tile, per frequency bin
                      (4 channels; a gain then only shifts the log, and the per-bin mean removes the shift)

apply(variant) must be called before a model is built or trained; apply(None) restores the original.
A checkpoint trained with a variant records it as config["input"]; aivis.models.nave.load reads it.
"""
from __future__ import annotations

import torch

from . import paths

VARIANTS = (None, "nomag", "logmag")
_ORIG: dict = {}


def _modules():
    paths.use_repo("nave")
    import nave_config as cfg       # noqa: E402
    import nave_features as nf      # noqa: E402
    return cfg, nf


def apply(variant: str | None) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown input variant {variant!r}")
    cfg, nf = _modules()
    if not _ORIG:
        _ORIG["apply_ablation"] = cfg.apply_ablation
        _ORIG["forward"] = nf.NAVEFeatureExtractor.forward
    orig_apply, orig_forward = _ORIG["apply_ablation"], _ORIG["forward"]
    if variant is None:
        cfg.apply_ablation = orig_apply
        nf.NAVEFeatureExtractor.forward = orig_forward
        cfg.apply_ablation(use_pcen=cfg.USE_PCEN, use_fdy=cfg.USE_FDY, conv_kernel=cfg.CONV_KERNEL)
        return

    def apply_ablation(*args, **kw):
        out = orig_apply(*args, **kw)
        if variant == "nomag":
            cfg.FEAT_CHANNELS = 3 if cfg.USE_PCEN else 2
        return out

    def forward(self, audio):
        x = orig_forward(self, audio)                      # (B, 4, F, T): |demeaned|, cos, sin, PCEN
        if variant == "nomag":
            return x[:, 1:]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, center=False, return_complex=True)
        lm = torch.log(stft.abs() + 1e-6)
        lm = lm - lm.mean(dim=-1, keepdim=True)
        return torch.cat([lm.unsqueeze(1), x[:, 1:]], dim=1)

    cfg.apply_ablation = apply_ablation
    nf.NAVEFeatureExtractor.forward = forward
    cfg.apply_ablation(use_pcen=cfg.USE_PCEN, use_fdy=cfg.USE_FDY, conv_kernel=cfg.CONV_KERNEL)
