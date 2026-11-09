"""Pre-registered test G1 to G3 (logs/preregistration.md, 23.10.2026): does NAVE's loss at high recording
levels come from the FDY kernel mixing?

    python -m aivis.phase1.fdy_test [--rungs A B C D] [--seeds 0 1 ...] [--device cuda]

FDYConv2d mixes 4 basis kernels per frequency band with a softmax computed from the time-averaged input.
For 36 dev calls (12 per class, the same calls as the channel test) and every ladder checkpoint:
    original        the call at its own level
    loud            +20 dB
    loud, mixing from original   +20 dB input, but every FDY layer uses the kernel mixing it computed on
                                 the original-level input (G1: does that undo the drop?)
    original, mixing from loud   original input with the mixing computed on the +20 dB input
                                 (G2: is the mixing alone enough?)
Rung A has no FDY, so only original and loud (G3). Score: highest probability of the call's class inside
the call (mean over the 36 calls). Results: outputs/phase1/tables/fdy_test.parquet, printed summary.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from .. import paths
from . import ablate, core, probes
from .core import CLASSES, TABLES

GAIN = 10.0     # +20 dB


class Mixing:
    """Records or replaces the softmax kernel mixing of every FDYConv2d (patched forward)."""
    mode = "normal"          # normal | record | inject
    store: dict[int, torch.Tensor] = {}


def patch_fdy():
    paths.use_repo("nave")
    import torch.nn.functional as F
    import nave_model as nm   # noqa: E402

    def forward(self, x):
        B, C, _, T = x.shape
        if Mixing.mode == "inject":
            a = Mixing.store[id(self)]
        else:
            a = torch.softmax(self.att(x.mean(dim=3)) / self.temperature, dim=1)
            if Mixing.mode == "record":
                Mixing.store[id(self)] = a
        Fp = a.shape[-1]
        w = self.weight.reshape(self.n_basis * self.out_ch, self.in_ch, self.kf, self.kt)
        b = self.bias.reshape(-1) if self.bias is not None else None
        y = F.conv2d(x, w, bias=b, stride=(self.sf, self.st), padding=(self.pf, self.pt))
        y = y.view(B, self.n_basis, self.out_ch, y.shape[-2], y.shape[-1])
        return (y * a.view(B, self.n_basis, 1, Fp, 1)).sum(dim=1)

    nm.FDYConv2d.forward = forward


def calls() -> pd.DataFrame:
    lab = pd.read_parquet(TABLES / "labels_validation.parquet")
    out = []
    for c in CLASSES:
        g = lab[(lab.cls == c) & (lab.snr_db > 4) & (lab.duration < 20)]
        g = g[((g.start_s + g.end_s) / 2 > 40) & ((g.start_s + g.end_s) / 2 < 3500)].sample(12, random_state=1)
        out.append(g)
    return pd.concat(out)


def run_ckpt(rung: str, seed: int, device: str, cl: pd.DataFrame, audio: dict) -> list[dict]:
    from ..models import nave
    model, spec, _ = nave.load(ablate.ckpt(rung, seed), device)
    has_fdy = rung != "A"
    rows = []
    for r in cl.itertuples():
        x = audio[r.label_id].to(device)
        f0, f1 = int((30 - r.duration / 2) / 0.02), int((30 + r.duration / 2) / 0.02)
        ci = CLASSES.index(r.cls)

        def peak(feat):
            with torch.no_grad():
                return float(torch.sigmoid(model(feat))[0, f0:f1, ci].max())

        with torch.no_grad():
            F0, Fl = spec(x), spec(x * GAIN)
        res = {}
        Mixing.mode, Mixing.store = ("record", {}) if has_fdy else ("normal", {})
        res["original"] = peak(F0)
        mix_orig = dict(Mixing.store)
        Mixing.mode, Mixing.store = ("record", {}) if has_fdy else ("normal", {})
        res["loud"] = peak(Fl)
        mix_loud = dict(Mixing.store)
        if has_fdy:
            Mixing.mode, Mixing.store = "inject", mix_orig
            res["loud, mixing from original"] = peak(Fl)
            Mixing.mode, Mixing.store = "inject", mix_loud
            res["original, mixing from loud"] = peak(F0)
            # how much the mixing itself changes (mean absolute change of the softmax weights)
            res["mixing change"] = float(np.mean([(mix_loud[k] - mix_orig[k]).abs().mean().item() for k in mix_orig]))
        Mixing.mode = "normal"
        for k, v in res.items():
            rows.append({"rung": rung, "seed": seed, "cls": r.cls, "label_id": r.label_id, "condition": k, "value": v})
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rungs", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    p.add_argument("--seeds", type=int, nargs="+", default=ablate.SEEDS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--append", action="store_true", help="add to an existing table (for running in parts)")
    a = p.parse_args(argv)
    patch_fdy()
    cl = calls()
    audio = {}
    for r in cl.itertuples():
        ctr = (r.start_s + r.end_s) / 2
        audio[r.label_id] = torch.from_numpy(probes.read("validation", r.dataset, r.filename, ctr - 30, 60).astype(np.float32))[None]
    out = TABLES / "fdy_test.parquet"
    rows = []
    for rung in a.rungs:
        for seed in a.seeds:
            rows += run_ckpt(rung, seed, a.device, cl, audio)
            print(f"[fdy_test] {rung} seed {seed}", flush=True)
    t = pd.DataFrame(rows)
    if a.append and out.exists():
        old = pd.read_parquet(out)
        old = old[~old.set_index(["rung", "seed"]).index.isin(t.set_index(["rung", "seed"]).index.unique())]
        t = pd.concat([old, t], ignore_index=True)
    t.to_parquet(out)
    summary(t)


def summary(t: pd.DataFrame) -> None:
    m = t[t.condition != "mixing change"].pivot_table(index=["rung", "seed"], columns="condition", values="value")
    m["drop"] = m["original"] - m["loud"]
    if "loud, mixing from original" in m:
        m["G1 recovery"] = (m["loud, mixing from original"] - m["loud"]) / m["drop"]
        m["G2 share"] = (m["original"] - m["original, mixing from loud"]) / m["drop"]
    mc = t[t.condition == "mixing change"].groupby(["rung", "seed"]).value.mean().rename("mixing change")
    m = m.join(mc)
    pd.set_option("display.width", 200)
    print("[fdy_test] mean peak score over 36 calls per checkpoint:\n" + m.round(3).to_string())
    print("[fdy_test] mean per rung:\n" + m.groupby("rung").mean().round(3).to_string())


if __name__ == "__main__":
    main()
