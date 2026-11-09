"""Pre-registered test H1 and H2 (ai-interactions/preregistration.md, 23.10.2026): where inside the full NAVE does the
loss at +20 dB arise?

    python -m aivis.phase1.layer_test                 # all 5 NAVE seeds (rung D), about 10 min on a laptop CPU
    python -m aivis.phase1.layer_test --seeds 666     # one seed
    python -m aivis.phase1.layer_test --test          # code check on synthetic audio, no data needed

For the same 36 dev calls as the channel and FDY tests, NAVE runs on the original and on the +20 dB audio.
At each of 15 sites (stem layers, the projection, the four Conformer blocks):
    rms ratio            per-channel RMS of the +20 dB activations over the original ones (median over channels)
    shape similarity     cosine similarity of the two activations after each channel is scaled to RMS 1
                         (1 = the louder input only changed the scale at this point)
    recovery             +20 dB run with the activations at this site rescaled, channel by channel, to the RMS of
                         the original run (only the scale is taken from the original run, not the content);
                         share of the drop in mean peak score that this undoes
Results: outputs/phase1/tables/layer_test.parquet and a printed summary per site.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from . import ablate, fdy_test
from .core import CLASSES, TABLES

SITES = ["stem.filterbank", "stem.feat_extractor.0", "stem.feat_extractor.1", "stem.feat_extractor.2",
         "stem.feat_extractor.3", "stem.feat_extractor.4", "stem.feat_extractor.5", "stem.feat_extractor.6",
         "stem.feat_extractor.7", "stem.residual_stack", "proj", "blocks.0", "blocks.1", "blocks.2", "blocks.3"]
LABEL = {"stem.filterbank": "FDY filterbank", "stem.feat_extractor.0": "FDY conv", "stem.feat_extractor.1": "BatchNorm",
         "stem.feat_extractor.2": "GELU", "stem.feat_extractor.3": "MaxPool", "stem.feat_extractor.4": "conv",
         "stem.feat_extractor.5": "BatchNorm", "stem.feat_extractor.6": "GELU", "stem.feat_extractor.7": "MaxPool",
         "stem.residual_stack": "residual stack", "proj": "projection", "blocks.0": "Conformer 1",
         "blocks.1": "Conformer 2", "blocks.2": "Conformer 3", "blocks.3": "Conformer 4"}
GAIN = 10.0


def chan_dim(x: torch.Tensor) -> int:
    return 1 if x.dim() == 4 else x.dim() - 1        # (B, C, F, T) or (B, T, D)


def chan_rms(x: torch.Tensor) -> torch.Tensor:
    d = chan_dim(x)
    dims = [i for i in range(x.dim()) if i != d]
    return x.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)


def run_seed(seed: int, device: str, items: list[dict]) -> list[dict]:
    from ..models import nave
    model, spec, _ = nave.load(ablate.ckpt("D", seed), device)
    mods = dict(model.named_modules())
    rows = []
    for it in items:
        x, f0, f1, ci = it["x"].to(device), it["f0"], it["f1"], it["ci"]
        with torch.no_grad():
            F0, Fl = spec(x), spec(x * GAIN)
        acts: dict[str, dict[str, torch.Tensor]] = {"orig": {}, "loud": {}}
        hooks = []

        def recorder(name, key):
            def h(_m, _i, out):
                acts[key][name] = (out[0] if isinstance(out, tuple) else out).detach()
            return h

        def peak(feat):
            with torch.no_grad():
                return float(torch.sigmoid(model(feat))[0, f0:f1, ci].max())

        for key, feat in (("orig", F0), ("loud", Fl)):
            hooks = [mods[s].register_forward_hook(recorder(s, key)) for s in SITES]
            p = peak(feat)
            for h in hooks:
                h.remove()
            if key == "orig":
                p_orig = p
            else:
                p_loud = p
        base = {"seed": seed, "label_id": it["label_id"], "cls": it["cls"], "peak_orig": p_orig, "peak_loud": p_loud}
        for s in SITES:
            a, b = acts["orig"][s], acts["loud"][s]
            ra, rb = chan_rms(a), chan_rms(b)
            ratio = float((rb / ra).flatten().median())
            sim = float(torch.nn.functional.cosine_similarity((a / ra).flatten(), (b / rb).flatten(), dim=0))

            def rescale(_m, _i, out, ra=ra):
                o = out[0] if isinstance(out, tuple) else out
                return o / chan_rms(o) * ra
            h = mods[s].register_forward_hook(rescale)
            p_fix = peak(Fl)
            h.remove()
            rows.append({**base, "site": s, "rms_ratio": ratio, "shape_similarity": sim, "peak_rescaled": p_fix})
    return rows


def real_items() -> list[dict]:
    from . import probes
    items = []
    for r in fdy_test.calls().itertuples():
        ctr = (r.start_s + r.end_s) / 2
        x = torch.from_numpy(probes.read("validation", r.dataset, r.filename, ctr - 30, 60).astype(np.float32))[None]
        items.append({"x": x, "f0": int((30 - r.duration / 2) / 0.02), "f1": int((30 + r.duration / 2) / 0.02),
                      "ci": CLASSES.index(r.cls), "cls": r.cls, "label_id": int(r.label_id)})
    return items


def test_items(n=3) -> list[dict]:
    g = torch.Generator().manual_seed(0)
    t = torch.arange(250 * 60) / 250
    items = []
    for i in range(n):
        x = 0.01 * torch.randn(250 * 60, generator=g)
        x[25 * 250:35 * 250] += 0.02 * torch.sin(2 * torch.pi * 26 * t[25 * 250:35 * 250])
        items.append({"x": x[None], "f0": 1250, "f1": 1750, "ci": 0, "cls": "bmabz", "label_id": -i - 1})
    return items


def summary(t: pd.DataFrame) -> None:
    per_seed = t.groupby(["seed", "site"]).agg(peak_orig=("peak_orig", "mean"), peak_loud=("peak_loud", "mean"),
                                               peak_rescaled=("peak_rescaled", "mean"), rms_ratio=("rms_ratio", "median"),
                                               shape_similarity=("shape_similarity", "mean")).reset_index()
    per_seed["recovery"] = (per_seed.peak_rescaled - per_seed.peak_loud) / (per_seed.peak_orig - per_seed.peak_loud)
    per_seed.to_parquet(TABLES / "layer_test_summary.parquet")
    pd.set_option("display.width", 200)
    m = per_seed.groupby("site")[["rms_ratio", "shape_similarity", "recovery"]].mean().reindex(SITES)
    m.index = [f"{i + 1:2d} {LABEL[s]}" for i, s in enumerate(SITES)]
    print("[layer_test] per site, mean over seeds:\n" + m.round(3).to_string())
    print("[layer_test] recovery per seed and site:\n"
          + per_seed.pivot_table(index="site", columns="seed", values="recovery").reindex(SITES).round(2).to_string())
    print("[layer_test] mean peak score: original", per_seed.peak_orig.mean().round(3), "+20 dB", per_seed.peak_loud.mean().round(3))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=ablate.SEEDS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test", action="store_true", help="synthetic audio, only to check that the code runs")
    a = p.parse_args(argv)
    items = test_items() if a.test else real_items()
    rows = []
    for seed in a.seeds:
        rows += run_seed(seed, a.device, items)
        print(f"[layer_test] seed {seed} done", flush=True)
    t = pd.DataFrame(rows)
    if a.test:
        print(t.groupby("site")[["rms_ratio", "shape_similarity", "peak_rescaled"]].mean().reindex(SITES).round(3))
        return
    t.to_parquet(TABLES / "layer_test.parquet")
    summary(t)


if __name__ == "__main__":
    main()
