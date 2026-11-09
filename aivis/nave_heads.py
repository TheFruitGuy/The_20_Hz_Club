"""Output heads of NAVE (test F1 to F4): the published 3 coarse classes, or 5 outputs that keep apart the call
types NAVE merges, applied as a reversible patch of NAVE-SED's code (like aivis.nave_input).

    heads None      NAVE as published: bmabz, d, bp
    heads "fine5"   bmabz, bmd, bpd, bp20, bp20plus. D is split into blue whale D-calls (bmd) and fin whale
                    40 Hz downsweeps (bpd), BP into bp20 and bp20plus. The Z-call units (bma, bmb, bmz) stay one
                    output, they are parts of one song.

For scoring, the 5 outputs are merged back to the 3 official classes by the maximum within each class, the
same rule as NAVE's own (unused) 7-class path (nave_train.collapse_probs_to_3class). Validation, threshold
tuning and model selection during training therefore run on 3 classes, exactly as for the published model.
A checkpoint trained with fine5 records config["heads"] = "fine5"; aivis.models.nave.load reads it.
"""
from __future__ import annotations

import numpy as np

from . import paths

FINE5 = ["bmabz", "bmd", "bpd", "bp20", "bp20plus"]
TO_FINE5 = {"bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz", "bmd": "bmd", "bpd": "bpd", "bp20": "bp20",
            "bp20plus": "bp20plus"}
GROUPS = {"bmabz": [0], "d": [1, 2], "bp": [3, 4]}          # columns of FINE5 per official class
HEADS = (None, "fine5")


def _cfg():
    paths.use_repo("nave")
    import nave_config as cfg       # noqa: E402
    return cfg


def n_out(heads: str | None) -> int:
    if heads not in HEADS:
        raise ValueError(f"unknown heads {heads!r}")
    return 5 if heads == "fine5" else 3


def apply(heads: str | None) -> None:
    """Set the width of NAVE's output layer before a model is built (training or loading)."""
    _cfg().N_CLASSES = n_out(heads)


def collapse(p: np.ndarray) -> np.ndarray:
    """(n_frames, 5) fine5 probabilities -> (n_frames, 3) in paths.CLASSES order (bmabz, d, bp)."""
    p = np.asarray(p)
    if p.shape[-1] != 5:
        raise ValueError(f"expected 5 outputs, got {p.shape[-1]}")
    return np.stack([p[..., GROUPS[c]].max(axis=-1) for c in paths.CLASSES], axis=-1)


def patch_training() -> None:
    """Make nave_train train on fine5 targets while it validates, tunes and selects on 3 classes.

    - annotations: the fine label column maps bma/bmb/bmz to bmabz (label_3class is kept as it was)
    - targets and class weights: the 5 fine5 names whenever NAVE is not in its 3-class mode
    - validation: the 5 outputs are merged to 3 with collapse() before post-processing
    """
    cfg = _cfg()
    import nave_train as nt         # noqa: E402

    apply("fine5")
    if getattr(nt, "_aivis_fine5", False):     # already patched (idempotent)
        return
    nt._aivis_fine5 = True
    cfg.USE_3CLASS = False          # NAVE flips this to True itself while it post-processes 3-class output
    three = list(cfg.CALL_TYPES_3)
    cfg.class_names = lambda: list(three) if cfg.USE_3CLASS else list(FINE5)

    orig_load = nt.load_annotations

    def load_annotations(datasets, manifest=None):
        df = orig_load(datasets, manifest=manifest).copy()
        unknown = set(df["annotation"]) - set(TO_FINE5)
        if unknown:
            raise ValueError(f"annotation labels without a fine5 class: {sorted(unknown)}")
        df["annotation"] = df["annotation"].map(TO_FINE5)
        return df

    def collapse_probs(all_probs: dict) -> dict:
        if not all_probs:
            return all_probs
        return {k: collapse(v).astype(v.dtype) for k, v in all_probs.items()}

    nt.load_annotations = load_annotations
    nt.collapse_probs_to_3class = collapse_probs
