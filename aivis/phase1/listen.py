"""Listening check (pre-registered LC1 to LC4): are the false alarms real calls nobody labelled, and are the
labels every model misses real calls?

    python -m aivis.phase1.listen pack [--workers 3]          # server, CPU only, needs the 5 NAVE seed runs
    python -m aivis.phase1.listen refs                        # server, adds reference examples to the pack
    python -m aivis.phase1.listen redraw                      # server, redraws images (fixed layout) and references
    python -m aivis.phase1.listen report [--pack DIR]         # after labelling, wherever key/ and the csvs are

Notebook 04_listening_check.ipynb has the labelling form (labeller()) and calls report().

Items (seed 2026, same for every annotator, shuffled):
fa      false alarms of the released NAVE seed (666) that all four other seeds also detect (agreement 4),
        at the LOSO thresholds, stratified over site x class
miss    labels missed by all 7 models (5 NAVE seeds, YOLO, Whale-VAD), stratified over site x class
hit     control: labels every model detects (should be judged a call)
bg      control: windows from the same recordings with no label and no detection of any model within 10 s,
        durations drawn from the false alarms of the same class (should be judged no call)

Blind: the annotator sees only the site, the class to look for, a spectrogram with the window marked in time
(no frequency box, no model output) and the audio sped up 10x. Group, file and times are in key/, which the
annotators do not open. reveal/ has the same window with the labels and every model's events, for after.

Pack layout:  pack/blind/{items.csv, img/, audio/}   pack/blind.zip   pack/key/   pack/reveal/
The blind folder is also in git at results/phase1/listening/blind (the form uses it by default); key/ and
reveal/ are never committed, they would unblind the check.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from . import build, core
from .core import CLASSES, TABLES

PACK = core.ROOT / "listening" / "pack"
RESULTS = paths.ROOT / "results" / "phase1" / "listening"
SEED = 2026
SR = 250
SPEED = 10                   # audio played 10x faster, 15 to 30 Hz becomes 150 to 300 Hz
CONTEXT_S = 60.0
CLEAR_S = 10.0               # background windows: nothing labelled or detected this close
VAL = paths.SITES["validation"]
ANSWERS = ["yes", "no", "unsure"]
OTHER = ["none", "bmabz", "d", "bp", "other sound"]
CLASS_LONG = {"bmabz": "Antarctic blue whale A, B or Z call (BMABZ)",
              "d": "D-call, a frequency downsweep (D)",
              "bp": "fin whale 20 Hz pulse (BP)"}
CLASS_COLOR = {"bmabz": "#1baf7a", "d": "#6250d6", "bp": "#c98500"}
MARK = "#d6336c"


def say(m):
    print(f"[listen] {m}", flush=True)


# ------------------------------------------------------------------------------------- sampling

def alloc(n: int, sizes: dict) -> dict:
    """Spread n items as evenly as possible over the cells, never more than a cell holds."""
    out = {k: 0 for k in sizes}
    active = [k for k in sizes if sizes[k] > 0]
    while n > 0 and active:
        share = max(1, n // len(active))
        for k in list(active):
            take = min(share, sizes[k] - out[k], n)
            out[k] += take
            n -= take
            if out[k] == sizes[k]:
                active.remove(k)
            if n == 0:
                break
    return out


def stratified(df: pd.DataFrame, n: int, by: list[str], rng) -> pd.DataFrame:
    groups = {k: g for k, g in df.groupby(by)}
    take = alloc(n, {k: len(g) for k, g in groups.items()})
    parts = [g.iloc[rng.choice(len(g), take[k], replace=False)] for k, g in groups.items() if take[k]]
    return pd.concat(parts) if parts else df.iloc[:0]


def time_iou(a0, a1, b0, b1):
    inter = np.maximum(0, np.minimum(a1, b1) - np.maximum(a0, b0))
    return inter / np.maximum(np.maximum(a1, b1) - np.minimum(a0, b0), 1e-9)


def load_models() -> dict[str, pd.DataFrame]:
    out = {}
    for m in ("yolo", "whalevad"):
        ev = pd.read_parquet(build.events_path(m, "loso"))
        out[m] = ev[ev.dataset.isin(VAL)]
    return out


def background(n: int, fa_dur: dict, gt: pd.DataFrame, busy: pd.DataFrame, rng) -> pd.DataFrame:
    """Windows with nothing labelled or detected within CLEAR_S, classes in turn, sites in turn."""
    import soundfile as sf
    files = {s: sorted((paths.audio_dir("validation") / s).glob("*.wav")) for s in VAL}
    rows, tries = [], 0
    while len(rows) < n and tries < 20000:
        tries += 1
        site, c = VAL[len(rows) % len(VAL)], CLASSES[len(rows) % len(CLASSES)]
        f = files[site][rng.integers(len(files[site]))]
        dur = float(rng.choice(fa_dur[c])) if len(fa_dur[c]) else 5.0
        length = sf.info(str(f)).duration
        if length < CONTEXT_S + dur:
            continue
        a = float(rng.uniform(CONTEXT_S / 2, length - CONTEXT_S / 2 - dur))
        b = a + dur
        g = gt[(gt.dataset == site) & (gt.filename == f.name)]
        e = busy[(busy.dataset == site) & (busy.filename == f.name)]
        if ((g.start_s < b + CLEAR_S) & (g.end_s > a - CLEAR_S)).any():
            continue
        if ((e.start_s < b + CLEAR_S) & (e.end_s > a - CLEAR_S)).any():
            continue
        rows.append({"group": "bg", "dataset": site, "filename": f.name, "cls": c, "start_s": a, "end_s": b})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- drawing

def read_span(dataset: str, filename: str, t0: float, span: float) -> np.ndarray:
    """Audio from t0 for span seconds, zero-padded outside the file."""
    import soundfile as sf
    n = int(round(span * SR))
    out = np.zeros(n, np.float32)
    with sf.SoundFile(str(paths.audio_dir("validation") / dataset / filename)) as f:
        a = int(round(t0 * SR))
        lo, hi = max(a, 0), min(a + n, f.frames)
        if hi > lo:
            f.seek(lo)
            out[lo - a:hi - a] = f.read(hi - lo, dtype="float32", always_2d=True)[:, 0]
    return out


def spectrogram(x: np.ndarray, nperseg: int = 256):
    """dB spectrogram with the median of every frequency bin removed (makes faint tonal calls stand out)."""
    from scipy.signal import spectrogram as sp
    f, t, S = sp(x, fs=SR, window="hann", nperseg=nperseg, noverlap=nperseg - nperseg // 25, mode="psd")
    S = 10 * np.log10(S + 1e-12)
    return f, t, S - np.median(S, axis=1, keepdims=True)


def show(ax, f, t, S, t_off, lim=None):
    lim = lim or (np.percentile(S, 2), np.percentile(S, 99.7))
    ax.pcolormesh(t + t_off, f, S, cmap="Greys", vmin=lim[0], vmax=lim[1], shading="auto", rasterized=True)
    ax.set_ylim(0, 125)
    ax.set_ylabel("Hz")
    return lim


def mark(ax, a, b):
    for x in (a, b):
        ax.axvline(x, color=MARK, ls="--", lw=1.3)


def spans(r) -> tuple[float, float, float, float]:
    dur = r.end_s - r.start_s
    span = max(CONTEXT_S, dur + 30)
    c0 = (r.start_s + r.end_s) / 2 - span / 2
    pad = max(4.0, dur / 2)
    return c0, span, r.start_s - pad, dur + 2 * pad


# fixed axes positions [left, bottom, width, height] in figure fractions, so the form can put a play line on
# exactly the right spot; each panel spans the same time as its audio file
BLIND_FIG = (11, 6.2)
BLIND_AX = {"context": [0.07, 0.58, 0.91, 0.36], "zoom": [0.07, 0.09, 0.91, 0.38]}
REF_FIG = (5.2, 2.5)
REF_AX = [0.12, 0.2, 0.85, 0.66]


def panel(ax_pos, audio_b64: str, label: str) -> dict:
    l, b, w, h = ax_pos
    return {"left": l, "right": l + w, "top": 1 - (b + h), "bottom": 1 - b, "audio": audio_b64, "label": label}


def blind_figure(r, path: Path):
    import matplotlib.pyplot as plt
    c0, span, z0, zspan = spans(r)
    x = read_span(r.dataset, r.filename, c0, span)
    f, t, S = spectrogram(x)
    fig = plt.figure(figsize=BLIND_FIG)
    axes = [fig.add_axes(BLIND_AX["context"]), fig.add_axes(BLIND_AX["zoom"])]
    show(axes[0], f, t, S, c0 - r.start_s)
    axes[0].set_xlim(c0 - r.start_s, c0 + span - r.start_s)
    axes[0].set_title(f"context, {span:.0f} s", fontsize=9, loc="left")
    f2, t2, S2 = spectrogram(x, 128)                 # finer in time for the zoom
    show(axes[1], f2, t2, S2, c0 - r.start_s)
    axes[1].set_xlim(z0 - r.start_s, z0 + zspan - r.start_s)
    axes[1].set_title("zoom on the marked window (finer time resolution)", fontsize=9, loc="left")
    axes[1].set_xlabel("seconds from the start of the window")
    for ax in axes:
        mark(ax, 0, r.end_s - r.start_s)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return x


def write_audio(x: np.ndarray, path: Path):
    """Sped up SPEED times (2500 Hz), then resampled to 8 kHz, which every browser plays."""
    import soundfile as sf
    from scipy.signal import resample_poly
    x = x - x.mean()
    x = resample_poly(x, 16, 5)                        # 2500 Hz -> 8000 Hz
    x = 0.9 * x / max(np.abs(x).max(), 1e-9)
    sf.write(str(path), x.astype(np.float32), 8000, subtype="PCM_16")


def reveal_figure(r, gt: pd.DataFrame, nave: dict, models: dict, path: Path):
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    c0, span, _, _ = spans(r)
    x = read_span(r.dataset, r.filename, c0, span)
    f, t, S = spectrogram(x)
    tracks = [("labels", gt)] + [(f"NAVE {s}", e) for s, e in nave.items()] + \
             [("YOLO", models["yolo"]), ("Whale-VAD", models["whalevad"])]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4]})
    show(axes[0], f, t, S, c0)
    g = gt[(gt.dataset == r.dataset) & (gt.filename == r.filename) & (gt.end_s > c0) & (gt.start_s < c0 + span)]
    for q in g.itertuples():
        axes[0].add_patch(mp.Rectangle((q.start_s, q.low_frequency), q.end_s - q.start_s, q.high_frequency - q.low_frequency,
                                       fill=False, ec=CLASS_COLOR[q.cls], lw=1.5))
        axes[0].text(q.start_s, q.high_frequency + 1, q.fine, fontsize=7, color=CLASS_COLOR[q.cls])
    ax = axes[1]
    for i, (name, ev) in enumerate(tracks):
        e = ev[(ev.dataset == r.dataset) & (ev.filename == r.filename) & (ev.end_s > c0) & (ev.start_s < c0 + span)]
        for c in CLASSES:
            ec = e[e.cls == c]
            ax.hlines(np.full(len(ec), i), ec.start_s, ec.end_s, color=CLASS_COLOR[c], lw=6)
    ax.set_yticks(range(len(tracks)), [n for n, _ in tracks], fontsize=8)
    ax.set_ylim(len(tracks) - 0.5, -0.5)
    ax.set_xlim(c0, c0 + span)
    ax.set_xlabel(f"seconds in {r.filename}")
    for a in axes:
        mark(a, r.start_s, r.end_s)
    for c in CLASSES:
        ax.plot([], [], color=CLASS_COLOR[c], lw=6, label=c)
    ax.legend(fontsize=7, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    extra = f", agreement {int(r.agree)} of 4" if r.group == "fa" else (f", label {r.fine}" if isinstance(r.fine, str) else "")
    axes[0].set_title(f"{r.item_id}  {r.group}  {r.cls}  {r.dataset}{extra}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------- pack

def pack(a) -> None:
    from . import seed_agree, seeds                  # torch, only needed here
    rng = np.random.default_rng(SEED)
    out = Path(a.pack)
    if out.exists():
        shutil.rmtree(out)
    for d in ("blind/img", "blind/audio", "key", "reveal"):
        (out / d).mkdir(parents=True, exist_ok=True)
    gt = core.load_labels("validation")
    seed_list = list(seeds.SEEDS)
    with ProcessPoolExecutor(min(a.workers, len(seed_list))) as ex:
        nave = dict(zip(seed_list, ex.map(seed_agree.seed_events, seed_list)))
    say("NAVE events of all seeds at their LOSO thresholds")
    hits = {}
    for s, ev in nave.items():
        _, hits[s] = seed_agree.match(ev, gt)
    anchor = nave[seed_agree.ANCHOR].reset_index(drop=True)
    anchor["gt_match"], _ = seed_agree.match(anchor, gt)
    anchor["tp"] = anchor.gt_match >= 0
    anchor["agree"] = seed_agree.agreement(anchor, nave)

    lab = pd.read_parquet(TABLES / "labels_wide_loso.parquet").set_index("label_id")
    L = gt.set_index("label_id").join(lab[["snr_db", "hit_yolo", "hit_whalevad"]])
    L["n_seeds_hit"] = pd.DataFrame({s: hits[s] for s in seed_list}).sum(axis=1)
    L[["hit_yolo", "hit_whalevad"]] = L[["hit_yolo", "hit_whalevad"]].fillna(False).astype(bool)
    others = L.hit_yolo | L.hit_whalevad
    L = L.reset_index()

    fa_pool = anchor[~anchor.tp & (anchor.agree == 4)].assign(group="fa")
    miss_pool = L[(L.n_seeds_hit == 0).to_numpy() & ~others.to_numpy()].assign(group="miss")
    hit_pool = L[(L.n_seeds_hit == 5).to_numpy() & (L.hit_yolo & L.hit_whalevad).to_numpy()].assign(group="hit")
    models = load_models()
    busy = pd.concat([anchor[["dataset", "filename", "start_s", "end_s"]]]
                     + [m[["dataset", "filename", "start_s", "end_s"]] for m in models.values()]
                     + [e[["dataset", "filename", "start_s", "end_s"]] for e in nave.values()])
    fa_dur = {c: (fa_pool[fa_pool.cls == c].end_s - fa_pool[fa_pool.cls == c].start_s).to_numpy() for c in CLASSES}

    picks = [stratified(fa_pool, a.n_fa, ["dataset", "cls"], rng),
             stratified(miss_pool, a.n_miss, ["dataset", "cls"], rng),
             stratified(hit_pool, a.n_hit, ["cls"], rng),
             background(a.n_bg, fa_dur, gt, busy, rng)]
    cols = ["group", "dataset", "filename", "cls", "start_s", "end_s", "label_id", "fine", "agree", "conf",
            "low_frequency", "high_frequency", "snr_db"]
    items = pd.concat([p.reindex(columns=cols) for p in picks], ignore_index=True)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items.insert(0, "item_id", [f"i{k + 1:03d}" for k in range(len(items))])
    # how close each item is to any label of its class (a false alarm right next to a label is a boundary case)
    near = []
    for r in items.itertuples():
        g = gt[(gt.dataset == r.dataset) & (gt.filename == r.filename) & (gt.cls == r.cls)]
        if r.group in ("miss", "hit"):
            g = g[g.label_id != r.label_id]
        near.append(float(time_iou(r.start_s, r.end_s, g.start_s.to_numpy(), g.end_s.to_numpy()).max()) if len(g) else 0.0)
    items["max_iou_other_label"] = near
    say(f"{len(items)} items: " + ", ".join(f"{k} {v}" for k, v in items.group.value_counts().items()))

    rows = []
    for k, r in enumerate(items.itertuples()):
        x = blind_figure(r, out / "blind" / "img" / f"{r.item_id}.png")
        c0, span, z0, zspan = spans(r)
        write_audio(x, out / "blind" / "audio" / f"{r.item_id}_context.wav")
        write_audio(read_span(r.dataset, r.filename, z0, zspan), out / "blind" / "audio" / f"{r.item_id}_window.wav")
        reveal_figure(r, gt, nave, models, out / "reveal" / f"{r.item_id}.png")
        rows.append({"item_id": r.item_id, "site": r.dataset, "cls": r.cls, "look_for": CLASS_LONG[r.cls],
                     "window_s": round(r.end_s - r.start_s, 1)})
        if (k + 1) % 20 == 0:
            say(f"drawn {k + 1} / {len(items)}")
    pd.DataFrame(rows).to_csv(out / "blind" / "items.csv", index=False)
    items.to_csv(out / "key" / "items_key.csv", index=False)

    # pool sizes, needed to turn the sample into a corrected precision
    pool = []
    for s in VAL:
        for c in CLASSES:
            an = anchor[(anchor.dataset == s) & (anchor.cls == c)]
            a4 = an[an.agree == 4]
            pool.append({"dataset": s, "cls": c, "events": len(an), "tp": int(an.tp.sum()),
                         "events_agree4": len(a4), "tp_agree4": int(a4.tp.sum()), "fp_agree4": int((~a4.tp).sum()),
                         "labels": int(((L.dataset == s) & (L.cls == c)).sum()),
                         "labels_missed_by_all_7": int(((miss_pool.dataset == s) & (miss_pool.cls == c)).sum())})
    pd.DataFrame(pool).to_csv(out / "key" / "pool.csv", index=False)
    (out / "key" / "meta.json").write_text(json.dumps({"seed": SEED, "made": time.strftime("%Y-%m-%d %H:%M"),
                                                       "n": {k: int(v) for k, v in items.group.value_counts().items()},
                                                       "speed": SPEED, "context_s": CONTEXT_S}, indent=1))
    shutil.make_archive(str(out / "blind"), "zip", out, "blind")
    say(f"done: {out} (send blind.zip to the other annotator, keep key/ and reveal/ closed until both are done)")


# ------------------------------------------------------------------------------------- reference examples

FINE_NAME = {"bma": "A call", "bmb": "B call", "bmz": "Z call", "bmd": "blue whale D-call",
             "bpd": "fin whale downsweep", "bp20": "20 Hz pulse", "bp20plus": "20 Hz pulse with higher component"}
REF_PCT = 90                 # clear but typical: the label nearest the 90th SNR percentile of its site and fine label
REF_GAP_S = 120.0            # never within 2 min of any item


def ref_figure(r, path: Path):
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    dur = r.end_s - r.start_s
    pad = max(4.0, dur / 2)
    span = dur + 2 * pad
    x = read_span(r.dataset, r.filename, r.start_s - pad, span)
    f, t, S = spectrogram(x, 128)
    fig = plt.figure(figsize=REF_FIG)
    ax = fig.add_axes(REF_AX)
    show(ax, f, t, S, -pad)
    ax.set_xlim(-pad, dur + pad)
    mark(ax, 0, dur)
    ax.add_patch(mp.Rectangle((0, r.low_frequency), dur, r.high_frequency - r.low_frequency, fill=False,
                              ec=CLASS_COLOR[r.cls], lw=1.4))
    ax.set_title(f"{r.fine}: {FINE_NAME.get(r.fine, r.fine)}", fontsize=9, loc="left")
    ax.set_xlabel("s", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return x


def refs(a) -> None:
    """Reference examples next to every item: per dev site, one clear label of every fine label, never one of
    the items and never within REF_GAP_S of one. Fine labels a site does not have come from another dev site.
    Writes pack/blind/refs/ and refs.csv, then rebuilds blind.zip. Leaves the items untouched."""
    out = Path(a.pack)
    key = pd.read_csv(out / "key" / "items_key.csv")
    gt = core.load_labels("validation")
    snr = pd.read_parquet(TABLES / "labels_wide_loso.parquet")[["label_id", "snr_db"]]
    g = gt.merge(snr, on="label_id").dropna(subset=["snr_db"])
    used = set(key.label_id.dropna().astype(int))
    ok = []
    for r in g.itertuples():
        k = key[(key.dataset == r.dataset) & (key.filename == r.filename)]
        ok.append(r.label_id not in used and not ((k.start_s < r.end_s + REF_GAP_S) & (k.end_s > r.start_s - REF_GAP_S)).any())
    g = g[ok]
    d = out / "blind" / "refs"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rows = []
    for site in VAL:
        for fine in FINE_NAME:
            c = g[(g.dataset == site) & (g.fine == fine)]
            src = site
            if c.empty:
                c = g[g.fine == fine]
                if c.empty:
                    continue
                src = c.dataset.value_counts().index[0]
                c = c[c.dataset == src]
            target = np.percentile(c.snr_db, REF_PCT)
            r = c.iloc[int(np.argmin(np.abs(c.snr_db.to_numpy() - target)))]
            name = f"{site}_{fine}"
            x = ref_figure(r, d / f"{name}.png")
            write_audio(x, d / f"{name}.wav")
            rows.append({"site": site, "cls": r.cls, "fine": fine, "name": name, "from_site": src,
                         "caption": f"{fine}: {FINE_NAME[fine]}" + ("" if src == site else f" (from {src})")})
            say(f"ref {name} from {src}, snr {r.snr_db:.1f} dB")
    pd.DataFrame(rows).to_csv(d / "refs.csv", index=False)
    shutil.make_archive(str(out / "blind"), "zip", out, "blind")
    say(f"done: {len(rows)} references in {d}")


def redraw(a) -> None:
    """Redraw the item images with the fixed layout (items, audio and key unchanged), then the references."""
    out = Path(a.pack)
    key = pd.read_csv(out / "key" / "items_key.csv")
    for k, r in enumerate(key.itertuples()):
        blind_figure(r, out / "blind" / "img" / f"{r.item_id}.png")
        if (k + 1) % 20 == 0:
            say(f"redrawn {k + 1} / {len(key)}")
    refs(a)


# ------------------------------------------------------------------------------------- spectrogram player

PLAYER_JS = r"""
function render({ model, el }) {
  const wrap = document.createElement("div");
  wrap.style.position = "relative"; wrap.style.cursor = "pointer";
  const img = document.createElement("img");
  img.style.width = "100%"; img.style.display = "block";
  wrap.appendChild(img);
  const bar = document.createElement("div");
  Object.assign(bar.style, {display: "flex", gap: "16px", flexWrap: "wrap", marginTop: "4px"});
  el.append(wrap, bar);
  let players = [];
  function build() {
    players.forEach(s => s.audio.pause());
    wrap.style.maxWidth = model.get("width") + "px";
    img.src = "data:image/png;base64," + model.get("png");
    wrap.querySelectorAll(".aivis-ph").forEach(n => n.remove());
    bar.innerHTML = ""; players = [];
    for (const p of model.get("panels")) {
      const line = document.createElement("div");
      line.className = "aivis-ph";
      Object.assign(line.style, {position: "absolute", top: (p.top * 100) + "%", height: ((p.bottom - p.top) * 100) + "%",
        width: "2px", background: "#ff9f1c", boxShadow: "0 0 3px #ff9f1c", left: (p.left * 100) + "%",
        display: "none", pointerEvents: "none"});
      wrap.appendChild(line);
      const box = document.createElement("div");
      const lab = document.createElement("div");
      lab.textContent = p.label; lab.style.fontSize = "12px";
      const audio = document.createElement("audio");
      audio.controls = true; audio.preload = "auto"; audio.src = "data:audio/wav;base64," + p.audio;
      box.append(lab, audio); bar.appendChild(box);
      const st = {p, line, audio}; players.push(st);
      const tick = () => {
        if (!audio.duration) return;
        const f = Math.min(audio.currentTime / audio.duration, 1);
        line.style.left = ((p.left + f * (p.right - p.left)) * 100) + "%";
        line.style.display = "block";
        if (!audio.paused) requestAnimationFrame(tick);
      };
      audio.addEventListener("play", () => {
        players.forEach(o => { if (o !== st) { o.audio.pause(); o.line.style.display = "none"; } });
        requestAnimationFrame(tick);
      });
      audio.addEventListener("seeked", tick);
      audio.addEventListener("timeupdate", tick);
    }
  }
  wrap.addEventListener("click", e => {
    const r = wrap.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    for (const s of players) {
      const p = s.p;
      if (x >= p.left && x <= p.right && y >= p.top && y <= p.bottom) {
        const go = () => { s.audio.currentTime = (x - p.left) / (p.right - p.left) * s.audio.duration; s.audio.play(); };
        s.audio.duration ? go() : s.audio.addEventListener("loadedmetadata", go, {once: true});
      }
    }
  });
  model.on("change:png", build);
  model.on("change:panels", build);
  build();
  return () => players.forEach(s => s.audio.pause());
}
export default { render };
"""


def player_widget():
    """Spectrogram with audio players and a moving play line (click the spectrogram to jump). None without anywidget."""
    try:
        import anywidget
        import traitlets
    except ImportError:
        return None

    class Player(anywidget.AnyWidget):
        _esm = PLAYER_JS
        png = traitlets.Unicode("").tag(sync=True)
        panels = traitlets.List([]).tag(sync=True)
        width = traitlets.Int(1100).tag(sync=True)

    return Player


def b64(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode()


# ------------------------------------------------------------------------------------- labelling form

def labeller(annotator: str, pack: str | Path | None = None, out: str | Path | None = None):
    """ipywidgets form, one item at a time. Every save rewrites out/annotations_<annotator>.csv, so it can be
    closed and resumed any time."""
    import ipywidgets as W

    blind = Path(pack) if pack else (RESULTS / "blind" if (RESULTS / "blind" / "items.csv").exists() else PACK / "blind")
    if (blind / "blind" / "items.csv").exists():      # pointed at the unzipped parent folder
        blind = blind / "blind"
    items = pd.read_csv(blind / "items.csv")
    out = Path(out) if out else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"annotations_{annotator.strip().lower()}.csv"
    ans = {}
    if csv.exists():
        for r in pd.read_csv(csv, keep_default_na=False).itertuples():
            ans[r.item_id] = {"answer": r.answer, "other": r.other, "notes": r.notes}
    rf = blind / "refs" / "refs.csv"
    refs_tab = pd.read_csv(rf) if rf.exists() else None
    ids = items.item_id.tolist()
    state = {"i": next((k for k, i in enumerate(ids) if i not in ans), 0)}

    head, progress = W.HTML(), W.HTML()
    Player = player_widget()
    img = Player(width=1100) if Player else W.Image(format="png", layout=W.Layout(width="100%", max_width="1100px"))
    def player(path: Path, caption: str):
        return W.VBox([W.HTML(f"<span style='font-size:12px'>{caption}</span>"),
                       W.Audio(value=path.read_bytes(), format="wav", autoplay=False, loop=False, controls=True)])
    audio = W.HBox()
    ref_head, ref_box, tip = W.HTML(), W.HBox(layout=W.Layout(flex_flow="row wrap")), W.HTML()
    answer = W.ToggleButtons(options=ANSWERS, value=None, description="call?", button_style="")
    other = W.Dropdown(options=OTHER, value="none", description="also heard")
    notes = W.Text(placeholder="optional notes", layout=W.Layout(width="60%"))
    prev_b, next_b, jump_b = W.Button(description="< back"), W.Button(description="save, next >", button_style="primary"), \
        W.Button(description="first unanswered")

    def save():
        if answer.value is None:
            return
        ans[ids[state["i"]]] = {"answer": answer.value, "other": other.value, "notes": notes.value}
        df = pd.DataFrame([{"item_id": k, **v, "annotator": annotator} for k, v in ans.items()])
        tmp = csv.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(csv)

    def draw():
        i = state["i"]
        r = items.iloc[i]
        head.value = (f"<h3 style='margin:4px 0'>{r.item_id} &nbsp; ({i + 1} of {len(ids)}) &nbsp; site {r.site}</h3>"
                      f"<p style='font-size:15px'>Is there a <b>{r.look_for}</b> between the dashed lines "
                      f"({r.window_s} s)? Yes if at least one call of that type is in it, even partly.</p>")
        wav_w, wav_c = blind / "audio" / f"{r.item_id}_window.wav", blind / "audio" / f"{r.item_id}_context.wav"
        lab_w, lab_c = "zoom / window (10x faster)", f"context, {CONTEXT_S:.0f} s (10x faster)"
        if Player:
            img.png = b64(blind / "img" / f"{r.item_id}.png")
            img.panels = [panel(BLIND_AX["zoom"], b64(wav_w), lab_w), panel(BLIND_AX["context"], b64(wav_c), lab_c)]
        else:
            img.value = (blind / "img" / f"{r.item_id}.png").read_bytes()
            audio.children = [player(wav_w, lab_w), player(wav_c, lab_c)]
        cells = []
        if refs_tab is not None:
            for q in refs_tab[(refs_tab.site == r.site) & (refs_tab.cls == r.cls)].itertuples():
                png, wav = blind / "refs" / f"{q.name}.png", blind / "refs" / f"{q.name}.wav"
                if Player:
                    cell = Player(png=b64(png), panels=[panel(REF_AX, b64(wav), q.caption)], width=380)
                else:
                    cell = W.VBox([W.Image(value=png.read_bytes(), format="png", layout=W.Layout(width="360px")),
                                   player(wav, q.caption)])
                cells.append(W.Box([cell], layout=W.Layout(margin="0 12px 12px 0", width="390px")))
            ref_head.value = (f"<p style='margin:14px 0 2px'><b>References</b>: labelled {r.cls} calls from {r.site} "
                              f"for comparison (box = the label, audio 10x faster)</p>")
        ref_box.children = cells
        tip.value = ("<p style='font-size:12px;color:#52514e;margin:2px 0 8px'>Play an audio file and an orange line "
                     "runs over its panel; click on a spectrogram to jump there.</p>") if Player else ""
        a = ans.get(r.item_id, {})
        answer.value = a.get("answer") or None
        other.value = a.get("other") or "none"
        notes.value = a.get("notes", "")
        progress.value = f"<b>{len(ans)} of {len(ids)} answered</b>, saved to {csv}"

    def go(step):
        def f(_):
            save()
            state["i"] = min(max(state["i"] + step, 0), len(ids) - 1)
            draw()
        return f

    def jump(_):
        save()
        state["i"] = next((k for k, i in enumerate(ids) if i not in ans), state["i"])
        draw()

    prev_b.on_click(go(-1))
    next_b.on_click(go(1))
    jump_b.on_click(jump)
    draw()
    return W.VBox([head, img, audio, tip, answer, W.HBox([other, notes]), W.HBox([prev_b, next_b, jump_b]), progress,
                   ref_head, ref_box])


# ------------------------------------------------------------------------------------- report

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def fleiss(W: pd.DataFrame) -> float:
    """Fleiss' kappa over the items every annotator answered (rows items, columns annotators)."""
    W = W.dropna()
    if W.empty or W.shape[1] < 2:
        return np.nan
    n = W.shape[1]
    counts = np.stack([(W == c).sum(axis=1).to_numpy() for c in ANSWERS], axis=1)
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n * (n - 1))
    p_j = counts.sum(axis=0) / counts.sum()
    pe = float((p_j ** 2).sum())
    return (float(p_i.mean()) - pe) / (1 - pe) if pe < 1 else np.nan


def load_answers(results: Path) -> pd.DataFrame:
    parts = [pd.read_csv(f, keep_default_na=False).assign(annotator=f.stem.removeprefix("annotations_"))
             for f in sorted(results.glob("annotations_*.csv"))]
    if not parts:
        raise SystemExit(f"no annotations_*.csv in {results}")
    return pd.concat(parts, ignore_index=True)


def report(a=None, pack: str | Path | None = None, results: str | Path | None = None, quiet: bool = False) -> dict:
    """Agreement, controls, share of real calls per group and class, corrected precision. Returns the tables
    and writes them to results/phase1/listening/."""
    pack = Path(pack or (a.pack if a else PACK))
    results = Path(results or (a.results if a else RESULTS))
    key = pd.read_csv(pack / "key" / "items_key.csv")
    pool = pd.read_csv(pack / "key" / "pool.csv")
    A = load_answers(results)
    names = sorted(A.annotator.unique())
    W = A.pivot(index="item_id", columns="annotator", values="answer")
    out = {}

    # quality: agreement between annotators and the two controls
    q = []
    for n in names:
        k = key.merge(A[A.annotator == n], on="item_id")
        pos, neg = k[k.group == "hit"], k[k.group == "bg"]
        q.append({"annotator": n, "answered": len(k), "of": len(key),
                  "hit_said_yes": float((pos.answer == "yes").mean()) if len(pos) else np.nan,
                  "bg_said_no": float((neg.answer == "no").mean()) if len(neg) else np.nan,
                  "unsure": float((k.answer == "unsure").mean())})
    Q = pd.DataFrame(q)
    if len(names) >= 2:
        both = W.dropna()
        pair = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = both[names[i]], both[names[j]]
                pair.append({"pair": f"{names[i]} / {names[j]}", "items": len(both), "agree": float((x == y).mean()),
                             "kappa_3way": kappa(x, y),
                             "kappa_yes_vs_rest": kappa(x == "yes", y == "yes")})
        ag = pd.DataFrame(pair)
        if len(names) >= 3:
            ag = pd.concat([ag, pd.DataFrame([{"pair": "all (Fleiss)", "items": len(both), "agree": np.nan,
                                                "kappa_3way": fleiss(W[names]),
                                                "kappa_yes_vs_rest": fleiss((W[names] == "yes").where(W[names].notna())
                                                                            .replace({True: "yes", False: "no"}))}])],
                           ignore_index=True)
        out["agreement"] = ag
    out["quality"] = Q

    # consensus: with one or two annotators everyone has to agree; with three or more a majority decides
    # (more than half of those who answered), otherwise unclear
    def cons(row):
        v = [x for x in row if isinstance(x, str)]
        if not v:
            return np.nan
        need = len(v) if len(v) <= 2 else len(v) // 2 + 1
        if sum(x == "yes" for x in v) >= need:
            return "real"
        if sum(x == "no" for x in v) >= need:
            return "not"
        return "unclear"
    key["consensus"] = key.item_id.map(W.apply(cons, axis=1))
    key["n_annotators"] = key.item_id.map(W.notna().sum(axis=1)).fillna(0).astype(int)
    k = key.dropna(subset=["consensus"])
    rows = []
    for (g, c), d in k.groupby(["group", "cls"]):
        n, real, unclear = len(d), int((d.consensus == "real").sum()), int((d.consensus == "unclear").sum())
        lo, hi = wilson(real, n)
        rows.append({"group": g, "cls": c, "items": n, "real": real, "not": int((d.consensus == "not").sum()),
                     "unclear": unclear, "share_real": real / n, "ci_lo": lo, "ci_hi": hi,
                     "share_real_if_unclear_real": (real + unclear) / n})
    out["shares"] = S = pd.DataFrame(rows)
    out["shares_site"] = (k.groupby(["group", "dataset", "cls"]).consensus.value_counts().unstack(fill_value=0)
                          .reset_index())

    # corrected precision of the full-agreement events: false alarms judged real count as hits
    fa = k[k.group == "fa"]
    pc = []
    for c in CLASSES:
        tp = fp = add_lo = add_hi = 0.0
        f_c = fa[fa.cls == c]
        for r in pool[pool.cls == c].itertuples():
            d = f_c[f_c.dataset == r.dataset]
            d = d if len(d) else f_c                     # empty cell: pooled share of the class
            if not len(d):
                continue
            tp += r.tp_agree4
            fp += r.fp_agree4
            add_lo += r.fp_agree4 * float((d.consensus == "real").mean())
            add_hi += r.fp_agree4 * float(d.consensus.isin(["real", "unclear"]).mean())
        n = tp + fp
        if n:
            pc.append({"cls": c, "events_agree4": int(n), "precision_labelled": tp / n,
                       "precision_corrected": (tp + add_lo) / n, "precision_corrected_hi": (tp + add_hi) / n})
    out["precision"] = pd.DataFrame(pc)

    results.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(results / f"report_{name}.csv", index=False)
    if not quiet:
        pd.set_option("display.width", 200)
        for name, df in out.items():
            print(f"\n== {name}\n" + df.round(3).to_string(index=False))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    q = sub.add_parser("pack")
    q.add_argument("--pack", default=str(PACK))
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--n-fa", type=int, default=45)
    q.add_argument("--n-miss", type=int, default=45)
    q.add_argument("--n-hit", type=int, default=10)
    q.add_argument("--n-bg", type=int, default=10)
    f = sub.add_parser("refs")
    f.add_argument("--pack", default=str(PACK))
    d = sub.add_parser("redraw")
    d.add_argument("--pack", default=str(PACK))
    r = sub.add_parser("report")
    r.add_argument("--pack", default=str(PACK))
    r.add_argument("--results", default=str(RESULTS))
    a = p.parse_args(argv)
    {"pack": pack, "refs": refs, "redraw": redraw, "report": report}[a.step](a)


if __name__ == "__main__":
    main()
