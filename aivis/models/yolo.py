"""The YOLO baseline (YOLO11s on spectrogram images), run without the interactive prompts.

Same pipeline as the baseline scripts (baselines/yolo in the task repo): 30 s windows every 15 s,
the baseline's own spectrogram function (band-pass 5 to 124 Hz, linear magnitude, contrast per window),
imgsz 640, confidence 0.1, class-agnostic NMS, no merging across windows, and the same box-to-time
conversion. The only difference is that the images stay in memory instead of being written as PNGs
(--save-images writes them too). All raw boxes are kept in boxes_raw.parquet, so other time mappings
or cross-window merging can be tried later without running the model again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from .. import io, paths


def baseline_dataset(workdir: Path):
    folder = paths.use_repo("task2", "baselines/yolo")
    import preprocess_data   # noqa: E402  (from external/task2_2025_2026/baselines/yolo)
    config = json.loads((folder / "dataset_config.json").read_text(encoding="utf-8"))
    workdir.mkdir(parents=True, exist_ok=True)     # the baseline creates images/ and labels/ in here
    return preprocess_data.YOLODataset(config, str(workdir)), config


def windows(duration_s: float, win_s: float, step: float):
    i = 0.0
    while i * win_s < duration_s:   # same loop as convert_challenge_annotations_to_yolo_evaluation
        yield i
        i += step


def boxes_to_rows(boxes: pd.DataFrame, win_s: float, f_top: float) -> list[dict]:
    """Same arithmetic as convert_yolo_detections_to_csv in the baseline."""
    if boxes.empty:
        return []
    b = boxes
    start = (b.x - b.w / 2 + b.window) * win_s
    end = b.w * win_s + start
    low = ((1 - (b.y + b.h / 2)) * f_top).clip(lower=0)
    high = ((1 - (b.y - b.h / 2)) * f_top).clip(upper=f_top)
    out = pd.DataFrame({"dataset": b.dataset, "filename": b.filename, "annotation": b.name,
                        "start_s": start, "end_s": end, "confidence": b.conf,
                        "low_frequency": low, "high_frequency": high})
    return out.to_dict("records")


def predict(recs, run: io.Run, device: str, weights: Path | None = None, conf: float = 0.1,
            imgsz: int = 640, batch_size: int = 32, save_images: bool = False, **_):
    from ultralytics import YOLO
    weights = Path(weights or paths.weights("yolo_baseline"))
    if not weights.exists():
        raise FileNotFoundError(f"{weights} is missing (`python scripts/aivis_setup.py weights` downloads it)")
    ds, config = baseline_dataset(run.dir / "yolo_work")
    model = YOLO(str(weights))
    names = model.names
    win_s, step = config["duration"], config["overlap"]
    block = int(win_s * config["desired_fs"])
    ydev = "0" if device == "cuda" else device

    raw, batch = [], []

    def flush():
        if not batch:
            return
        results = model.predict([b[2] for b in batch], imgsz=imgsz, conf=conf, agnostic_nms=True,
                                device=ydev, verbose=False, save=False)
        for (rec, i, _), r in zip(batch, results):
            xywhn = r.boxes.xywhn.cpu().numpy()
            for (x, y, w, h), c, k in zip(xywhn, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                raw.append({"dataset": rec.dataset, "filename": rec.filename, "window": i, "cls": int(k),
                            "name": names[int(k)], "conf": float(c), "x": float(x), "y": float(y),
                            "w": float(w), "h": float(h)})
        batch.clear()

    for rec in tqdm(recs, desc="yolo", unit="file"):
        with sf.SoundFile(str(rec.path)) as f:
            for i in windows(rec.duration_s, win_s, step):
                f.seek(int(i * block))
                chunk = f.read(block, dtype="float32", always_2d=True)[:, 0]
                if len(chunk) < block:
                    chunk = np.pad(chunk, (0, block - len(chunk)))
                img, _ = ds.create_chunk_spectrogram(torch.from_numpy(chunk))
                gray = np.ascontiguousarray(np.flipud(img))
                if save_images:
                    from PIL import Image
                    Image.fromarray(gray).save(ds.images_folder / f"{rec.dataset}_{Path(rec.filename).stem}_{i}.png")
                batch.append((rec, i, np.repeat(gray[:, :, None], 3, axis=2)))   # what cv2 makes of the grey PNG
                if len(batch) >= batch_size:
                    flush()
    flush()

    boxes = pd.DataFrame(raw, columns=["dataset", "filename", "window", "cls", "name", "conf", "x", "y", "w", "h"])
    boxes.to_parquet(run.dir / "boxes_raw.parquet", index=False)
    rows = boxes_to_rows(boxes, win_s, config["desired_fs"] / 2 - ds.F_MIN)
    return rows, {"weights": str(weights), "weights_sha256": io.sha256(weights, 12), "classes": names,
                  "conf": conf, "imgsz": imgsz, "window_s": win_s, "window_step": step,
                  "spectrogram": {k: config[k] for k in ("nfft", "win_len", "hop_len", "log", "desired_fs")}}
