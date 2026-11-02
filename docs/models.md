# The models, the scorer, and what to watch out for

All three run through `python -m aivis.predict <model>`. The wrappers use each repo's own code where
it exists, so we test the models as published and not our re-implementation of them.

## YOLO baseline (`aivis/models/yolo.py`)

- Source: `external/task2_2025_2026/baselines/yolo` (commit in `config/external.toml`), YOLO11s
  pretrained on COCO, weights `task2_baseline_weights_2026.pt` from Zenodo 20281116. 9.4 M parameters.
- Input: 30 s windows every 15 s, the baseline's own spectrogram function: Butterworth band-pass
  5 to 124 Hz on each window, nfft 512, window 256, hop 20 (80 ms), linear magnitude, grey value scaled
  per window to the 98th percentile, image size 640.
- Output: boxes with confidence 0.1, class-agnostic NMS, no merging across the overlapping windows.
- The wrapper reproduces the baseline's PNG pipeline exactly (checked on a test file: same boxes, same
  times, same confidences), it only keeps the images in memory. `--save-images` writes the PNGs as well.
- Quirks to keep in mind:
  - every call shows up in two windows, so duplicates are common, and the scorer counts each extra one as a false positive
  - box times are stretched slightly near the window edges (up to about 0.5 s)
  - in training, a call only gets a label in the window where it starts
  - one loud sound changes the grey scale of the whole 30 s window
- `boxes_raw.parquet` keeps every box with its window, so other time mappings or cross-window
  de-duplication can be tested without running the model again.

## Whale-VAD (`aivis/models/whalevad.py`)

- Source: `external/Whale-VAD`, installed as the `whalevad` package; weights v0.1.0
  `WhaleVAD_ATBFL_3P-c6f6a07a.pt` from the GitHub release. CNN + 2-layer BiLSTM, 1.06 M parameters.
- Input: STFT 256 / hop 5 (20 ms frames, 129 bins), complex mean removed per frequency bin over the
  tile, channels magnitude (linear), cos and sin of the phase. Because of the mean removal, the tile
  length changes the input; we use 30 s tiles with 2 s overlap.
- Output: the released head has 7 units although the model was trained on the 3 joined classes. The
  weight vectors form three groups (outputs 0-2, 3-4, 5-6) and synthetic calls give the order
  bmabz, d, bp, so we take the maximum per group. All 7 are saved as `probs7` in the npz files.
  Confirm the grouping on real dev calls early on.
- No thresholds or post-processing were published. We use NAVE's post-processing and 0.5 for every
  class unless thresholds are given; `python -m aivis.tune` gives in-sample and LOSO thresholds.
- Times: frame f is placed at f * 0.02 s (start of the STFT window, same as NAVE). The window centre
  is 0.512 s later; whether Whale-VAD was trained with the start or the centre convention is unknown.
  For short D-calls this matters at IoU 0.3, so it is worth testing both.
- The magnitude input is linear and not normalised, so the recording level could matter.

## NAVE (`aivis/models/nave.py`)

- Source: `external/NAVE-SED`, checkpoint `pretrained/nave.pt` (k = 153, PCEN and FDY on). 2.71 M parameters.
- Input: the Whale-VAD channels plus a fixed PCEN channel, 60 s tiles with 2 s overlap (the setup of
  the reported numbers).
- Thresholds: from the checkpoint. They were tuned on all three dev sites, so dev scores with them are
  in-sample. The reported NAVE number is leave-one-site-out, which `python -m aivis.tune` reproduces.
- Post-processing: 500 ms median filter, threshold, merge gaps up to 0.5 s, keep events of 0.5 to 30 s.
- The configuration was picked on the dev sites (ablation ladder, kernel size), so its dev score is the
  best of several tries, while the YOLO baseline was not tuned at all. Keep that in mind when comparing
  dev scores.

## The tiling rule for both frame models

Tiles of the given length every (length - overlap) seconds while they fit, the same rule as NAVE's
evaluation. The last few seconds of a recording can stay uncovered; `--cover-tail` adds one tile that
ends at the end of the file.

## The scorer (`aivis/evaluate.py`)

- We use `evaluation/evaluation_submissions.py` of the task repo, the leaderboard scorer.
  Do not use `evaluation/evaluation.py`: it maps D-calls to "d" but scores "fm", so D drops out.
- Rules: 1D IoU above 0.3, greedy matching in the order of the prediction rows, labels looked up in a
  ±2 min window around each file's predictions, extra detections of one call count as false positives.
- The ±2 min window does not check the site. If two sites overlap in time, a detection could match a
  label of the other site; `--strict` scores each site separately (the scorer warns when sites overlap).
- macro B like the leaderboard: mean precision and mean recall over the classes, where a class without
  any detection is left out of the precision mean (pandas skips the NaN). macro A is the mean class F1.
- NAVE's own tuner and metric match the other way round (each label takes its best unmatched
  detection, IoU >= 0.3), so tuned thresholds are close to, but not exactly, optimal for the scorer.

## Not included yet

- DeepVoice (`--with-deepvoice`): code only, no weights. Retraining their single YOLOv11m needs a GPU
  (their recipe: 25 epochs, image size 320, about 61,000 images). Follow `external/biodcase-2026/yolo/README.md`
  inside the conda environment `aivis-deepvoice`.
- Training any of the models. The paper variants (YOLO without COCO, Whale-VAD in our pipeline, ...)
  will need training scripts; they go into `aivis/` when we get there.
