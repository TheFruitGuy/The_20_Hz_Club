# Decisions

Anything that changes how numbers are produced. Newest at the bottom.

| date | decision | why | who |
|---|---|---|---|
| 18.10.2026 | one shared environment (Python 3.11, torch 2.8, ultralytics 8.3.91, pandas 2) for all three models | the baseline needs torchaudio 2.8 and pandas 2; Whale-VAD needs Python 3.11+ | Matthias |
| 18.10.2026 | Whale-VAD: 7 outputs grouped by maximum (0-2 bmabz, 3-4 d, 5-6 bp) | weight correlations and synthetic calls; to be confirmed on dev calls | Matthias |
| 18.10.2026 | Whale-VAD uses NAVE's post-processing, thresholds 0.5 unless tuned | no post-processing or thresholds published | Matthias |
| 18.10.2026 | scoring with evaluation_submissions.py (leaderboard), not evaluation.py | evaluation.py drops the D class | Matthias |
| 22.10.2026 | Phase I exports: all validation and evaluation sites, YOLO at confidence 0.01 (baseline output = rows with confidence >= 0.1), frame probabilities saved for Whale-VAD and NAVE, one run folder per model and site in outputs/phase1/ | thresholds can then be swept and tuned for all three models the same way without running the models again | Matthias |
| 22.10.2026 | Phase I thresholds: grid of 0.05 for the frame models, 0.01 to 0.95 for YOLO; LOSO over the three validation sites; eval sites use thresholds tuned on all three; all models treated the same | a fair comparison needs the same tuning for everyone; NAVE's checkpoint thresholds were in-sample, Whale-VAD had none | Matthias |
| 22.10.2026 | Phase I matching: own fast copy of the official scorer, checked to give identical counts | sweeps over thresholds and IoU need thousands of scorings | Matthias |
| 23.10.2026 | frequency-shift augmentation: single-sideband shift, uniform -1.5 to +1.5 Hz, on 80 % of training segments, everything else NAVE's own training (5 seeds) | covers the measured drift (about 2 Hz from 2005 to 2020) in both directions without leaving the band of the calls; symmetric so the model is not told which way the drift goes | Matthias |
| 23.10.2026 | ladder checkpoints: nave_best.pt of the original ladder runs; where a config was run twice for a seed, the earlier complete run (the one from the ladder period) | the later k31 runs belong to the kernel sweep | Matthias |
| 23.10.2026 | input variants for M1 to M4: nomag (cos, sin, PCEN) and logmag (log(|STFT| + 1e-6) minus its mean over the tile per frequency bin, in place of the linear magnitude), both with the full NAVE recipe | nomag asks whether the magnitude is needed at all; logmag keeps the magnitude but makes a gain an additive shift that the per-bin mean removes | Matthias |
| 25.10.2026 | listening check (LC1 to LC4): blind, two annotators, label and background controls | dev labels are incomplete, so false alarms may be real calls; two blind annotators so the check does not rest on one person's judgement, the same weakness as the labels | Matthias |
