# Phase I: what runs, what comes out

One command, resumable (stop it any time, start it again):

```bash
caffeinate -i python -m aivis.phase1.run_all     # macOS; elsewhere without caffeinate
```

On a recent MacBook this takes about 12 h for the predictions (validation sites first, about 5 h,
then the evaluation sites) and about 2 h for the probes. `--skip-probes` leaves the probes out.
Progress: `outputs/phase1/progress.log`. Then open `notebooks/02_phase1.ipynb`.

| step | module | output in `outputs/phase1/` |
|---|---|---|
| predictions, all val + eval sites, YOLO at conf 0.01, frame probabilities saved | `predict_all` | `<model>/<site>/` |
| label table with SNR, level and noise per recording | `build labels`, `build sitestats` | `tables/labels_validation.parquet`, `tables/site_stats.parquet` |
| frame models: events at 19 thresholds, peak score per label | `build grid` | `tables/grid/`, `tables/peaks/` |
| tp/fp/fn per model, site, class, threshold, IoU 0.1/0.3/0.5 | `build sweep` | `tables/sweep.parquet` |
| default, in-sample and leave-one-site-out thresholds | `build thresholds` | `tables/thresholds.json` |
| events at those thresholds, matched with the official rules | `build events` | `tables/events_<model>_<variant>.parquet` |
| outcome per label and per event, false-alarm regions (idea 1) | `anatomy` | `tables/label_outcomes_loso.parquet`, `event_outcomes_...`, `labels_wide_...`, `regions_...` |
| ranking under 18 scoring variants (idea 5) | `summaries ranking` | `tables/ranking.parquet` |
| agreement and estimated precision on the new sites (idea 2) | `summaries agreement` | `tables/regions_all.parquet`, `agreement.parquet`, `combo_precision_dev.parquet` |
| blind audit sample (idea 2) | `summaries audit` | `audit/audit_sheet.csv` (key separate) |
| probes: level, transient, shift, truncation, context, density, background swap, SNR, position (ideas 3, 4) | `probes` | `probes/`, `tables/probe_results.parquet` |
| all figures | `views` | `results/phase1/figures/` |

## Choices worth knowing

- Matching is a fast copy of `evaluation_submissions.py` (same file order, same greedy rule, same
  2 min window) and gives identical tp/fp/fn. Events of the frame models are re-made from the saved
  float16 probabilities with NAVE's post-processing; against the float32 run this changed one event
  out of 1209 in a test (a value right at the threshold).
- Thresholds: frame models on a grid of 0.05 steps, YOLO on 0.01 to 0.95. LOSO here means over the
  three validation sites (tune on two, apply to the third). The evaluation sites use the thresholds
  tuned on all three. Every model gets the same treatment, NAVE's checkpoint thresholds and Whale-VAD's
  0.5 are only the "default" variant.
- Label SNR: mean power in the label's band during the label over the median band power in the 30 s
  around it (label-free frames only). It is (call + noise) / noise, so it never goes far below 0 dB.
- "Missed almost" uses a different score per model: the smoothed frame probability for the frame
  models and the best box confidence (IoU > 0.1) for YOLO.
- Probe SNR is call power over noise power in the call's band, a nominal value: measured on the clips
  it comes out about 1.5 dB higher above 0 dB and is noisy below -6 dB.
- Probe backgrounds are 60 s stretches with no label and no event of any model; they can still contain
  other sounds (ships, ice, instruments), which is realistic but worth a look in the browser.
- The pasted-call probes lose the call's natural context, so their reference is "shift 0" (the
  pasted call without any change), not the untouched "base" clip.

## Seed tests (S1 to S6)

All five NAVE checkpoints through the same analyses: `python -m aivis.phase1.seeds run`, then
`python -m aivis.phase1.seeds report`. Needs the main run, the probe clips and the shifted audio of the
follow-ups. About 1.2 GB of frame probabilities per seed in `outputs/phase1/seeds/`.

## Ablation ladder and frequency-shift augmentation (L1 to L4, A1 to A4)

- `python -m aivis.phase1.train_aug --seed S` retrains the full NAVE with frequency-shift augmentation
  (single sideband, -1.5 to +1.5 Hz on 80 % of training segments), otherwise NAVE's own training.
  Runs land in `outputs/phase1/aug_runs/`.
- `python -m aivis.phase1.ablate run` then `report` evaluates the four ladder rungs and the augmented
  models: dev scores, the -0.6 Hz shifted Kerguelen audio, the base/level/shift probes, and the original
  Kerguelen 2019/20 audio (D and AUG only). `ablate list` shows which checkpoints exist.

## Input variants (M1 to M4)

`python -m aivis.phase1.train_input --variant nomag|logmag --seed S` retrains NAVE without the linear
magnitude channel (nomag) or with a log magnitude normalized per frequency bin over each tile (logmag),
otherwise NAVE's own training. Runs land in `outputs/phase1/input_runs/`; `aivis.phase1.ablate` evaluates
them as rungs NOMAG and LOGMAG. The variant is stored in the checkpoint, `aivis.models.nave.load` rebuilds
the matching front end (`aivis/nave_input.py`).
