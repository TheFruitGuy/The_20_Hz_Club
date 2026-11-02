# Phase I: important AI interactions (AI-aided pair)

Phase I question, agreed by all four of us: how do YOLO, Whale-VAD and NAVE behave on BioDCASE 2026
Task 2 beyond one leaderboard score? Test ids (C1, F1, ...) refer to `logs/preregistration.md`.

## 1. Setup: one environment and one scorer for three models (18.10.2026)

- **Prompt (shortened):** set up one repo so all four of us can run the three models and the official
  scorer the same way, on Windows, macOS and Linux.
- **AI suggestion:** pinned repos and weights in `config/external.toml`, one conda environment, a
  resumable setup script, a smoke test on a synthetic recording; a fast copy of the official matcher for
  threshold sweeps.
- **Accepted, with a check:** the fast matcher had to give identical tp/fp/fn to
  `evaluation_submissions.py` before we used it. The AI also pointed out that `evaluation.py` drops the
  D class, so we score with the leaderboard script (`logs/decisions.md`, 18.10).
- **Effect:** the human-only pair could start from the same outputs on a laptop.

## 2. Turning "behaviour" into views and probes (around 20 to 22.10.2026)

- **Prompt (shortened):** Phase I is about behaviour, not scores. The three detectors read the audio in
  different ways (YOLO: 30 s spectrogram images every 15 s; Whale-VAD: 30 s tiles; NAVE: 60 s tiles with
  a PCEN channel), so each design should fail in its own places. Propose per-call views and controlled
  probes that test where each one breaks, and keep every model on the same thresholds and matching.
- **AI suggestion:** ten hypotheses (C1 to C10), one view or probe each: an outcome per label and per
  detection (hit, merged, wrong class, missed almost, duplicate, background), rankings under 18 scoring
  variants, agreement between models on the unlabelled eval sites, and synthetic probes (shift a call in
  frequency, change the level, add a distant transient, move a call to a tile edge, swap the background).
- **Accepted:** the probe set and the outcome categories, implemented in `aivis/phase1/`.
- **Changed:** the first write-up (`results/phase1/findings.md`) read like conclusions. We treated all of
  it as exploratory and required a pre-registered follow-up for anything that goes into a presentation.
  That rule shaped the rest of Phase I.
- **Rejected:** model agreement as a precision estimate on the new sites. Models that share training
  data also share false alarms, so the estimate needed a check on a dev site with labels, in-sample:
  it gave 0.5 to 0.6 for BMABZ on kerguelen2014, the real precision is 0.8 to 0.93. We use it only as a
  hint, marked as such; precision on unlabelled audio comes from the blind listening check (section 6).

## 3. Frequency drift: from a probe result to the main finding (22 to 23.10.2026)

- **AI suggestion:** the shift probe showed all three models losing Z-calls within half a hertz; the AI
  connected this to the known downward drift of Antarctic blue whale calls and proposed measuring the
  Z-call peak per year, a shifted-audio test on real dev recordings (F1, F2), and frequency-shift
  augmentation.
- **Accepted:** the measurement (27.3 Hz in 2005 to 25.3 Hz in 2020) and F1 (BMABZ recall falls by 16 to
  36 points at -0.6 Hz), replicated over five NAVE seeds (S2, S3) and the ablation ladder (L1 to L4).
- **Changed:** the augmentation range. We chose a symmetric -1.5 to +1.5 Hz so the model is not told the
  direction of the drift (`logs/decisions.md`, 23.10).
- **Refuted AI predictions:** L2 (the frequency-dynamic stem would make NAVE more shift-sensitive) and
  the "D and BP stay within 10 %" part of F2.

## 4. NAVE's two weak spots: tile edges and loudness (23.10.2026 onwards)

- **AI suggestion:** recall by distance to the nearest 60 s tile edge, then half-tile overlap with centre
  stitching (E1, E2); for loudness, check whether the PCEN channel or the magnitude causes the +20 dB
  drop (S5), and whether mixing levels, rescaling inside the network or an input without magnitude fixes
  it (G1 to G3, H1 to H2, M1 to M4).
- **Accepted:** E2 (+0.025 macro F1 without retraining) and the input variants (M1: no level
  dependence without the magnitude channel).
- **Refuted:** G1 (level mixing does not recover the full NAVE) and H2 (rescaling does not work after the
  Conformer blocks). We kept both as negative results instead of trying variants until one worked.

## 5. Seeds, shared misses and the ensemble (with the human-only pair's results)

- **Context:** the human-only pair found first that the NAVE seeds miss the same Z-calls, and asked how
  an ensemble of the three models would do.
- **Prompt (shortened):** extend the seed-miss result to all classes and to YOLO and Whale-VAD; test
  fusion, routing and voting of the three models.
- **AI suggestion and result:** 36 % of all labels are missed by all five seeds (2.5 % expected by
  chance), 29 % by all seven models; fusion +0.006 (EN1), routing no gain (EN2), voting trades recall for
  precision (EN3), fusion leans on YOLO for Z-calls (EN4).
- **Failure we caught:** the first fusion run made NAVE look 0.18 worse than its known single-model
  score. Cause: a float32 rounding issue in the AI-written fusion code. Comparing against the known
  single-model number found it; without that check the ensemble table would have been wrong.

## 6. Listening check: how far can we trust the labels? (25.10 to 02.11.2026)

- **Prompts (verbatim):** "build the sample pack, maybe make an notebook that is easy fillable for each
  one and then you can automatically read out the results"; "should the notebook not include a few
  labeled examples … before you code ask if unclear"; "possible to add a slider on the spectogramm so we
  can see then the sound should accure?"
- **AI suggestion:** a blind sample of detections, labels and background controls (LC1 to LC4), a
  notebook with an audio player, references per fine label and automatic Fleiss kappa.
- **Changed by us:** two annotators became all four of us; references from the same site; a playhead on
  the spectrogram.
- **Failure we caught:** one reference (Kerguelen 2015 D) shows the call outside its label box. The AI
  picked it automatically; we found it when checking the references by eye, and it is still to be replaced.

## 7. Figures for the midpoint (29.10 to 02.11.2026)

- **Prompt (shortened):** for each key slide, give us several alternative ways to show the same numbers,
  clean and fast to read, in the deck palette and from the real tables; do not change the slide yet.
- **AI suggestion:** five to six options per figure (wall of calls, chance comparison, alluvial,
  spectrogram gallery, swarm; dumbbell and slope charts), rendered from the real tables.
- **Chosen by us:** the chance comparison for shared misses (it puts the baseline next to the result),
  the dumbbell for drift (one before and after per model) and the slope chart for loudness (`aivis/phase1/slidefigs.py`).
- **Corrected by us:** the AI used our internal "macro B" in place of the metric name F1, mixed points and
  percentage points, and cited the Deep Voice report informally.

## Failure patterns we noticed

- Exploratory results written up as findings; confident text before any test was registered.
- Numerical bugs that look like results (float32 in the fusion code).
- Plausible but wrong predictions: close to half of the registered tests were refuted, not supported or
  only partly supported, several of them AI-proposed mechanisms (L2, G1, EN4, D1, D7).
- Internal shorthand leaking into communication (macro B, pp).
- Estimates that look like measurements (agreement-based precision).

## What worked

- Pre-registration before every run: it turned fast AI suggestions into tests we can trust or refute.
- Checking against a number we already know (official scorer counts, single-model scores).
- Asking for several options and choosing ourselves, for analyses and for figures.
- Keeping humans on the label question: a blind listening check instead of the AI's judgement.
