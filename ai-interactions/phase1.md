# Phase I: important AI interactions (AI-aided pair)

Phase I question, agreed by all four of us: how do YOLO, Whale-VAD and NAVE behave on BioDCASE 2026
Task 2 beyond one leaderboard score? Test ids (C1, F1, ...) refer to [`preregistration.md`](preregistration.md).

Each entry: what we asked for, what the AI produced, and what we decided.

## 1. Setup: one environment and one scorer for three models (18.10.2026)

- **We asked for:** one repo in which all four of us run the three models and the official scorer the
  same way on Windows, macOS and Linux, with versions pinned in one place.
- **AI produced:** pinned repos and weights in `config/external.toml`, one conda environment, a
  resumable setup script, a smoke test on a synthetic recording, and a fast copy of the official matcher
  for threshold sweeps.
- **We decided:** the fast matcher is only used after it gives identical tp/fp/fn to
  `evaluation_submissions.py`; scoring uses the leaderboard script because `evaluation.py` drops the D
  class.
- **Effect:** the human-only pair could start from the same outputs on a laptop.

## 2. Turning "behaviour" into views and probes (around 20 to 22.10.2026)

- **We asked for:** per-call views and controlled probes, not scores. The three detectors read the audio
  in different ways (YOLO: 30 s spectrogram images every 15 s; Whale-VAD: 30 s tiles; NAVE: 60 s tiles
  with a PCEN channel), so each design should fail in its own places. Every model on the same thresholds
  and matching.
- **AI produced:** ten hypotheses (C1 to C10), one view or probe each: an outcome per label and per
  detection (hit, merged, wrong class, missed almost, duplicate, background), rankings under 18 scoring
  variants, agreement between models on the unlabelled eval sites, and synthetic probes (shift a call in
  frequency, change the level, add a distant transient, move a call to a tile edge, swap the background).
- **We decided:**
  - keep the probe set and the outcome categories (`aivis/phase1/`);
  - treat the first write-up the AI drafted from these views as exploratory only, and require a
    pre-registered follow-up for anything that goes into a presentation. That rule shaped the rest of
    Phase I;
  - not use model agreement as a precision estimate. Models trained on the same data share false
    alarms, so the estimate needed a check on a labelled dev site first: 0.5 to 0.6 for BMABZ on
    kerguelen2014 against a real 0.8 to 0.93. It stays a marked hint; precision on unlabelled audio comes
    from the blind listening check (section 6).

## 3. Frequency drift: from a probe result to the main finding (22 to 23.10.2026)

- **Starting point:** the shift probe showed all three models losing Z-calls within half a hertz, which
  matches the documented downward drift of Antarctic blue whale calls.
- **We asked for:** a direct measurement of the Z-call peak per year, a test on real dev recordings
  shifted to the 2019/20 pitch (F1, F2), and its replication over seeds and ablation rungs.
- **AI produced:** the measurement (27.3 Hz in 2005 to 25.3 Hz in 2020), the shifted-audio test (BMABZ
  recall down 16 to 36 points at -0.6 Hz), five NAVE seeds (S2, S3) and the ablation ladder (L1 to L4).
- **We decided:** frequency-shift augmentation with a symmetric range of -1.5 to +1.5 Hz, so the model
  is not told the direction of the drift.
- **Kept as refuted:** L2 (the frequency-dynamic stem would make NAVE more shift-sensitive) and the "D and
  BP stay within 10 %" part of F2.

## 4. NAVE's two weak spots: tile edges and loudness (23.10.2026 onwards)

- **We asked for:** recall by distance to the nearest 60 s tile edge and a fix without retraining; for
  loudness, the cause of the +20 dB drop and which input change removes it.
- **AI produced:** half-tile overlap with centre stitching (E1, E2); tests of PCEN against magnitude
  (S5), level mixing (G1 to G3), rescaling inside the network (H1, H2) and inputs without the magnitude
  (M1 to M4).
- **We decided:** centre stitching goes into the pipeline (+0.025 macro F1, no retraining); the input
  without magnitude is the lead for Phase II (no level dependence, M1). G1 and H2 were refuted and stay
  in the record as negative results instead of trying variants until one worked.

## 5. Seeds, shared misses and the ensemble (with the human-only pair's results)

- **Context:** the human-only pair found first that the NAVE seeds miss the same Z-calls, and asked how
  an ensemble of the three models would do.
- **We asked for:** the seed-miss result on all classes and with YOLO and Whale-VAD added, and fusion,
  routing and voting of the three models.
- **AI produced:** 36 % of all labels missed by all five seeds (2.5 % expected by chance), 29 % by all
  seven models; fusion +0.006 (EN1), routing no gain (EN2), voting trades recall for precision (EN3),
  fusion leans on YOLO for Z-calls (EN4).
- **We caught:** the first fusion run made NAVE look 0.18 worse than its known single-model score. We
  traced it to a float32 rounding issue in the AI-written fusion code by comparing against the known
  number; without that check the ensemble table would have been wrong.

## 6. Listening check: how far can we trust the labels? (25.10 to 02.11.2026)

- **We asked for:** a blind sample of detections, labels and background controls (LC1 to LC4), packed
  into a notebook each of us fills in, with the agreement read out automatically.
- **AI produced:** the sample pack, a notebook with an audio player and automatic Fleiss kappa.
- **We decided:**
  - all four of us label, not two, so no single judgement decides;
  - labelled reference examples per fine label in the notebook, from the same site as the item, always
    shown below it;
  - a playhead on the spectrogram, so the sound and the picture can be matched.
- **We caught:** one reference (Kerguelen 2015 D) shows the call outside its label box. We found it
  when checking the automatically picked references by eye; it is still to be replaced.

## 7. Figures for the midpoint (29.10 to 02.11.2026)

- **We asked for:** several alternative ways to show the numbers of each key slide, clean and fast to
  read, in the deck palette and from the real tables, before changing any slide.
- **AI produced:** five to six options per figure (wall of calls, chance comparison, alluvial,
  spectrogram gallery, swarm; dumbbell and slope charts).
- **We decided:** the chance comparison for shared misses (it puts the baseline next to the result).
  After feedback from Amal and the team we replaced the other figures with ones that show
  the mechanism: where NAVE listens in pitch, how far faint calls stay below the threshold, and what
  loudness does to found calls and false alarms (`aivis/phase1/slidefigs.py`, `results/phase1/slides/`).
- **We corrected:** "macro B" (our internal name) back to F1, consistent percentage points, and an
  academic citation for the Deep Voice report.

## Failure patterns we noticed

- Exploratory results written up as findings; confident text before any test was registered.
- Numerical bugs that look like results (float32 in the fusion code).
- Plausible but wrong predictions: close to half of the registered tests were refuted, not supported or
  only partly supported, several of them AI-proposed mechanisms (L2, G1, EN4, D1, D7).
- Internal shorthand leaking into communication (macro B, pp).
- Estimates that look like measurements (agreement-based precision).

## What worked

- Pre-registration before every run: it turned fast AI output into tests we can trust or refute.
- Checking against a number we already know (official scorer counts, single-model scores).
- Asking for several options and choosing ourselves, for analyses and for figures.
- Keeping humans on the label question: a blind listening check instead of the AI's judgement.
