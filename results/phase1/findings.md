# Phase I, first findings (22.10.2026)

First pass over the full Phase I run (all validation and evaluation sites, sxrk11, commit 78e6b82).
All of this is **exploratory**: the expectations were not written into `logs/preregistration.md`
before the views were looked at. Before we use any of it in the midpoint, each finding below should get
a pre-registered follow-up test (suggested ones are listed).

Figures: `results/phase1/figures/`. Tables: `outputs/phase1/tables/`. Operating point everywhere:
leave-one-site-out (LOSO) thresholds over the three dev sites, IoU 0.3, official matching rules.

## Checks

- Our matcher gives the same tp/fp/fn as the official `evaluation_submissions.py`.
- NAVE: macro B 0.520 LOSO on the three dev sites (paper: 0.51 LOSO over all dev sites). Plausible.
- Whale-VAD: the grouping of its 7 outputs holds on real calls. On six dev files, bma/bmb/bmz peak on
  outputs 0 to 2, bmd/bpd on 3 and 4, bp20/bp20plus on 5 and 6, everything else stays near 0.
- Whale-VAD casey2017 BP: F1 0.00 here, their paper reports 0.025. Same picture.

| macro B, dev sites joined | YOLO | YOLO + NMS | Whale-VAD | NAVE |
|---|---|---|---|---|
| published/default thresholds | 0.355 | 0.451 | 0.378 | 0.542 |
| in-sample tuned | 0.328 | 0.421 | 0.374 | 0.537 |
| LOSO | 0.325 | 0.410 | 0.353 | 0.520 |

## 1. YOLO loses most of its precision to duplicates (C1, C2), fig 1b, 5b

22 to 55 % of YOLO's detections are a second detection of a call that is already hit (BP 55 %,
BMABZ 29 %, D 22 %). They come from the 50 % window overlap, which the baseline does not merge. The
scorer counts them as false positives. Simple temporal NMS (IoU 0.5) raises YOLO's macro B by about
0.09 in every variant. That alone moves YOLO from last to second, ahead of Whale-VAD (fig 5b).
NAVE stays first in all 18 scoring variants. The order of YOLO and Whale-VAD depends on whether
duplicates are merged, so the "frame models beat YOLO" part of the leaderboard story is partly a
decoding artefact.

Follow-up: pre-register "YOLO + NMS beats Whale-VAD on LOSO macro B on the eval-like split" and check it
on the train sites as held-out data (needs predictions on the train sites).

## 2. The frame models merge, YOLO duplicates (C2), fig 1a, 1b

Different models, different kinds of error on the same calls:
- NAVE merges BP pulses: 23 % of BP labels are "merged" (one NAVE event covers two or more labelled
  pulses), Whale-VAD 16 %, YOLO 4 %.
- Fragmentation is rare for all three (under 1 %). NAVE's 0.5 s merge gap and median filter prevent it.
- Edges: YOLO's onset and offset errors are the tightest around 0 (fig 6d), which fits C3 (box
  regression copies the annotators' box style). The frame models spread more, and NAVE ends BMABZ
  calls about 0.5 s early (median offset error -0.56 s).

## 3. Whale-VAD barely detects D, and nobody detects bpd (C10), fig 1a, 6c, 5a

- Whale-VAD D: 12 % of D labels hit, 91 % of its D detections are background. Its D outputs stay low on
  real D calls (median peak 0.1 to 0.3), so no threshold helps (F1 under 0.15 at any threshold, fig 5a).
- Split by fine label, the D class is two problems. bmd (blue whale D) is found by YOLO and NAVE at
  about 50 %, by Whale-VAD at 6 to 19 %. bpd (fin whale 40 Hz downsweep) is close to 0 % for all three
  models at both Kerguelen sites (n = 344 each).
- kerguelen2014 BMABZ is hard for everyone (bma recall 24 to 42 % against 72 to 87 % at the other two
  sites), with many "missed almost" labels for the frame models. Worth a look in the browser: box style,
  or a lower SNR at that site?

## 4. NAVE misses calls at its tile edges (C8), fig 4d, confirmed on real labels

In the position probe, NAVE's detection rate falls from about 85 % to 20 to 40 % when the call centre
sits at 58 to 60 s, exactly where one 60 s tile ends and the next starts (2 s overlap). Whale-VAD dips at
its own edges (30 s and 58 s), YOLO is flat (50 % window overlap).

On the real dev labels, NAVE's recall by margin between the call and the nearest tile edge:

| margin | call cut by an edge | 0 to 2 s | 2 to 5 s | 5 to 10 s | more than 10 s |
|---|---|---|---|---|---|
| NAVE recall | 0.29 | 0.41 | 0.59 | 0.61 | 0.55 |
| labels | 973 | 1136 | 1840 | 3016 | 10662 |

About 12 % of all labels sit within 2 s of an edge. If they were found at the rate of the others, NAVE's
recall would rise by roughly 3 points. Cheap to test without retraining: run NAVE with `--overlap-s 30`.

## 5. Z-calls have moved down in frequency, and all three models are tuned to the old frequency (C6), fig 4b

- Shift probe: BMABZ detection falls steeply for small downward shifts. Detection rate at -1 / -0.5 / 0 /
  +0.5 / +1 Hz: NAVE 8 / 60 / 90 / 80 / 50 %, Whale-VAD 2 / 55 / 85 / 70 / 42 %, YOLO 5 / 50 / 62 / 55 / 38 %.
  It is asymmetric: shifting up hurts less than shifting down. D and BP are much less sensitive. We
  checked that the shift itself is right (a -1 Hz shift moves the call's peak from 25.9 to 24.8 Hz).
- Measured Z-call peak frequency at Kerguelen (150 calls each, median): 2005 labels 27.3 Hz, 2014 26.0,
  2015 26.0, 2019 25.5, 2020 25.3 (the last two from calls all three models agree on). That is about
  0.13 Hz per year, in line with the literature.
- So the evaluation Kerguelen calls sit about 0.5 to 0.7 Hz below the dev calls, right on the steep part
  of the curves. BMABZ detections per hour at Kerguelen drop from 12 to 26 (2014/15) to 6 to 9
  (2019/20) for all three models. That fits, but fewer calls would look the same.
- Caveat: the 2019/20 numbers come from calls the models still detect, so the real shift is probably
  larger, not smaller.

Follow-up: pre-register the size of the drop and test frequency-shift augmentation (C6). This could be
the strongest single result for the paper.

## 6. Level and transients: Whale-VAD depends on the recording level, NAVE on what else is in its tile (C4, C5), fig 4a, 4e

- Level (C5): YOLO is perfectly flat (its per-window contrast removes the level). Whale-VAD's D and BP
  detection rises from about 5 to 10 % at -20 dB to 55 to 70 % at +20 dB. NAVE is flat up to +10 dB,
  then drops (BMABZ 95 % to 45 % at +20 dB). As predicted for Whale-VAD and YOLO; NAVE's drop at high
  levels was not predicted.
- Distant transient (C4): we expected YOLO to suffer most. It doesn't: -15 % at most, and only when the
  burst is inside one of the 30 s windows that contain the call. NAVE suffers most. A +30 dB burst
  anywhere in its 60 s tile costs 27 to 44 % of detections, which fits H6 (statistics over the whole
  tile change single decisions). Whale-VAD only reacts when the burst falls in the same 30 s tile.

## 7. Noise and new sites (C9), fig 3a, 4f, 2a

- SNR50 (median over the four backgrounds): NAVE -0.2 / 3.5 / 3.7 dB (BMABZ / D / BP), Whale-VAD
  2.2 / 6.5 / 6.7, YOLO 5.9 / 4.5 / 16.3. YOLO needs a much stronger signal for BP.
- Background swap: no clear penalty for DDU or 2019/20 Kerguelen backgrounds for any model. The site
  effect is not in the background noise alone, as far as this probe can show.
- Estimating precision on the new sites from model agreement does not work well. On kerguelen2014,
  the estimate from the combination rates of all three dev sites (so even in-sample) is 0.5 to 0.6 for BMABZ, the real precision 0.8 to 0.93. We
  should present those estimates only with that warning, and the blind audit
  (`outputs/phase1/audit/audit_sheet.csv`, 360 regions) becomes more important.
- D detections per hour are 2 to 4 times higher at DDU than at any dev site, for all three models.
  More D calls there, or a shared false alarm? The audit will tell.

## Not supported or unclear

- Context masking (C7): little effect for BMABZ and D. Whale-VAD BP gets worse with more context, NAVE
  D slightly better. No clear story yet.
- 34 % of dev labels are missed by all three models. Prime material for the label audit (C11).

## Pre-registered follow-ups (run 22.10.2026, see logs/preregistration.md)

**F1, dev calls moved to the 2019/20 frequency.** kerguelen2014 and kerguelen2015 shifted down by
0.6 Hz, same LOSO thresholds. BMABZ recall (both sites pooled):

| | original | shifted -0.6 Hz | change |
|---|---|---|---|
| NAVE | 0.55 | 0.19 | -36 points |
| Whale-VAD | 0.49 | 0.27 | -23 points |
| YOLO | 0.33 | 0.17 | -16 points |

D changes by less than 1.5 points for all three, BP by +1 to +5. So a drift of about four and a half
years costs every model a large part of its BMABZ recall, and the frame models more than YOLO.
Supported as pre-registered.

**F2, eval calls moved back.** kerguelen2019 and kerguelen2020 shifted up by 0.6 Hz. BMABZ detections
per hour rise by 154 % (NAVE), 430 % (Whale-VAD) and 193 % (YOLO). Without labels we cannot say how many
of the new detections are real, but together with F1 the simplest reading is that the models miss
many Z-calls at the newer Kerguelen sites because of the frequency drift. Not as pre-registered for
the other classes: NAVE also gains 94 % BP and 69 % D detections, so NAVE is frequency-sensitive for
more than BMABZ (or the shift moves noise into its bands). Whale-VAD D -10 %, YOLO BP -17 %.

**E1, NAVE with 30 s tile overlap.** Calls cut by an old tile edge are now found at 0.58 instead of 0.29,
calls within 2 s at 0.54 instead of 0.41. But calls further from an edge lose 4 to 8 points, and overall
recall falls from 0.542 to 0.523. Averaging two overlapping tiles changes the probabilities that the
thresholds were tuned for. The edge effect is confirmed, the simple fix does not help as it is. Next
to try: keep only the centre part of each tile instead of averaging, or retune the thresholds for the
new overlap.

**E2, centre stitching.** NAVE with 30 s overlap, Whale-VAD with 15 s (half its tile), each frame taken
from the tile where it lies furthest from an edge, thresholds retuned LOSO (they barely moved).

| | 2 s overlap, averaged | half-tile overlap, centre, old thresholds | same, retuned |
|---|---|---|---|
| NAVE recall, labels at an old edge | 0.36 | 0.61 | 0.65 |
| NAVE recall, all other labels | 0.567 | 0.553 | 0.574 |
| NAVE macro B, dev joined (LOSO) | 0.520 | 0.531 | 0.545 |
| Whale-VAD macro B, dev joined (LOSO) | 0.353 | 0.363 | 0.362 |

Supported as pre-registered. The tile-edge misses were a decoding problem, not a model problem: a
change to the inference alone, no retraining, gives NAVE +0.025 macro B, mostly through BMABZ
(recall 0.63 to 0.71). Whale-VAD gains little, which fits: in the real data it did not miss more
calls at its edges. Note for the paper: our 2 s numbers are what NAVE's own pipeline does, so the
published NAVE score is a little below what the model can do.

Side note: the first E2 attempt asked Whale-VAD for a 30 s overlap on 30 s tiles. That made the tiling
loop endless and took down rk10 and rk11 (fixed, the code now refuses an overlap that is not shorter
than the tile).

## More checks (22.10.2026, exploratory, no new model runs)

**Error bars.** Bootstrap over the 585 dev recordings (resampled within each site, 1000 times), LOSO
operating point, macro B joined: NAVE 0.520 [0.484, 0.550], Whale-VAD 0.353 [0.306, 0.388], YOLO 0.325
[0.297, 0.353]. NAVE minus Whale-VAD +0.167 [0.138, 0.199], NAVE minus YOLO +0.196 [0.161, 0.225]. Whale-VAD
minus YOLO +0.029 [-0.023, 0.069]: the order of these two is not settled even before merging YOLO's
duplicates. Per site the intervals are wide (casey2017 NAVE [0.34, 0.52]), so single-site differences
should not be over-read.

**Why NAVE drops at high levels.** NAVE's magnitude channel is linear like Whale-VAD's, only the PCEN
channel adapts. On 36 dev calls at +20 dB: feeding the louder audio only to the PCEN channel leaves the
peak scores unchanged (0.93 / 0.83 / 0.62 for BMABZ / D / BP), feeding it only to the magnitude and phase
channels gives the same drop as the full +20 dB (0.74 / 0.63 / 0.52). So the PCEN channel is level
invariant and the magnitude channel is the weak point. At normal levels this does not matter: the
recordings of all sites have about the same level in the call bands (5 to 120 Hz, within 2 dB).

**DDU has a large DC offset, and it does not matter.** The broadband RMS of the DDU files is about 15 dB
higher than at the dev sites, but that is a DC offset (about -27 dB against -49 to -66 dB); the AC level
is normal. Adding the same offset to dev calls changes neither NAVE's nor Whale-VAD's scores (YOLO
filters it out). Ruled out as an explanation for the DDU results.

**Level and recall per file.** Rank correlation between recording level and recall per file is
weak overall (Whale-VAD 0.28, NAVE 0.04, YOLO -0.05). Recording level does not explain the site
differences on dev.

**Loud sounds in the same tile, in real recordings.** For every dev label, the loudest other second in
its 60 s NAVE tile compared with the call. Within the same SNR band, recall falls for all three models
when something louder than the call is in the tile (high-SNR calls: NAVE 0.78 to 0.64, Whale-VAD 0.77 to
0.55, YOLO 0.52 to 0.37 when the burst is more than 5 dB above the call). NAVE loses the least, so the
strong transient effect of the probes (+30 dB bursts) does not carry over to the sounds that actually
occur in these recordings, which are rarely more than 10 dB above the call.

**Truncation (C2).** Keeping the end of a call works better than keeping its start, for all classes and
models (at 50 %: NAVE BMABZ 0.62 vs 0.48, D 0.38 vs 0.20). The evidence sits more in the second half of
the calls. YOLO needs most of the call (BP 16 % even with the full call in the pasted condition).

**Density (C7).** A BP pulse in a train of four (13 s apart) is not detected more often than alone (NAVE
0.69 vs 0.72, Whale-VAD 0.44 vs 0.41, YOLO 0.16 vs 0.16). No sign that the models use the pulse rhythm.

**Level and false alarms.** At +20 dB Whale-VAD's false alarms per clip double (1.0 to 2.3), NAVE's
halve (1.3 to 0.7, together with its lower recall). YOLO is unaffected.

## Five NAVE seeds (pre-registered S1 to S6, 22.10.2026)

Seed 666 is the released checkpoint behind every earlier result. Seeds 0, 1, 42 and 1337 had not been
looked at, so they test the NAVE findings on fresh models.

| seed | macro B, 2 s pipeline | centre stitching | gain | BMABZ recall change at -0.6 Hz |
|---|---|---|---|---|
| 0 | 0.491 | 0.510 | +0.019 | -33 points |
| 1 | 0.516 | 0.519 | +0.003 | -35 |
| 42 | 0.512 | 0.523 | +0.010 | -35 |
| 666 (released) | 0.520 | 0.545 | +0.024 | -36 |
| 1337 | 0.480 | 0.500 | +0.021 | -26 |

- Frequency drift (S2, S3), the magnitude channel as the cause of the high-level drop (S5) and the burst
  sensitivity (S6) hold for every seed. These are now confirmed results, not single-checkpoint observations.
- Scores (S1): five-seed mean 0.504 (sd 0.018). The released seed is the best of the five, so its dev score
  is about 0.017 optimistic. For the paper: report NAVE as the five-seed mean. It stays far ahead of
  Whale-VAD (0.353) and YOLO (0.325).
- Centre stitching (S4) helps every seed but by less than predicted: +0.015 over all five, +0.003 for seed 1.
  Real but smaller than seed 666 suggested.

## NAVE ablation ladder (pre-registered L1 to L4, 23.10.2026)

Rungs: A no PCEN, no FDY, kernel 31; B + FDY; C + PCEN; D + kernel 153 (= NAVE). Five seeds each.

| rung | macro B (LOSO) | probe: BMABZ detection at 0 / -0.5 / -1 Hz | BMABZ recall change at -0.6 Hz | lost at -20 dB (D class) | lost at +20 dB (BMABZ) |
|---|---|---|---|---|---|
| A | 0.337 | 0.69 / 0.34 / 0.01 | -22 points | 29 points | 2 points |
| B | 0.375 | 0.74 / 0.48 / 0.02 | -35 | 63 | 47 |
| C | 0.398 | 0.75 / 0.50 / 0.02 | -23 | 4 | 38 |
| D | 0.504 | 0.87 / 0.62 / 0.06 | -33 | 2 | 47 |

- **The frequency sensitivity comes from the training data, not from NAVE's design (L1 supported).** Even the
  plain base model (A) loses about a third of its Z-call detections at -0.5 Hz and almost all at -1 Hz.
  Every addition raises the detection rate at the training frequency, but none makes the drop smaller.
- **FDY is not the cause (L2 refuted).** On the probe, FDY makes the drop smaller, on the shifted recordings
  larger; no consistent effect. PCEN and the wide kernel change the probe drop by under 1 point (L3).
- **PCEN protects against quiet recordings (L4, for D):** without PCEN, D detection drops by 29 to 63 points
  at -20 dB, with PCEN by 2 to 4. Not for BMABZ.
- **Exploratory, not registered: the loudness problem starts with FDY.** The base model is almost blind to
  +20 dB (2 points lost); from B on, every rung loses 38 to 47 points. Together with the channel test
  (the drop comes from the magnitude channel) this suggests the frequency-dynamic stem reacts to the absolute
  magnitude scale. Worth a registered test before it goes into the paper.

## Does the FDY kernel mixing cause the loss at high levels? (pre-registered G1 to G3, 23.10.2026)

FDY mixes 4 basis kernels per frequency band with a softmax over the time-averaged input. Test on 36 dev
calls and all ladder checkpoints: +20 dB with the mixing taken from the original-level input (does the drop go
away?), and the original level with the mixing from the +20 dB input (is the mixing alone enough?).

| rung | mean peak, original | +20 dB | +20 dB, original mixing | recovered share of the drop |
|---|---|---|---|---|
| A (no FDY) | 0.365 | 0.389 | | no drop |
| B (+ FDY) | 0.465 | 0.261 | 0.453 | 96 % (4 of 5 seeds above 70 %) |
| C (+ PCEN) | 0.458 | 0.223 | 0.380 | 67 % |
| D (NAVE) | 0.838 | 0.620 | 0.385 | none, the swap makes it worse |

- Without FDY there is no level effect (G3). In the simple FDY model the mixing is the whole story: giving the
  loud input the original mixing restores the scores (B).
- In the full NAVE it is not: there the swap hurts instead of helping, so the loss at +20 dB comes from
  elsewhere in the model (the magnitude channel reaching the later layers). G1 refuted for NAVE.
- Swapping in the loud mixing at the original level always hurts, more than the full +20 dB (G2). That shows
  the mixing and the input are tuned to each other; it is not proof of the cause on its own.
- For the paper: "FDY makes NAVE level-sensitive" is not supported. What holds: the plain model is level-robust,
  every FDY model is not, and in the simplest FDY model the mixing explains it.

## Where inside NAVE the level loss arises (pre-registered H1, H2, 23.10.2026)

36 dev calls, 5 NAVE seeds, +20 dB. At 15 sites: how much louder the activations are, whether they are only a
scaled copy (shape similarity, 1 = only the scale changed), and how much of the drop is undone by rescaling
them channel by channel to the original scale.

| site | RMS ratio | shape similarity | recovered share of the drop |
|---|---|---|---|
| first FDY layer | 2.35 | 0.64 | 55 % |
| stem, later layers | 1.0 to 1.9 | 0.65 to 0.87 | negative (rescaling makes it worse) |
| projection into the Conformer | 2.38 | 0.55 | 14 % |
| Conformer block 1 | 0.96 | 0.88 | 53 % |
| Conformer blocks 2 to 4 | 1.00 | 0.92 to 0.94 | 18 to 25 % |

- The level loss is not a simple scale problem at any one point. Already after the first layer the louder
  input gives a different representation (shape similarity 0.64), not just a louder one: the first layer mixes
  the magnitude channel, which grows by 20 dB, with phase and PCEN channels, which do not. Per-channel
  rescaling afterwards cannot undo such a mix. At most about half of the drop can be undone at any site.
- So the weak point is the input itself (the linear magnitude channel, as the channel test showed), not a
  particular layer. The obvious fix would be to normalize the magnitude channel per tile. Since all real
  sites have about the same level in the call bands, this matters little in practice.
- For the paper: one sentence. "NAVE's magnitude channel makes it sensitive to large level changes; the PCEN
  channel is level-invariant; recording levels at the BioDCASE sites are similar enough that this does not
  affect the results."

## Challenge leaderboard after the organisers' label fix (22.10.2026)

The organisers rescored all submissions after fixing an error in the Raven to BioDCASE annotation
conversion of the evaluation labels. Numbers from https://biodcase.github.io/challenge2026/task2-results
(before: release of 06.07.2026, after: update of 22.10.2026, gh-pages history of biodcase.github.io).

| system | F (%) before -> after | precision | recall | rank |
|---|---|---|---|---|
| Nagl_JKU_NAVE | 36.1 -> 39.6 (+3.5) | 35.9 -> 41.9 | 36.4 -> 37.4 | 21 -> 22 |
| Nagl_JKU (Whale-VAD ensemble) | 39.3 -> 40.6 (+1.3) | 54.4 -> 61.8 | 30.7 -> 30.3 | 16 -> 20 |
| yolo_baseline | 37.6 -> 38.2 (+0.6) | 38.4 -> 39.7 | 36.8 -> 36.8 | 19 -> 24 |
| Guan_GISP_HEU_1 (winner) | 49.9 -> 54.9 | 47.5 -> 56.5 | 52.5 -> 53.3 | 1 |

- NAVE gained the most and is now above the YOLO baseline (it was 1.5 points below). The gain is mostly
  precision (+6 points), so part of its former false alarms were real calls with wrongly converted label
  boxes. The gap to the ensemble shrank from 3.2 to 1.0 points.
- Ranks fell although every score rose, because other teams gained more.
- The per-site table (DDU2021, Kerguelen2020, DDU2019, Kerguelen2019) was not rescored: it shows the July
  values for every team. Do not use it until it is updated.
- On the evaluation sites NAVE, the ensemble and YOLO are within 2.4 points; on the dev sites NAVE leads by
  about 0.2 macro B. Fits the drift and new-site results (sections 5 and 7).

## Frequency-shift augmentation (pre-registered A1 to A4, 24.10.2026)

NAVE retrained with single-sideband shifts of -1.5 to +1.5 Hz on 80 % of the training segments, 5 seeds,
compared with the 5 unaugmented NAVE seeds (rung D).

| | NAVE | NAVE + shift augmentation |
|---|---|---|
| BMABZ recall change, Kerguelen 2014/15 at -0.6 Hz (A1) | -33.1 points | -5.8 points (all seeds -4.4 to -7.9) |
| LOSO macro B, dev joined (A2) | 0.504 (sd 0.018) | 0.483 (sd 0.007) |
| BMABZ detections per hour, Kerguelen 2019/20 (A3) | 7.1 | 17.0 (+140 %) |
| shift probe, BMABZ detection at 0 / -0.5 / -1 Hz (A4) | 0.87 / 0.62 / 0.06 | 0.97 / 0.84 / 0.69 |

- The drift sensitivity is mostly trained away (A1 supported for every seed): the frequency tuning of
  L1 came from the training data, and data is also the fix.
- On the new Kerguelen years the augmented models fire 2.4 times as often for Z-calls (A3); BP +23 %,
  D -4 %. Without labels this is a rate, not recall; the blind audit has to say how many are real.
- Cost at the dev frequency: -0.021 macro B on average (A2, just outside the expected 0.02, not refuted),
  with less spread between seeds.
- The shift curve is much flatter but not flat (A4 partly supported): 28 points lost at -1 Hz instead of 81.
  Detection at the training frequency goes up (0.87 to 0.97), and the curve now also reaches +2.5 Hz.
- Not changed: the loss at +20 dB (35 to 41 points for every class) is the magnitude-channel problem of
  H1/H2, which augmenting the frequency does not touch (M1 to M4 test that).

## NAVE without the linear magnitude channel (pre-registered M1 to M4, 25.10.2026, 5 seeds)

NOMAG: cos, sin of the phase and PCEN. LOGMAG: log magnitude minus its mean over the tile per frequency
bin, in place of the linear magnitude. Both with the full NAVE recipe, 5 seeds each, against NAVE (rung D).

| | NAVE | NOMAG | LOGMAG |
|---|---|---|---|
| LOSO macro B, dev joined (M2, M3) | 0.504 | 0.473 | 0.466 |
| detection lost at +20 dB, BMABZ / BP / D (M1) | 47 / 24 / 42 | 0 / 1 / -1 | 4 / 4 / 2 |
| detection lost at -20 dB, BMABZ / BP / D (M1) | 15 / -9 / 2 | 5 / -3 / -3 | -1 / 0 / -1 |
| BMABZ recall change, Kerguelen 2014/15 at -0.6 Hz | -33 | -21 | -19 |
| shift probe, BMABZ drop at -0.5 / -1 Hz (M4) | 26 / 82 | 18 / 80 | 17 / 84 |

- Removing (or normalising) the linear magnitude makes NAVE level-robust (M1 supported for both): the
  loss at +20 dB goes from 47 points to almost nothing. This closes the line C5, G1 to G3, H1, H2: the
  level weakness was the linear magnitude channel, and the fix is in the input, not in the network.
- It costs some accuracy at the dev level: NOMAG -0.031, LOGMAG -0.038 macro B (M2, M3 not supported,
  not refuted). The phase and PCEN carry most, not all, of what the magnitude gave.
- The frequency tuning stays (M4 supported): at -1 Hz both lose as much as NAVE, so the drift
  sensitivity does not come from the input representation (consistent with L1 and A1).
- Exploratory: on the real shifted Kerguelen audio both lose less than NAVE (-19 to -21 instead of -33,
  in every seed), and LOGMAG keeps more calls shifted upwards (0.55 at +2 Hz against 0.25). Small shifts
  hurt less without the linear magnitude, large ones just as much. Not pre-registered.
- The real sites differ by less than 2 dB in the call bands, so the level weakness hardly matters for
  the scores here; it matters for recorders with other gains. Augmentation (A1 to A4) and a different
  magnitude channel fix two different weaknesses and could be combined.

## Ensembles of the three models (pre-registered EN1 to EN4, 25.10.2026)

All with LOSO thresholds, compared with NAVE alone through the same code (macro B 0.520, as in Phase I).

| | macro B | mean P | mean R | F1 BMABZ |
|---|---|---|---|---|
| NAVE alone | 0.520 | 0.542 | 0.500 | 0.518 |
| best model per class, chosen LOSO (EN2) | 0.515 | 0.538 | 0.494 | 0.503 |
| at least 2 of 3 models (EN3) | 0.502 | 0.635 | 0.415 | 0.570 |
| NAVE confirmed by another model | 0.504 | 0.659 | 0.408 | 0.559 |
| all 3 models | 0.279 | 0.730 | 0.172 | 0.394 |
| exploratory: 2 of 3 for BMABZ, NAVE for D and BP | 0.527 | 0.588 | 0.478 | 0.570 |
| weighted score fusion, weights and threshold LOSO per class (EN1) | 0.525 | 0.557 | 0.497 | 0.570 |
| YOLO as a score track (highest box confidence per frame) | 0.439 | 0.641 | 0.333 | 0.527 |

- Routing gains nothing (EN2 supported): NAVE is the best model in 8 of 9 class-folds.
- Voting buys precision with more recall than it gains (EN3 supported): +0.09 mean precision, -0.02 macro B.
- Exploratory (chosen after seeing EN3): for Z-calls the models complement each other. Voting only for
  BMABZ gives macro B 0.527 (+0.007). Choosing per class and fold between NAVE and the voting sets on the
  tuning sites picks voting for BMABZ in all three folds, with the same result, so the choice itself is
  not overfitted; the idea is. Per site the BMABZ gain comes from precision: casey2017 F1 0.527 to 0.638,
  kerguelen2015 0.522 to 0.591, kerguelen2014 0.501 to 0.480. NAVE's BMABZ false alarms are mostly its own,
  the other two models do not share them. Small gain, three sites; it needs the eval sites or a new
  pre-registration before it counts.
- Weighted score fusion (EN1, EN4): 0.525, +0.006 over NAVE, not supported (expected +0.02). The whole
  gain is again BMABZ (0.518 to 0.570), D gets a little worse. The tuning gives YOLO the largest weight for
  Z-calls in every fold (0.5 against 0.25 / 0.25), so EN4 is refuted: the Z-call complement comes from
  YOLO. Two different methods, voting and fusion, find the same thing: the three models only complement
  each other for Z-calls, and only through precision; a strong single model leaves little to gain.
- Side result: YOLO read as a per-frame score track (which merges its duplicate boxes) reaches 0.439 on
  its own, against 0.325 for its boxes and 0.41 with temporal NMS (C1).
- Bug found on the way: the first fusion run mixed float32 and float64 thresholds, so empty filler rows
  were added to the real counts and NAVE alone came out at 0.342 instead of 0.520. Fixed in
  aivis.phase1.ensemble (clean_counts); with the fix every single model reproduces its Phase I score.


## Shared misses across seeds and models, seed agreement as confidence (25.10.2026, exploratory)

aivis.phase1.seed_agree: every NAVE seed at its own LOSO thresholds (2 s pipeline), YOLO and Whale-VAD from
Phase I (LOSO). The table column "missed_by_all_8_models" of this run means the 7 distinct models (5 NAVE
seeds, YOLO, Whale-VAD); renamed in the code.

| | BMABZ | D | BP | all |
|---|---|---|---|---|
| labels missed by all 5 NAVE seeds | 26.9 % | 51.7 % | 45.1 % | 36.4 % |
| the same if the seeds missed independently | 0.9 % | 8.5 % | 5.5 % | 2.5 % |
| of these, also missed by YOLO and Whale-VAD | 76.8 % | 83.1 % | 83.8 % | 80.9 % |
| YOLO and Whale-VAD both miss, over all labels | 35.4 % | 58.5 % | 55.4 % | 45.1 % |
| YOLO and Whale-VAD both miss, when all 5 seeds hit | 8.9 % | 20.4 % | 18.3 % | 12.6 % |
| missed by all 7 models | 20.6 % | 42.9 % | 37.8 % | 29.4 % |
| median SNR of the 5-seed misses (all labels) | 2.4 (3.8) dB | 2.2 (2.9) dB | 3.9 (5.2) dB | 2.8 (4.1) dB |

- The seeds fail on the same calls (about 15 times more often than chance), and these are mostly the calls
  the other two architectures miss as well: 81 % against 45 % over all labels. 29 % of all dev labels are
  missed by all seven models. This is the strongest pointer so far that the limit is in the data and the
  labels, not in one model (C11). These labels are fainter (median SNR 2.8 against 4.1 dB), but most are not
  below 0 dB, so faintness alone does not explain them: the listening check takes its sample from here.
- Agreement between seeds orders the events well: precision of seed 666's events rises from 0.07 (no other
  seed agrees) to 0.55 (all four) for BMABZ, 0.10 to 0.72 for BP, 0.20 to 0.69 for D.
- But it does not transfer between sites as a precision estimate: learnt on two dev sites, it estimates the
  third with a mean absolute error of 0.15 and up to 0.41 (kerguelen2014 BMABZ: estimated 0.29, real 0.70),
  no better than agreement between the three models in Phase I. The base precision differs between sites,
  so agreement ranks events within a site but cannot replace labels on a new one.

## Event decoding without retraining (pre-registered D1 to D9, 30.10.2026, 5 seeds)

aivis.phase1.decode, tables decode_*_v2. Every decoder parameter is tuned per class on two dev sites and
applied to the third, like the thresholds. Centre stitching, LOSO macro B (dev joined), mean (sd) over 5 seeds:

| decoder | macro B | gain over base | seeds with a gain |
|---|---|---|---|
| base (published: 0.5 s median filter, merge 0.5 s, keep 0.5 to 30 s) | 0.519 (0.017) | | |
| caps (per-class merge gap, min and max duration) | 0.543 (0.011) | +0.024 (0.011) | 5 of 5 |
| smooth (caps plus per-class median filter length) | 0.544 (0.010) | +0.025 (0.009) | 5 of 5 |
| csebb (change-point sound event bounding boxes) | 0.516 (0.018) | -0.003 (0.010) | 2 of 5 |
| csebb_caps | 0.536 (0.017) | +0.017 (0.010) | 5 of 5 |
| hyst (duration-aware hysteresis) | 0.544 (0.010) | +0.024 (0.011) | 5 of 5 |
| route (best of smooth, csebb_caps, hyst per class) | 0.546 (0.011) | +0.026 (0.012) | 5 of 5 |

- Per-class duration limits are the whole gain. Everything more elaborate lands within 0.003 of caps,
  so by the rule fixed before the run, caps is the decoder to adopt. With centre stitching NAVE goes from
  0.504 (published 2 s pipeline) to 0.543 without retraining, and the seed spread shrinks (sd 0.018 to 0.011).
- The chosen limits are stable: BMABZ at least 2 s (30 of 30 folds), D 2 to 15 s, BP at most 8 s
  (30 of 30). The gain is mostly precision (BMABZ 0.475 to 0.524, D 0.520 to 0.594).
- cSEBB does not help (D1 refuted). It finds more of the long bmz calls but loses D.
- Per call type: the D gain comes from blue whale D-calls (bmd recall 0.504 to 0.577). Fin whale 40 Hz
  downsweeps (bpd) are found at 7 % even with the published decoding, and no decoder rescues them
  (hysteresis 5 %). bp20 is found at 0.32 against 0.57 for bp20plus. These are model limits, not decoding
  ones: the next lever for them is training (separate bmd and bpd outputs, or more bpd data), not decoding.
- The gain holds at every held-out site but is small at kerguelen2014 (+0.009), the site with most fin
  whale calls.

## The next NAVE pipeline: augmented model, centre stitching, caps (pre-registered R1 to R3, 30.10.2026)

aivis.phase1.decode run --models nave AUG --decoders base caps, tables decode_*_recipe. LOSO macro B (dev joined),
mean (sd) over 5 seeds:

| | 2 s, base (published) | centre, caps |
|---|---|---|
| NAVE | 0.504 (0.018) | 0.543 (0.011) |
| NAVE + frequency-shift augmentation (AUG) | 0.483 (0.007) | **0.551 (0.004)** |

- The augmentation's dev cost (A2, -0.021) was a decoding cost: caps gains twice as much for AUG as for NAVE
  (+0.048 against +0.024), mostly by removing short D false alarms (AUG D precision 0.480 to 0.596). With
  centre stitching and caps, AUG is level with NAVE (+0.008 (0.014), within seed noise) and far more stable
  across seeds (sd 0.004 against 0.011).
- The drift robustness survives: Z-call recall at -0.6 Hz drops by 5.5 points with AUG + caps (every seed
  under 8), against 30.6 for NAVE + caps.
- The gain holds at all three held-out sites (AUG centre caps over its base: +0.083 / +0.030 / +0.035).
- Unchanged: fin whale downsweeps (bpd) stay at about 5 % recall. That needs training, not decoding.
- Recipe for the next submission, no new training needed: AUG checkpoints, 60 s tiles with 30 s overlap and
  centre stitching, caps decoding (per class: BMABZ min 2 s, max 30 s; D min 2 s, max 15 s, merge gap 0.5 s;
  BP max 8 s, merge gap 0.5 s; thresholds tuned on the dev sites). On dev that is +0.047 over the published
  NAVE pipeline. Dev sites are 2014 to 2017; the eval sites are newer, which is where the augmentation should
  matter most, and that cannot be scored without eval labels.
