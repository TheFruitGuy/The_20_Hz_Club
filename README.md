# The 20 Hz Club

Group project for JKU 364.036 AI and Visualization, WS 2026.

## Team

| name | Phase I |
|---|---|
| Matthias Nagl | AI-aided |
| Klaus Lex | AI-aided |
| Abhirup Mitra | human-only |
| Gaurav Parajuli | human-only |

The pairs switch in Phase II.

## Project

- **Task:** [BioDCASE 2026 Task 2](https://biodcase.github.io/challenge2026/task2), detecting Antarctic
  blue and fin whale calls in long underwater recordings (250 Hz). Three classes: BMABZ (blue whale A, B
  and Z-calls), D (D-calls and 40 Hz downsweeps) and BP (fin whale 20 Hz pulses).
- **Models:** the YOLO baseline of the task (object detection on spectrogram images), Whale-VAD and NAVE
  (both give a probability for every 20 ms frame).
- **Dataset:** the task's dev set (train and validation sites with expert labels) and eval audio (four
  new sites from 2019/20, no public labels), both from Zenodo.
- **Phase I question:** how do the three detectors behave beyond one leaderboard score? Which calls do
  they miss, how do they react to pitch, loudness and faint calls, and how far can we trust the labels?

## Phase I results

Figures of the midpoint slides are in [`results/phase1/slides/`](results/phase1/slides/), made with
`python -m aivis.phase1.slidefigs` from the Phase I tables. Every test, with
what we expected before running it, is in [`logs/preregistration.md`](logs/preregistration.md).

Human-only pair:

- The five NAVE seeds miss the same calls: 27 % of the blue whale calls are missed by all five, against
  0.9 % if the seeds failed independently.
- NAVE only listens in a narrow pitch band. The Z-calls got lower over the years, so at the 2019/20 pitch
  it finds 19 of 100 known calls instead of 55. Shifting the 2019/20 audio up gives 2.5 times more
  detections per hour.

AI-aided pair:

- Faint calls never reach the threshold. A typical call only gets over the model's own threshold above
  about 2 dB SNR for NAVE, 4 dB for Whale-VAD and 10 dB for YOLO.
- Each model goes wrong in its own way: YOLO finds calls twice and misses others, Whale-VAD adds short
  noise detections, NAVE merges calls that are close together.
- Half of the 2019/20 Z-calls at Kerguelen fall below the pitch band where NAVE listens.
- Loudness changes what each model finds: at +20 dB Whale-VAD gives more than twice as many false alarms
  and NAVE finds fewer calls, YOLO hardly changes.

## Repo

```
aivis/              our code: running the models, scoring, the Phase I analyses (aivis/phase1/)
notebooks/          01 first look at data and models, 02 Phase I tables, 03 listening check
logs/               preregistration.md (tests, written before each run), decisions.md
results/phase1/     slide figures (aivis/phase1/slidefigs.py)
ai-interactions/    record of the AI-aided pair's GenAI use
config/ scripts/    pinned versions and the setup script
```

Midpoint files: [`ai_reflection_midpoint.md`](ai_reflection_midpoint.md),
[`ai-interactions/`](ai-interactions/), [`team-info.json`](team-info.json), [`workload.json`](workload.json).

## Setup

Needs conda and git, works on Windows, macOS and Linux.

```bash
git clone https://github.com/TheFruitGuy/The_20_Hz_Club.git
cd The_20_Hz_Club
conda env create -f environment.yml
conda activate aivis
python scripts/aivis_setup.py all
```

The setup clones the task repo, Whale-VAD and NAVE-SED into `external/` (commits pinned in
`config/external.toml`), installs the packages, downloads the data from Zenodo (about 7 GB) and the
weights, and runs all three models on a short synthetic recording as a test. It can be run again and only
does what is missing. Useful options: `--data-dir PATH` to keep the data on another disk, `--torch cpu`
without a GPU. `python scripts/aivis_setup.py check` shows what is in place.

## Running

```bash
python -m aivis.predict nave --split validation        # also yolo, whalevad; --split evaluation
python -m aivis.evaluate outputs/nave/validation       # official leaderboard scorer
python -m aivis.tune outputs/nave/validation           # thresholds, in-sample and leave-one-site-out
python -m aivis.phase1.run_all                         # everything for Phase I, then notebooks/02_phase1.ipynb
```

Outputs go to `outputs/` and are not in git. Not in this repo either: our own NAVE checkpoints (seeds and
variants) and the audio of the listening check. The released NAVE checkpoint comes with the setup.

## Data

[Dev set](https://zenodo.org/records/18832958) and [eval set](https://zenodo.org/records/20485251) on
Zenodo, both CC BY 4.0. The YOLO baseline and the scorer come from the task repo, Whale-VAD and NAVE-SED
from their own repos.
