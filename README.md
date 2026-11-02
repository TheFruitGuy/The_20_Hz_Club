# The 20 Hz Club: three whale call detectors under the microscope

Group project for JKU 364.036 AI and Visualization (WS 2026).

| | |
|---|---|
| **Team** | The 20 Hz Club: Matthias Nagl, Klaus Lex, Abhirup Mitra, Gaurav Parajuli |
| **Phase I pairs** | AI-aided: Matthias Nagl, Klaus Lex. Human-only: Abhirup Mitra, Gaurav Parajuli (roles switch in Phase II) |
| **Task** | Sound event detection of Antarctic blue and fin whale calls in long 250 Hz recordings ([BioDCASE 2026 Task 2](https://biodcase.github.io/challenge2026/task2)), three classes: BMABZ (blue whale A, B and Z-calls), D (D-calls and 40 Hz downsweeps), BP (fin whale 20 Hz pulses) |
| **Models** | YOLO baseline of the task (spectrogram object detection), Whale-VAD (frame model), NAVE (frame model; the released checkpoint plus our own seeds and variants) |
| **Dataset** | BioDCASE 2026 Task 2 dev set (train and validation sites, expert labels) and eval audio (4 new sites, 2019/20, no public labels), from Zenodo, plus synthetic probes made from it (shifted, louder, cut and moved calls) |
| **Phase I question** | How do the three detectors behave beyond one score: which calls they miss, how they react to frequency drift, loudness and tile edges, and how far the labels can be trusted |

Midpoint files: [`ai_reflection_midpoint.md`](ai_reflection_midpoint.md), [`ai-interactions/`](ai-interactions/),
[`team-info.json`](team-info.json), [`workload.json`](workload.json). Phase I findings:
[`results/phase1/findings.md`](results/phase1/findings.md), slide figures in `results/phase1/slides/`
(`python -m aivis.phase1.slidefigs`), report figures in `results/phase1/report/`.

Not in this repo: the private NAVE checkpoints (five seeds, ablation ladder, augmented and input variants,
`checkpoints/`) and the audio pack of the listening check (`results/phase1/listening/`). The released NAVE
checkpoint is downloaded by the setup; the others are available on request.

This repo sets up the data, the three models and the official scorer the same way for all of us,
so our outputs and scores are comparable.

## Setup

You need [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda, Anaconda or Miniforge) and git.
The same commands work on Windows (Anaconda Prompt or PowerShell), macOS and Linux:

```bash
git clone https://github.com/TheFruitGuy/The_20_Hz_Club.git
cd The_20_Hz_Club
conda env create -f environment.yml
conda activate aivis
python scripts/aivis_setup.py all
```

After that, `conda activate aivis` is all you need. The setup can be run again at any time, it only
does what is still missing:

1. clones the task repo (YOLO baseline and scorer), Whale-VAD and NAVE-SED into `external/`, at the commits pinned in `config/external.toml`
2. installs the packages with pip (PyTorch 2.8, ultralytics 8.3.91, pandas 2, ...) and registers the Jupyter kernel "Python (aivis)"
3. downloads the dev set (5.6 GB) and the eval audio (1.1 GB) from Zenodo, checks the md5 and unpacks them into `data/`
4. puts the weights into `weights/` (YOLO baseline from Zenodo, Whale-VAD from its GitHub release, NAVE from its repo)
5. shows what is in place and runs all three models on a synthetic recording (about a minute on CPU)

You need about 10 GB of free space for the data and 1 to 5 GB for the packages (the CUDA build of
PyTorch is the big part).

| option | when |
|---|---|
| `--data-dir D:\biodcase` | keep the data somewhere else (bigger disk, shared folder); remembered in `config/local.toml` |
| `--dev-zip FILE`, `--eval-zip FILE` | you already downloaded the zips |
| `--no-data`, `--skip-eval` | skip all downloads, or only the eval set |
| `--keep-zips` | keep the zips after unpacking |
| `--torch cpu` / `--torch cuda` | force the PyTorch build. Default: CUDA on Linux and on Windows with an NVIDIA GPU, otherwise CPU (MPS on a Mac) |
| `--with-deepvoice` | also the DeepVoice code, with its own conda environment `aivis-deepvoice` |

If the unpacked dev set is already somewhere (on a server, say), point `--data-dir` to the folder that
contains `dev/` (and `eval/`), and the download is skipped. The environment variable `AIVIS_DATA`
overrides `config/local.toml`.

## Running the models

```bash
python -m aivis.predict yolo --split validation
python -m aivis.predict whalevad --split validation
python -m aivis.predict nave --split validation
python -m aivis.predict nave --split evaluation                                        # the 4 eval sites
python -m aivis.predict whalevad --split validation --sites casey2017 --limit-files 2  # quick try
```

The device is picked automatically (CUDA if there is a GPU, YOLO also uses MPS on a Mac), or set it
with `--device cpu|cuda|mps`. Each run writes a folder `outputs/<model>/<split>[_<tag>]/`:

- `detections.csv`: one row per detected call, with the columns the scorer needs first
- `frame_probs/`: Whale-VAD and NAVE, frame probabilities (20 ms) per recording
- `boxes_raw.parquet`: YOLO, every box of every window before the conversion to times
- `run.json`: settings, thresholds, weights hash, commits and package versions

and adds a line to `logs/runs.csv`. Running the same thing again needs `--tag NAME` (or `--overwrite`).

What each model does by default:

| | YOLO baseline | Whale-VAD | NAVE |
|---|---|---|---|
| input | 30 s spectrogram images every 15 s | 30 s tiles, 2 s overlap | 60 s tiles, 2 s overlap |
| thresholds | confidence 0.1 | 0.5 per class (none published) | from the checkpoint, tuned in-sample on the dev sites |
| post-processing | none, as in the baseline | NAVE's | NAVE's (500 ms median, merge gaps up to 0.5 s, keep 0.5 to 30 s) |

More detail and the known quirks of each model are in [docs/models.md](docs/models.md).

## Scoring

```bash
python -m aivis.evaluate outputs/nave/validation
```

This uses the official leaderboard scorer from the task repo and prints precision, recall and F1 per
site and class, plus macro A (mean of the class F1s) and macro B (F1 of the mean precision and recall,
what the leaderboard ranks by). It writes `metrics.csv` and `metrics.json` into the run folder.

Thresholds for the frame models, tuned in-sample and leave-one-site-out:

```bash
python -m aivis.tune outputs/whalevad/validation
python -m aivis.evaluate outputs/whalevad/validation/detections_loso.csv
```

## In a notebook

```python
from aivis import io, paths
det = io.read_detections(paths.OUTPUTS / "nave/validation/detections.csv")
probs = io.load_frame_probs(paths.OUTPUTS / "nave/validation/frame_probs/casey2017/<file>.npz")["probs"]
```

`jupyter lab` works from the activated environment (or pick the kernel "Python (aivis)" in VS Code). Start with `notebooks/01_first_look.ipynb`: it checks the setup, plots the label statistics, runs the three models on a few dev recordings and shows every model's output on one timeline.

## Phase I

`caffeinate -i python -m aivis.phase1.run_all` runs all three models on every validation and evaluation
recording and builds the tables for the Phase I analyses (error anatomy, thresholds and ranking,
agreement on the new sites, probes). Then open `notebooks/02_phase1.ipynb`. Details in
[docs/phase1.md](docs/phase1.md).

## What is where

```
config/external.toml   pinned repos, data and weights (versions are changed here and nowhere else)
environment.yml        the conda environment (Python 3.11)
scripts/aivis_setup.py the setup: install, clone, data, weights and check steps, or all at once
aivis/                 our code: predict, evaluate, tune, smoke_test, formats
docs/                  models.md (what each model does), conventions.md (formats, how we log runs)
logs/                  runs.csv (every run), decisions.md, preregistration.md
ai-interactions/       record of the AI-aided pair's GenAI use (Phase I)
results/               small result files: Phase I findings, report and slide figures
external/ data/ weights/ outputs/   made by setup and by runs, not in git
```

## How we work

- Nothing gets changed inside `external/`. If a model needs a change, it goes into `aivis/` (or into a fork of that repo with a new pin) and into `logs/decisions.md`.
- Before an experiment, the expectation goes into `logs/preregistration.md` and gets committed.
- AI use goes into `ai-interactions/`, we need it for the course report.
- One branch per person or pair, merged by pull request. No data or outputs in git; small result tables can go into `results/`.

Details in [docs/conventions.md](docs/conventions.md).

## Data and licences

- Data: [dev set](https://zenodo.org/records/18832958) and [eval set](https://zenodo.org/records/20485251) on Zenodo, both CC BY 4.0 (cite the task and Miller et al. 2021). Dev labels come from one expert per site-year; the eval labels are not public, the organisers re-annotated that data in a multi-annotator campaign.
- Code we pull in: the task repo with the YOLO baseline (built on ultralytics, AGPL-3.0), Whale-VAD (GPL-3.0), NAVE-SED, and optionally DeepVoice (MIT).
- This repo has no licence yet. Our wrappers import Whale-VAD and ultralytics, so GPL-3.0 would be the easy choice if we publish it.

## Troubleshooting

- Zenodo slow or blocked: download the zips in the browser and use `--dev-zip` / `--eval-zip`, or point `AIVIS_ZENODO_BASE` to a mirror.
- Not enough space: `--data-dir` on a bigger disk. The zips are deleted after unpacking unless `--keep-zips`.
- No GPU on Linux and PyTorch too big: `--torch cpu`.
- Windows: `python scripts/aivis_setup.py check` shows `cuda=False` although there is an NVIDIA GPU: run `python scripts/aivis_setup.py install --torch cuda`. Downloads use `curl.exe` (part of Windows 10 and 11), otherwise Python itself.
- Windows with very long folder names: clone the repo into a short path such as `C:\aivis` if git or unzipping complains.
- Intel Mac: PyTorch 2.8 has no builds for Intel Macs any more, use a Linux machine or one of the JKU servers.
- `aivis.tune` on all dev files takes a while (NAVE's grid search); give it more processes with `--workers`.
- See what is in place at any time: `python scripts/aivis_setup.py check`
