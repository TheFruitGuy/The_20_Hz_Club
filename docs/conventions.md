# Conventions

Short version: every run goes through `aivis.predict`, gets its own folder and a line in
`logs/runs.csv`; every decision and every AI use gets a line in `logs/`; expectations are written
down before we look at results.

## Output folder of a run

`outputs/<model>/<split or site>[_<tag>]/`

| file | content |
|---|---|
| `detections.csv` | `dataset, filename, annotation, start_datetime, end_datetime` (what the scorer reads), then `start_s, end_s` (seconds from the file start), `confidence`, `low_frequency, high_frequency` (YOLO only). Classes are `bmabz`, `d`, `bp`. Times are UTC ISO 8601. |
| `frame_probs/<site>/<file>.npz` | Whale-VAD and NAVE. `probs` (frames x 3, float16, order bmabz, d, bp), `classes`, `hop_s` (0.02), `frame_len_s` (1.024); Whale-VAD also `probs7` (all 7 raw outputs). Frame f starts at f * 0.02 s. |
| `boxes_raw.parquet` | YOLO. `dataset, filename, window, cls, name, conf, x, y, w, h`, box in normalised image coordinates, window index i means the window starts at i * 30 s. |
| `run.json` | the command line options, thresholds and where they came from, post-processing settings, weights file and sha256, commits of this repo and of the external repos, package versions, host, user, run time. |
| `metrics*.csv`, `metrics*.json` | written by `aivis.evaluate` |
| `thresholds.json`, `detections_insample.csv`, `detections_loso.csv` | written by `aivis.tune` |

Outputs stay out of git (they are big and can be recomputed). If a result goes into a report, copy the
small table into `results/` and commit it with a line in `logs/decisions.md` saying which run it came from.

## Naming runs

`--tag` is the name of an experiment variant, short and the same for all models in one comparison, for
example `seed1`, `loso`, `level+10db`, `M2_nococo`. The plan's variant names (M1 to M5 for YOLO,
S1 to S7 for the frame models) are good tags.

## logs/

- `runs.csv`: one line per run, written automatically. Commit it with your other changes; on a merge
  conflict keep both sides' lines.
- `decisions.md`: date, what we decided, why, who. Anything that changes how numbers are produced belongs
  here (a new pin, a different threshold source, a bug fix in a wrapper).
- `preregistration.md`: before an experiment, what we expect, what would prove us wrong and which view
  or test decides. Commit before running; anything added later is marked as exploratory.
- `ai-interactions/` (repo root, formerly `logs/ai_log.md`): needed for the AI4Vis part of the course. Important prompts, what we accepted, changed
  or rejected and why, time spent or saved, and failure patterns we noticed.

## Code

- Nothing gets edited inside `external/`. Changes go into `aivis/` or into a fork with a new pin in
  `config/external.toml`.
- New analysis code: a module in `aivis/` if others will use it, a notebook in `notebooks/` if not.
  Notebooks read from `outputs/` through `aivis.paths` and `aivis.io`, never from hardcoded paths.
- Retrained models: 3 seeds at least, the seed in the tag.
- Figures for the report and the paper: made by a script from `outputs/`, saved as PDF or SVG, colours
  fixed (YOLO blue `#1F6FB2`, Whale-VAD orange `#D9822B`, NAVE purple `#7B3F9E`).
