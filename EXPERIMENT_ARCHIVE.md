# Experiment code archive

Prepared on 2026-09-16. This repository collects the available SCAR experiment
code in the current project and the earlier SCARCE working directory referenced
by the consolidated experiment report.

## Included code

| Location | Contents |
| --- | --- |
| `train.py`, `models/`, `ratio_estimation/`, `datasets/` | Current conditional density-ratio estimators and synthetic data implementation |
| `experiments/`, `scripts/`, `configs/`, `src/` | Architecture comparisons, next-round experiments, Phase B0–B4, diagnostics and analyses |
| `wisteria/` | Current cluster job templates and submission scripts |
| `tests/` | Current experiment and estimator tests |
| `reports/`, `ABLATION_RESULTS.md` | Available written experiment results and report-building scripts |
| `legacy/SCARCE/` | Earlier original SCARCE, domain-shift SCARCE, pooled SCAR, DIW and ADIW code, tests, documentation and cluster scripts |

The current project's six existing Git commits are retained. The legacy folder
is a snapshot of its working files, including previously uncommitted experiment
code; it is not a merge of that project's Git history. Its original README is
retained for provenance and usage.

## IWPU reproduction

Added on 2026-09-16 in `legacy/IWPU/`: an independent from-paper IWPU
reimplementation, including source, configs, runners, tests and cluster scripts.
See its README for setup and its reproduction statement for assumptions.
The original working directory is preserved; datasets and raw outputs are excluded.

## Running experiments

Use the root README for current experiments. The legacy project has its own
imports and dependencies: run it from `legacy/SCARCE/` and follow its README,
COLAB.md and WISTERIA.md. Do not run all legacy and current tests together from
the root because their top-level module names overlap.

Cluster account, environment modules and filesystem paths must be adapted to the
execution environment. Historical reports and report-generation scripts retain
their original local source paths; those links may not resolve on GitHub.

## Scope and exclusions

This is a source-code archive, not a full data or trained-model backup. Downloaded
datasets, checkpoints, caches, runtime logs and raw output directories are not
included. Python dataset source files are included. Written reports are retained,
but their underlying raw results may need to be copied separately to regenerate
every table. Other unrelated local projects are outside this archive.

Packaging checks cover file completeness, Git integrity and common credential
patterns. Experiments were not rerun as part of this upload.
