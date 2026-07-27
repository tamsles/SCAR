# Phase B repository audit and code-path map

Audit date: 2026-07-27. Baseline commit: `b1d4a12`.

## Existing paths reused

- `experiments/next_round.py` is the one-task training entry point. It owns
  paired data preparation, ratio pretraining, the SCARCE classifier loop,
  per-epoch evaluation, checkpoints, and run-level JSON/CSV output.
- `scripts/run_next_round.py` expands JSON-compatible YAML matrices and maps one
  deterministic task to one process.
- `models/ratio_factory.py`, `models/ratio_estimator.py`, and
  `models/conditional_ratio.py` provide the method registry and the unified,
  multihead, fusion, and separate conditional estimators.
- `ratio_estimation/trainer.py` trains target-positive domain discriminators;
  `ratio_estimation/statistics.py` already separates raw, clipped, normalized,
  and classifier-used weight stages.
- `scripts/summarize_next_round.py` provides the existing paired bootstrap/t-CI,
  CSV aggregation, and plotting conventions.
- `wisteria/submit_next_round.sh` and
  `wisteria/job_next_round_multigpu.template.sh` provide the established
  one-task-per-GPU PJM/Aquarius launcher.
- The original SCARCE bridge is loaded from
  `experiments/scarce_ratio_comparison.py`; its classifier, SCAR risk, ADIW KMM
  updates, optimizer, and MNIST loader remain the source of truth.

## Phase A artifacts

- The remote Wisteria repository contains 95 completed Phase A runs under
  `results/next_round_phase_a`.
- The local synchronized copy is under
  `outputs/wisteria_next_round_phase_a`, including run configs, summaries,
  epoch trajectories, four ratio stages, discriminator diagnostics, per-class
  metrics, fusion gates, figures, and the Phase A experiment report.
- The audited domain convention is source `d=0`, target `d=1`, hence neural
  target/source ratios use
  `D(x)/(1-D(x)) * pi_source/pi_target`.

## Causal-design issue found

The existing `no_iw` path is source-only, while ADIW draws both source and
target weak batches and uses `combined_adiw_scar_loss`. Therefore
`adiw_ones - no_iw` is not a pure weight contrast. Phase B adds
`matched_no_iw`, which uses the ADIW source/target mixing and schedule with
unit source weights but no ratio estimator.

## Phase B extension policy

Phase B extends the existing runner and output schema. It does not replace the
SCARCE classifier framework. New logic is isolated in:

- `src/weight_interventions.py`
- `src/ratio_diagnostics.py`
- `src/ratio_calibration.py`
- `src/synthetic_gaussian.py`
- `src/risk_recovery_metrics.py`
- `scripts/run_phase_b0.py` through `scripts/run_phase_b4.py`
- `scripts/analyze_phase_b.py`

Every formal runner records split/config/code hashes, decomposed seeds, and
`target_test_used_for_selection=false`.
