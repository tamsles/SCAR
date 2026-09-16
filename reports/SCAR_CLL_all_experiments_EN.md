# SCAR / Complementary-Label Learning: Complete Available Experiment Inventory

Prepared for supervisor review, September 8, 2026. This document consolidates all formal SCAR/CLL experiment batches located in the local repositories and synchronized results, plus the early incomplete SCARCE run. It does not substitute PU-learning results for CLL results.

**Reporting convention.** Classification accuracy is in percent. Unless explicitly stated otherwise, each run contributes its mean target accuracy over the final 10 classifier epochs, and tables report the mean ± population standard deviation across seeds. Different batches remain separate because their data splits, training paths, or ratio settings differ. Smoke/debug runs are not treated as completed scientific comparisons.

**Fine-tuning versus pooling.** Both are listed explicitly. Genuine fine-tuning means source pretraining followed by target-domain fine-tuning; no completed SCAR result for that procedure was located. Pooled SCAR trains on combined source/target data without importance weighting. The historical label ftSCAR refers to this pooled implementation, not a separate fine-tuning experiment.

## 1. Method inventory and missing baseline results

| Method or family | What is available |
|---|---|
| Original SCARCE | One completed MNIST run and one incomplete run; Section 2. |
| Source-only SCAR / trainSCAR / no_iw / current_no_iw | Completed results in Sections 3, 6, 7, and 11; these names occur in different training batches. |
| Target-only SCAR / testSCAR | Completed original five-seed comparison; Section 3. |
| Pooled SCAR (historically labeled ftSCAR) | Completed original comparison, four-shift experiment, and Phase A results; Sections 3, 4, and 6. Trains on combined source and target data without importance weighting. |
| ADIW-SCAR / adiw / adiw_original | Completed domain-shift, Phase A, B0, B1, and B4 experiments. Uses online KMM weights in this implementation. |
| Conditioned / unified, multihead, fusion, separate | Completed neural density-ratio comparisons; Sections 3, 5, and 6, with further diagnostics in Phase B. |
| matched_no_iw, adiw_ones, adiw_mean, adiw_shuffle | Completed training-pipeline and weight-assignment interventions; Section 7. |
| Estimated global/conditional ratios and oracle ratios | Completed Gaussian CLL benchmark; Section 10. Oracle access is diagnostic only. |
| mtSCAR (source/target multi-task classifiers) | No completed SCAR training result or dedicated implementation was located in the inspected local repositories. Earlier discussion proposed the CLL analogue of mtPU; no verified SCAR accuracy is available. |
| mtsSCAR (single-network multi-task SCAR) | No separately identified completed result or dedicated implementation was located. It must not be assigned mtsPU accuracy or silently equated with pooled SCAR. |
| Genuine source-pretraining/target-fine-tuning SCAR | No completed result was located. The historical ftSCAR label did not implement this procedure. |
| DANN/MMD/CLARINET + SCAR; pseudo-label GIW + SCAR | Discussed as possible CLL baselines; no completed results were located in the inspected SCAR outputs. PU DAPU/UDAPU/GIW results are not SCAR results. |
| DIW mode in the early SCARCE main.py | A code path exists, but no separate completed DIW-mode result was located in the early result directory. Do not equate this with the later ADIW-SCAR result. |

The earlier baseline-design discussion mapped mtPU and mtsPU to potential CLL counterparts. That discussion is evidence of a proposal, not a completed experiment. The absence statements above are limited to the local code/results and task history inspected; they are not proof that no unsynchronized external run exists.

## 2. Early original SCARCE run (incomplete repetition set)

| Experiment | Available completion | Accuracy (%) |
|---|---|---:|
| Original SCARCE, MNIST, MLP, 200 epochs | Run 1 complete; reported last-10-epoch mean | 91.052 |
| Same intended five-run experiment | Run 2 saved through epoch 94; complete five-run aggregate absent | Not available |

This is the original MNIST SCARCE setting, not the later source/target rotation-30 protocol. The single completed result cannot be reported as a five-seed mean ± SD or compared directly with the shifted-domain tables.

Source: [early per-epoch record](<C:/Users/Tamsles/Documents/Codex/2026-07-10/wwangwitsel-scarce-git-https-github-com/work/SCARCE/result/detail/Res_detail_mnist_random_SCARCE_mlp_adam_lr0.001_wd1e-05_b256_e200_s0_r5.csv>).

## 3. Original domain-training and neural-ratio comparison

MNIST + MLP; target rotation 30°; source weak set 59,000; target weak training set 1,000; independent target test set 10,000. Adam, learning rate 0.001, weight decay 0.00001, batch size 256, 200 classifier epochs; five seeds (0–4). Ordinary labels are used only for evaluation.

| Method | Accuracy (%) |
|---|---:|
| ADIW-SCAR | 60.98 ± 2.44 |
| ftSCAR / pooled SCAR | 60.31 ± 1.68 |
| trainSCAR | 54.30 ± 0.86 |
| testSCAR | 38.82 ± 2.44 |
| Conditioned-Ratio-SCAR | 60.65 ± 2.84 |
| Multihead-Ratio-SCAR | 60.99 ± 1.85 |
| Fine-tuning SCAR (source pretraining → target fine-tuning) | Unavailable: no completed result located |

trainSCAR uses only source complementary labels. testSCAR uses only the small target weak training set, never the held-out test set. Conditioned-Ratio-SCAR conditions a shared estimator on class and weak-label state; Multihead-Ratio-SCAR uses shared features with separate conditional-ratio outputs. Their mean accuracies are close to ADIW-SCAR; this batch does not establish a neural-ratio accuracy advantage.

Source: [original comparison and per-seed values](<C:/Users/Tamsles/Documents/SCAR/outputs/scarce_comparison/scar_method_comparison.csv>).

## 4. Four types of distribution shift

Separate MNIST/MLP batch; 59,000 source and 1,000 target weak examples, 200 epochs, five seeds. Both ADIW and the unweighted pooled-data baseline are included.

| Shift intervention | Method | Accuracy (%) |
|---|---|---:|
| Covariate shift: target rotation 30° | ADIW-SCAR | 60.97 ± 0.39 |
| Covariate shift: target rotation 30° | Pooled SCAR (no IW) | 60.03 ± 1.08 |
| Label-conditional shift: target labels cyclically permuted by +1 | ADIW-SCAR | 0.87 ± 0.09 |
| Label-conditional shift: target labels cyclically permuted by +1 | Pooled SCAR (no IW) | 0.92 ± 0.09 |
| Complementary-label mechanism shift | ADIW-SCAR | 91.45 ± 0.14 |
| Complementary-label mechanism shift | Pooled SCAR (no IW) | 91.64 ± 0.27 |
| Joint shift: all three interventions | ADIW-SCAR | 9.18 ± 1.67 |
| Joint shift: all three interventions | Pooled SCAR (no IW) | 9.32 ± 1.59 |

The very low accuracy under label permutation belongs to a deliberate label-mapping stress test. It is not the standard rotation-only benchmark.

Source: [four-shift aggregate and per-seed results](<C:/Users/Tamsles/Documents/Codex/2026-07-14/rang/outputs/wisteria_shift_ablation_20260715/shift_ablation_summary.csv>).

## 5. Full neural-ratio architecture comparison

Separate 20-run batch: four architectures × five seeds, MNIST rotation 30°, 10 ratio-training epochs and 200 classifier epochs. The classifier and data sizes match the stated MNIST family, but these values must remain separate from Section 3.

| Architecture | Description | Accuracy (%) |
|---|---|---:|
| unified | Shared class/state-conditioned ratio estimator | 61.50 ± 2.27 |
| multihead | Shared backbone with branch-specific ratio heads | 61.94 ± 1.16 |
| fusion | Learned combination of global and conditional ratios | 61.15 ± 2.16 |
| separate | Independent ratio estimators for the class/state branches | 56.28 ± 2.25 |

Source: [full architecture results](<C:/Users/Tamsles/Documents/SCAR/outputs/wisteria_full_results_20260725/summary.json>).

## 6. Phase A: rotation severity and matched architecture comparison

The full batch contains 95 runs, including the pooled comparator. Each available method/angle cell contains five seeds. MNIST source size 59,000, target weak size 1,000; ratio clipping [0.01, 20] and conditional-branch normalization. The no_iw baseline is source-only; it does not match the ADIW source/target mixing pipeline.

| Method | Rotation 10° | Rotation 20° | Rotation 30° |
|---|---:|---:|---:|
| no_iw | 86.16 ± 0.76 | 72.37 ± 1.25 | 54.59 ± 2.35 |
| pooled_scar | 86.66 ± 0.52 | 75.37 ± 1.57 | 59.55 ± 2.98 |
| adiw | 87.06 ± 0.28 | 76.15 ± 1.39 | 62.69 ± 2.88 |
| unified | 85.91 ± 0.80 | 75.58 ± 1.14 | 60.66 ± 1.27 |
| multihead | 85.96 ± 0.76 | 74.96 ± 1.04 | 59.66 ± 3.52 |
| fusion | 86.38 ± 0.55 | 75.94 ± 0.58 | 59.36 ± 1.90 |
| separate_full | Not run in this matrix | Not run in this matrix | 52.01 ± 2.63 |

Source: [Phase A setting summary](<C:/Users/Tamsles/Documents/SCAR/outputs/wisteria_next_round_phase_a/summary/summary_by_setting.csv>).

## 7. Phase B0: weight assignment and training-pipeline interventions

60 completed runs: six interventions × ten seeds. MNIST rotation 30°, target weak size 1,000. Source-only and pipeline-matched unit-weight baselines are explicitly distinguished.

| Method | Intervention | Accuracy (%) |
|---|---|---:|
| current_no_iw | Current source-only no-IW baseline | 53.70 ± 1.79 |
| matched_no_iw | ADIW-matched source/target mixing and schedule, without a ratio estimator | 60.29 ± 2.01 |
| adiw_original | Original online KMM importance weights | 61.79 ± 1.85 |
| adiw_ones | ADIW path with all source weights replaced by one | 61.09 ± 2.45 |
| adiw_mean | ADIW path with sample weights replaced by their mean | 60.99 ± 2.33 |
| adiw_shuffle | ADIW path with weights shuffled across samples | 60.47 ± 1.90 |

The gain from matching the training pipeline is much larger than the difference between original and unit/shuffled weights. A source-only baseline alone cannot isolate the benefit of importance weighting.

Source: [B0 report](<C:/Users/Tamsles/Documents/SCAR/reports/phase_b0_adiw_causal_decomposition.md>).

## 8. Phase B1: ratio correctness audit (no CLL classification accuracy)

20 completed audits: four methods × five seeds. These runs do not train/evaluate a downstream CLL classifier. Their domain-discriminator accuracy must not be reported as target class accuracy. Lower weighted MMD and domain AUC closer to 0.5 indicate better domain matching.

| Method | CLL accuracy | Mean weighted MMD | Mean post-hoc domain AUC | Mean domain accuracy (%) |
|---|---|---:|---:|---:|
| adiw_original | Not applicable | 0.1531 | 0.8644 | 80.79 |
| fusion | Not applicable | 0.4508 | 0.9995 | 98.67 |
| multihead | Not applicable | 0.4645 | 0.9953 | 98.41 |
| unified | Not applicable | 0.4623 | 0.9990 | 98.73 |

Source: [B1 audit report](<C:/Users/Tamsles/Documents/SCAR/reports/phase_b1_ratio_audit.md>).

## 9. Phase B2: frozen-score clipping, normalization, and calibration

57 classifier runs: 15 uncalibrated settings × three seeds, plus four calibrated settings × three seeds. A single unified ratio estimator is trained per seed; frozen scores are reused. MNIST rotation 30°, upper ratio clip 20, 200 classifier epochs. Calibration candidates were selected without target class labels.

| Lower ratio clip | Normalization | Calibration | Accuracy (%) |
|---:|---|---|---:|
| 0 | none | none | 20.98 ± 2.55 |
| 0 | global | none | 18.73 ± 2.09 |
| 0 | conditional_branch | none | 19.53 ± 3.12 |
| 1E-06 | none | none | 34.17 ± 2.41 |
| 1E-06 | none | temperature | 29.87 ± 2.66 |
| 1E-06 | none | platt | 28.83 ± 3.69 |
| 1E-06 | global | none | 32.66 ± 0.34 |
| 1E-06 | global | temperature | 30.34 ± 4.90 |
| 1E-06 | global | platt | 30.46 ± 3.42 |
| 1E-06 | conditional_branch | none | 33.43 ± 2.44 |
| 0.0001 | none | none | 55.67 ± 3.83 |
| 0.0001 | global | none | 54.23 ± 2.94 |
| 0.0001 | conditional_branch | none | 55.05 ± 3.19 |
| 0.001 | none | none | 60.26 ± 1.54 |
| 0.001 | global | none | 60.99 ± 1.41 |
| 0.001 | conditional_branch | none | 60.40 ± 2.15 |
| 0.01 | none | none | 61.28 ± 0.89 |
| 0.01 | global | none | 61.19 ± 2.12 |
| 0.01 | conditional_branch | none | 62.09 ± 1.26 |

Higher lower-clipping floors strongly improve accuracy in this grid. This should not be interpreted as successful density-ratio recovery: the largest improvements occur when many raw ratios are collapsed to the floor. The grid is a diagnostic comparison, not a test-label-selected final model.

Source: [B2 report](<C:/Users/Tamsles/Documents/SCAR/reports/phase_b2_postprocessing_ablation.md>).

## 10. Phase B3: Gaussian CLL benchmark with known density ratios

400 runs: four shifts × five methods × twenty seeds. Three classes, 2,000 source examples, 1,000 target weak examples, 2,000 target test examples; 50 classifier epochs. Oracle rows use analytically known ratios and are diagnostic references, not deployable weakly supervised baselines.

| Method | No shift | Homogeneous conditional shift | Heterogeneous conditional shift | Class-prior shift only |
|---|---:|---:|---:|---:|
| no_iw | 98.62 ± 0.20 | 96.56 ± 0.38 | 77.18 ± 0.76 | 98.89 ± 0.21 |
| global_estimated_ratio | 98.62 ± 0.19 | 96.27 ± 0.52 | 76.66 ± 1.00 | 98.88 ± 0.21 |
| conditional_estimated_ratio | 98.63 ± 0.18 | 98.18 ± 0.26 | 77.26 ± 0.93 | 98.88 ± 0.20 |
| oracle_global_ratio | 98.62 ± 0.20 | 95.97 ± 0.64 | 76.37 ± 1.01 | 98.89 ± 0.21 |
| oracle_conditional_ratio | 98.62 ± 0.20 | 98.42 ± 0.26 | 77.55 ± 1.06 | 98.93 ± 0.19 |

Source: [B3 ground-truth benchmark report](<C:/Users/Tamsles/Documents/SCAR/reports/phase_b3_gaussian_ground_truth.md>).

## 11. Phase B4: class-dependent MNIST rotations

100 runs: five rotation patterns × four methods × five seeds. Source weak size 59,000, target weak size 1,000; 10 ratio epochs and 200 classifier epochs; clipping [0.01, 20] with conditional-branch normalization.

| Setting | Target rotation angles for classes 0–9 |
|---|---|
| H0: no shift | 0° for every class |
| H1: homogeneous | +30° for every class |
| H2: heterogeneous magnitude | 10°, 10°, 20°, 20°, 30°, 30°, 40°, 40°, 50°, 50° |
| H3: sparse shift | 0° for classes 0–4; +60° for classes 5–9 |
| H4: opposite directions | +30° for classes 0–4; −30° for classes 5–9 |

| Method | H0 | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|---:|
| current_no_iw | 91.15 ± 0.20 | 53.27 ± 1.84 | 56.57 ± 0.73 | 58.28 ± 0.72 | 50.32 ± 1.91 |
| adiw_original | 91.31 ± 0.37 | 62.15 ± 2.23 | 64.96 ± 1.24 | 66.25 ± 1.46 | 57.54 ± 1.36 |
| unified | 74.00 ± 12.97 | 60.22 ± 3.03 | 61.60 ± 0.82 | 54.72 ± 6.33 | 56.96 ± 0.74 |
| multihead | 81.87 ± 3.16 | 60.57 ± 2.10 | 60.38 ± 1.29 | 58.96 ± 4.10 | 55.31 ± 1.42 |

Neither conditional architecture met the preregistered practical-improvement criterion against ADIW in any heterogeneous setting (paired gain >1 percentage point with a positive lower 95% t-confidence bound).

Source: [B4 report](<C:/Users/Tamsles/Documents/SCAR/reports/phase_b4_heterogeneous_mnist.md>).

## 12. Earlier synthetic ratio-estimation ablations (no downstream CLL accuracy)

Two completed synthetic batches, each with ten methods and seeds 7, 17, 29. These test density-ratio recovery rather than classifier accuracy. The table reports log-ratio RMSE, mean ± population SD; lower is better. Domain-validation accuracy in these files is not CLL classification accuracy.

| Ratio estimator / ablation | Mixed covariate and mild label shift: RMSE | Heterogeneous branch shift: RMSE | CLL accuracy |
|---|---:|---:|---|
| proposed | 0.2230 ± 0.0110 | 0.4184 ± 0.0314 | Not measured |
| multihead | 0.2316 ± 0.0120 | 0.3125 ± 0.0264 | Not measured |
| no_class_condition | 0.2550 ± 0.0127 | 0.4195 ± 0.0675 | Not measured |
| no_state_condition | 0.2578 ± 0.0249 | 0.4057 ± 0.0502 | Not measured |
| no_prior_correction | 0.2629 ± 0.0240 | 0.6090 ± 0.0449 | Not measured |
| branch_none | 0.2161 ± 0.0146 | 0.4430 ± 0.0556 | Not measured |
| branch_inverse_frequency | 0.2153 ± 0.0124 | 0.4594 ± 0.0407 | Not measured |
| marginal_ratio | 0.2523 ± 0.0112 | 0.4287 ± 0.0379 | Not measured |
| unit_ratio | 1.1120 ± 0.0082 | 0.5750 ± 0.0038 | Not measured |
| oracle_ratio | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Not measured |

proposed is the conditioned estimator with balanced BCE; branch_none removes branch balancing; branch_inverse_frequency uses inverse-frequency balancing; no_class_condition/no_state_condition remove the respective conditioning input; no_prior_correction omits count-prior correction in the relevant unbalanced setup; marginal_ratio estimates a single pooled density ratio; unit_ratio uses weights of one; oracle_ratio uses known ratios. The marginal density-ratio baseline is distinct from pooling source and target classifier training data.

Source: [mixed synthetic ratio ablation](<C:/Users/Tamsles/Documents/SCAR/outputs/ablation_mixed/summary.csv>).


Source: [heterogeneous synthetic ratio ablation](<C:/Users/Tamsles/Documents/SCAR/outputs/ablation_branch_shift/summary.csv>).

## 13. Completeness and provenance

- All five formal Phase B stages are included: B0=60, B1=20, B2=57, B3=400, B4=100 (637 records). Classification means and population SDs were recomputed from per-run summary.json files; group sizes and unique seeds were checked.
- Original SCAR, four-shift, full-architecture, and Phase A means were checked against stored per-seed values. The 91.052% early SCARCE value was recomputed from the last ten epochs of its one completed run.
- The initial five-seed ADIW result also appears in a separate July 14 archive; it is the same 60.978 ± 2.438% result in Section 3, not an additional independent experiment.
- Incomplete runs and smoke/debug runs are not silently promoted to formal results. The incomplete original repetition set is explicitly shown in Section 2.
- mtSCAR/mtsSCAR are explicitly listed as lacking verified results. Proposed baseline names and available PU implementations are not evidence that their CLL versions were run.
- This is an inventory of located local/synchronized evidence. No new models were trained and no unsynchronized remote files were inspected for this compilation.


Source: [reproducible aggregation and checks](<C:/Users/Tamsles/Documents/SCAR/reports/build_all_scar_en.ps1>).

