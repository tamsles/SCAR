# Ablation and baseline results

Wisteria job `9258749` ran all conditions with seeds `7, 17, 29` on an
NVIDIA A100 using PyTorch 1.8.1. Each trainable method used the same data,
optimizer, backbone width, clipping, and evaluation samples.

The primary metric is RMSE against the known selected conditional log-ratio
`log p_target(x | bar_y_k=z) / p_source(x | bar_y_k=z)`. Feature-moment RMSE
measures how closely importance-weighted source branch means match target
branch means. Lower values are better.

## Mixed covariate and mild label shift

| Method | Log-ratio RMSE | Moment RMSE | Correlation |
|---|---:|---:|---:|
| Oracle reference | 0.0000 | 0.0563 | 1.0000 |
| Inverse-frequency branches | 0.2153 ± 0.0124 | 0.0614 | 0.9822 |
| No branch balancing | 0.2161 ± 0.0146 | 0.0613 | 0.9824 |
| Proposed conditioned/balanced | 0.2230 ± 0.0110 | 0.0615 | 0.9810 |
| Multihead | 0.2316 ± 0.0120 | 0.0609 | 0.9824 |
| Marginal-ratio baseline | 0.2523 ± 0.0112 | 0.0592 | 0.9776 |
| No class condition | 0.2550 ± 0.0127 | 0.0666 | 0.9753 |
| No state condition | 0.2578 ± 0.0249 | 0.0608 | 0.9761 |
| Wrongly omit count prior | 0.2629 ± 0.0240 | 0.0613 | 0.9741 |
| Unit-ratio baseline | 1.1120 ± 0.0082 | 0.3589 | 0.0000 |

The proposed estimator reduces log-ratio RMSE by 79.9% versus unit weights and
11.6% versus a single marginal ratio. Because branch counts are not severely
imbalanced here, `none` and `inverse_frequency` are slightly better than
forcing every branch to contribute equally.

## Heterogeneous class-specific branch shift

| Method | Log-ratio RMSE | Moment RMSE | Correlation |
|---|---:|---:|---:|
| Oracle reference | 0.0000 | 0.0453 | 1.0000 |
| Multihead | 0.3125 ± 0.0264 | 0.0685 | 0.8760 |
| No state condition | 0.4057 ± 0.0502 | 0.0690 | 0.8179 |
| Proposed conditioned/balanced | 0.4184 ± 0.0314 | 0.0773 | 0.8048 |
| No class condition | 0.4195 ± 0.0675 | 0.0800 | 0.8161 |
| Marginal-ratio baseline | 0.4287 ± 0.0379 | 0.0796 | 0.7894 |
| No branch balancing | 0.4430 ± 0.0556 | 0.0785 | 0.7845 |
| Inverse-frequency branches | 0.4594 ± 0.0407 | 0.0827 | 0.7662 |
| Unit-ratio baseline | 0.5750 ± 0.0038 | 0.2066 | 0.0000 |
| Wrongly omit count prior | 0.6090 ± 0.0449 | 0.0785 | 0.6795 |

The direct multihead output is clearly best when branches have strongly
different shifts. The more strongly shared conditioned head becomes a capacity
bottleneck and is only 2.4% better than the marginal baseline. This supports
keeping both estimator types configurable rather than treating one as
universally superior.

Balanced BCE improves the conditioned model over unbalanced branch averaging
in this harder scenario. When unbalanced loss is used, omitting count-based
prior correction performs worse than unit weights, confirming that the prior
must match the discriminator's effective sampling distribution.

## Statistical implementation note

`balanced_bce` separately normalizes source and target losses inside every
`(k,z)` branch, so its effective source/target prior is 1:1 and its correction
is one. Raw count correction is applied only when domain contributions remain
unbalanced. A dedicated unit test verifies this behavior.

Complete per-seed records and all metrics are stored under
`outputs/ablation_mixed/` and `outputs/ablation_branch_shift/`.

## Matched comparison with original SCAR methods

The original SCARCE data path and MLP were reused for a matched MNIST
rotation-30 experiment: 59,000 source weak examples, 1,000 target weak
examples, 10,000 rotated test examples, seeds 0--4, 200 classifier epochs,
and final-10-epoch mean accuracy. The original baseline records and the fresh
ratio-estimator runs were all produced on Wisteria.

| Rank | Method | Target accuracy, mean ± population SD | Δ vs ftSCAR | Δ vs ADIW |
|---:|---|---:|---:|---:|
| 1 | Multihead-Ratio-SCAR | 60.994 ± 1.854 | +0.685 | +0.016 |
| 2 | ADIW-SCAR | 60.978 ± 2.438 | +0.669 | +0.000 |
| 3 | Conditioned-Ratio-SCAR | 60.654 ± 2.845 | +0.344 | -0.324 |
| 4 | ftSCAR / pooled SCAR | 60.309 ± 1.678 | +0.000 | -0.669 |
| 5 | trainSCAR | 54.298 ± 0.858 | -6.011 | -6.680 |
| 6 | testSCAR | 38.821 ± 2.437 | -21.489 | -22.157 |

Multihead ranks first numerically, but it is statistically tied with ADIW:
the paired difference is `+0.016` percentage points (`p=0.987`, 95% CI
`[-2.478, 2.510]`). Its `+0.685`-point difference from ftSCAR is also not
statistically clear (`p=0.585`, 95% CI `[-2.520, 3.889]`). Conditioned is
likewise tied with ftSCAR (`+0.344`, `p=0.795`) and ADIW (`-0.324`,
`p=0.710`). Five seeds are too few to establish sub-point improvements.

Both ratio estimators beat trainSCAR and testSCAR by a large margin, but so
does direct pooling. The evidence therefore supports using both domains; it
does not show a reliable classification-accuracy gain from neural ratio
weighting in this separable 30-degree rotation setting.

The ratio discriminator reached 100% validation accuracy for every neural
run. Ratios hit both configured bounds (`0.00247875` and `50`), while mean
source weights stayed within `0.002479--0.002487`. This support-separation
and clipping saturation explains why learned branch weights add little over
pooled training here.

For a matched 500-128-64 binary estimator replicated over all `2q=20`
branches, independent training would require about 9.30M ratio parameters.
The conditioned and multihead shared models use 467,677 and 466,184
parameters respectively: about 95% fewer, with one shared-backbone forward
instead of 20 independent forwards.

The complete comparison table, per-seed values, training traces, JSON
records, and checkpoints are under `outputs/scarce_comparison/`.
