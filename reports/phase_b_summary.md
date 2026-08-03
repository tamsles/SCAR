# Next-Round Phase B summary

**Evidence status:** formal matrix complete.

Formal run counts: B0=60, B1=20, B2=57, B3=400, B4=100 (total=637). Target test labels were used only for offline evaluation and never for selection.

## Answers to the seven preregistered questions

1. **Does ADIW benefit from sample-specific weight assignment? Statistically uncertain, and not the main source of the gain.** Original-minus-shuffle was 1.319 pp (t 95% CI [-0.158, 2.795]); original-minus-ones was 0.697 pp (t 95% CI [-1.198, 2.591]). The shuffle t interval crosses zero, while its bootstrap interval is positive, so the data support at most a small correspondence effect rather than a robust primary mechanism.

2. **Is the ADIW advantage mainly pipeline, mixing, or loss-scale related? Pipeline/mixing is data-supported; loss scale is not separately identified.** Ones-minus-current was 7.393 pp (t 95% CI [5.895, 8.891]); matched-no-IW minus current was 6.593 pp (t 95% CI [5.289, 7.897]). By contrast, original-minus-mean was 0.803 pp (t 95% CI [-0.855, 2.461]), so the dominant reproducible gain comes from matching the ADIW training pipeline rather than per-sample weights.

3. **Does conditional ratio modeling help only under heterogeneous class-wise shift? No practical advantage was found.** B4 produced 0 qualifying heterogeneous contrasts out of 6; conditional-minus-ADIW deltas ranged from -11.530 to -0.574 pp. In B3, the estimated conditional model also beat the global model by 1.917 pp in the homogeneous shift, so any synthetic advantage is not exclusive to heterogeneity. The oracle heterogeneous delta of 1.181 pp triggered B4, but that is diagnostic-only evidence.

4. **Does the current conditional estimator recover the correct ratio? Only partially on synthetic data, and not convincingly on MNIST.** In heterogeneous B3 its mean oracle correlation was 0.804 with ratio MSE 3.2287. In B1, conditional-method post-hoc domain AUCs were 0.9953-0.9995, far from the ideal 0.5, showing that weighted MNIST source samples remained readily distinguishable from target samples.

5. **Do clipping and branch normalization amplify saturation artifacts? Clipping is the dominant stabilizer; branch normalization is not a consistent amplifier in this grid.** Conditional-branch accuracy rose from 19.532% at lower clip 0 to 62.093% at 0.01, where 99.96% of raw ratios were below the floor. The target-label-free selected 1e-6 configurations scored 32.663-34.172% uncalibrated and 28.826-30.461% after temperature/Platt calibration. Thus performance improves mainly as saturated ratios are collapsed toward the floor; calibration did not rescue the selected ratios.

6. **Can the weak-risk estimator reliably rank models and select checkpoints? Not generally; reliability is setting-dependent.** In B3, weak-risk Pearson/Spearman were 0.993/0.959 with regret 0.0052; in B4 they fell to 0.354/0.304 with regret 1.2463. B4 IW-risk Pearson was 0.186 with regret 1.1163, so importance weighting did not provide reliable cross-method selection there.

7. **Should the next round expand to 10-20 seeds or a target-weak-size sweep? Do not broadly scale the current conditional estimator.** B3 already has 20 seeds, B0 has 10, and the five-seed B4 matrix shows no qualifying conditional advantage, including significant negative effects in H2/H3. First redesign ratio estimation and target-label-free selection; then prioritize a preregistered target-weak-size sweep. Expanding B0 to 20 seeds is reasonable only if a more precise estimate of the small original-versus-shuffle effect is scientifically important.

## Evidence classification

- **Data-supported:** most ADIW gain comes from the matched training pipeline; strong lower clipping stabilizes training; current conditional methods have no practical B4 advantage.
- **Statistically uncertain:** the small sample-to-weight correspondence effect in B0.
- **Oracle-only diagnostic:** heterogeneous Gaussian oracle conditional-minus-global = 1.181 pp.
- **Not yet identifiable:** the separate causal contributions of loss scale versus source-target mixing, and target-weak-size effects.

## Stage reports

- B0 causal decomposition: `phase_b0_adiw_causal_decomposition.md`
- B1 ratio audit: `phase_b1_ratio_audit.md`
- B2 frozen post-processing: `phase_b2_postprocessing_ablation.md`
- B3 ground-truth benchmark: `phase_b3_gaussian_ground_truth.md`
- B4 heterogeneous MNIST: `phase_b4_heterogeneous_mnist.md`
