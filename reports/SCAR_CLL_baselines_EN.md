# SCAR Complementary-Label Learning: Baselines and Accuracy

We evaluated complementary-label learning (CLL) methods on MNIST under a fixed 30° rotation of the target domain. The source training set contains 59,000 examples, and the target training set contains 1,000 examples, both with complementary labels generated under the SCAR assumption. Evaluation uses an independent target-domain test set of 10,000 examples. Ordinary class labels are used only for evaluation.

All methods use an MLP classifier trained for 200 epochs. For each seed, accuracy is averaged over the final 10 epochs. The table reports **mean target classification accuracy ± population standard deviation (%)** across five random seeds (0–4).

| Method | Description | Target accuracy (%) |
|---|---|---:|
| trainSCAR | Trains using only source-domain complementary-labeled data. | 54.30 ± 0.86 |
| testSCAR | Trains using only the small target-domain complementary-labeled training set. | 38.82 ± 2.44 |
| ADIW-SCAR | Combines source and target complementary-labeled data with ADIW importance weighting. | 60.98 ± 2.44 |
| Conditioned-Ratio-SCAR | Uses a shared neural estimator conditioned on the class and complementary-label indicator to estimate branch-specific density ratios for SCAR training. | 60.65 ± 2.84 |
| Multihead-Ratio-SCAR | Uses a shared neural backbone with separate output heads to estimate branch-specific density ratios for SCAR training. | 60.99 ± 1.85 |

The name **testSCAR** refers to training on the target-domain weakly labeled training set; it does not mean training on the held-out evaluation set. The source-only and target-only baselines use different training-set sizes, so their difference does not isolate the effect of domain shift alone.

Multihead-Ratio-SCAR and ADIW-SCAR have nearly identical mean accuracy in this experiment (a difference of 0.016 percentage points); these results do not establish an accuracy advantage for the multihead estimator over ADIW-SCAR.

*Scope:* The pooled-data variant is omitted from this table. Its historical name, ftSCAR, did not represent source pretraining followed by target fine-tuning. These results are from the original five-seed comparison; later experimental batches are not mixed into this table.

Source: [five-seed comparison](C:/Users/Tamsles/Documents/SCAR/outputs/scarce_comparison/scar_method_comparison.csv) and [experimental protocol](C:/Users/Tamsles/Documents/SCAR/outputs/scarce_comparison/original_ABLATION_REPORT.md).
