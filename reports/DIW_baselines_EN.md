# IWPU / DIW: Baselines and Classification Accuracy

Results from our independent implementation are reported as **mean accuracy ± sample standard deviation (%)**. Each entry aggregates 30 runs: three target PU training sizes—10P/50U, 20P/100U, and 40P/200U—with ten random seeds per size. The standard deviation therefore reflects variation across both seeds and target training sizes.

**IO:** input-output relation shift. **Support:** support shift. **FMNIST:** Fashion-MNIST. Diabetes uses a race-based domain shift. PU denotes positive-unlabeled learning.

| Method | MNIST IO | MNIST Support | FMNIST IO | FMNIST Support | CIFAR-10 IO | CIFAR-10 Support | Diabetes |
|---|---:|---:|---:|---:|---:|---:|---:|
| IWPU / DIW | 70.17 ± 8.96 | 75.21 ± 4.67 | 91.68 ± 4.86 | 96.65 ± 1.32 | 70.10 ± 4.01 | 82.71 ± 4.01 | 71.52 ± 1.47 |
| tePU | 72.99 ± 6.69 | 73.24 ± 5.02 | 88.54 ± 3.93 | 96.01 ± 1.83 | 67.64 ± 6.21 | 66.87 ± 8.56 | 62.43 ± 4.62 |
| trPU | 55.51 ± 1.82 | 58.82 ± 3.31 | 73.21 ± 7.51 | 90.97 ± 1.29 | 66.79 ± 5.48 | 82.92 ± 2.88 | 71.39 ± 4.21 |
| ftPU | 62.34 ± 6.80 | 67.30 ± 4.71 | 69.80 ± 6.89 | 91.79 ± 2.81 | 64.73 ± 3.53 | 83.95 ± 2.42 | 71.98 ± 1.67 |
| mtPU | 70.31 ± 8.09 | 73.66 ± 3.98 | 90.82 ± 4.12 | 95.43 ± 1.98 | 69.94 ± 4.22 | 82.22 ± 3.68 | 72.08 ± 1.30 |
| mtsPU | 69.63 ± 8.65 | 74.48 ± 4.20 | 90.76 ± 4.75 | 95.83 ± 1.64 | 69.88 ± 4.34 | 82.15 ± 3.96 | 71.81 ± 1.26 |
| DAPU | 69.52 ± 8.98 | 75.68 ± 4.48 | 91.27 ± 4.29 | 95.78 ± 1.51 | 70.68 ± 4.22 | 83.64 ± 3.40 | 71.96 ± 1.32 |
| UDAPU | 55.55 ± 1.93 | 64.90 ± 4.49 | 74.23 ± 8.14 | 93.34 ± 2.37 | 66.84 ± 5.36 | 84.17 ± 2.88 | 69.95 ± 6.88 |
| GIW (approximate implementation) | 71.50 ± 7.38 | 73.72 ± 6.13 | 88.50 ± 3.83 | 95.44 ± 2.00 | 70.75 ± 3.62 | 80.96 ± 4.33 | 55.64 ± 9.62 |

## Method Descriptions

| Method | Implementation used in these experiments |
|---|---|
| IWPU / DIW | Alternates between PU classification and joint density-ratio estimation to update importance weights dynamically. |
| tePU | Trains a PU classifier using only target-domain PU data. |
| trPU | Trains a PU classifier using only source-domain PU data. |
| ftPU | Pretrains a PU classifier on source-domain data, then fine-tunes it on target-domain PU data. |
| mtPU | Shares a feature extractor between separate source and target classifier heads and optimizes a weighted combination of the two domains' PU risks. |
| mtsPU | Uses one shared classifier and optimizes a weighted combination of source and target PU risks. |
| DAPU | Combines source and target PU risks with maximum mean discrepancy (MMD) feature alignment. |
| UDAPU | Combines source PU risk with MMD alignment using unlabeled target-domain data. |
| GIW | Constructs pseudo-positive/negative labels and applies dynamic importance weighting; the implementation used here is approximate. |

**Protocol note.** These are locally reproduced results, not the paper's reported values or results from the authors' official code. Image results come from the original full-baseline experiment; Diabetes results come from the corrected preprocessing rerun. A subsequent correction to the per-domain batch-size convention was evaluated separately in the IWPU-versus-two-step experiment. Consequently, this table documents the completed baseline experiments and should not be presented as a full rerun under that final corrected protocol.

**Source records:** image-task results from `reports/main_table.csv`; Diabetes results from `reports/tabular_no_target_rescale_v2/main_table.csv` in the IWPU reproduction project. Values are unchanged from the verified baseline summary prepared on September 8, 2026.
