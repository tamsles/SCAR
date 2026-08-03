# Phase B1: ratio correctness audit

Domain convention: source=0, target=1; neural ratios use `D/(1-D) × pi_source/pi_target`. ADIW uses KMM and has no discriminator probability/odds.

| Method | n | weighted MMD | post-hoc AUC | post-hoc accuracy |
|---|---:|---:|---:|---:|
| adiw_original | 5 | 0.1531 | 0.8644 | 0.8079 |
| fusion | 5 | 0.4508 | 0.9995 | 0.9867 |
| multihead | 5 | 0.4645 | 0.9953 | 0.9841 |
| unified | 5 | 0.4623 | 0.9990 | 0.9873 |
