# Phase B2: frozen-score post-processing ablation

Each seed trains one ratio estimator; all classifier configurations reuse its frozen logits and the same classifier/batch seeds.

Calibration candidates were selected without target class labels: `[{'lower_clip': 1e-06, 'normalization': 'global'}, {'lower_clip': 1e-06, 'normalization': 'none'}]`.

| lower | normalization | calibration | n | accuracy | lower clip rate | nESS | CV | MMD | AUC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | conditional_branch | none | 3 | 19.532 | 0.0000 | 0.0100 | 10.1140 | 0.3008 | 0.8310 |
| 0 | global | none | 3 | 18.732 | 0.0000 | 0.0082 | 11.1899 | 0.3126 | 0.8192 |
| 0 | none | none | 3 | 20.979 | 0.0000 | 0.0000 | 147.6196 | 0.4766 | 0.9451 |
| 0.0001 | conditional_branch | none | 3 | 55.046 | 0.9926 | 0.0898 | 3.2651 | 0.1779 | 0.8599 |
| 0.0001 | global | none | 3 | 54.228 | 0.9926 | 0.0703 | 3.7307 | 0.1700 | 0.8558 |
| 0.0001 | none | none | 3 | 55.669 | 0.9926 | 0.0001 | 126.5126 | 0.1705 | 0.8383 |
| 0.001 | conditional_branch | none | 3 | 60.396 | 0.9980 | 0.2039 | 2.0179 | 0.1464 | 0.8650 |
| 0.001 | global | none | 3 | 60.993 | 0.9980 | 0.1636 | 2.3142 | 0.1473 | 0.8726 |
| 0.001 | none | none | 3 | 60.264 | 0.9980 | 0.0003 | 56.9625 | 0.1541 | 0.8819 |
| 0.01 | conditional_branch | none | 3 | 62.093 | 0.9996 | 0.3783 | 1.3056 | 0.1346 | 0.8549 |
| 0.01 | global | none | 3 | 61.195 | 0.9996 | 0.3188 | 1.4945 | 0.1417 | 0.8709 |
| 0.01 | none | none | 3 | 61.278 | 0.9996 | 0.0140 | 9.1191 | 0.1354 | 0.8693 |
| 1e-06 | conditional_branch | none | 3 | 33.432 | 0.9369 | 0.0144 | 8.3515 | 0.1980 | 0.7955 |
| 1e-06 | global | none | 3 | 32.663 | 0.9369 | 0.0116 | 9.3439 | 0.2154 | 0.7302 |
| 1e-06 | global | platt | 3 | 30.461 | 0.2761 | 0.0215 | 6.7920 | 0.2432 | 0.8328 |
| 1e-06 | global | temperature | 3 | 30.343 | 0.2374 | 0.0262 | 6.3522 | 0.2208 | 0.8137 |
| 1e-06 | none | none | 3 | 34.172 | 0.9369 | 0.0001 | 147.3964 | 0.3888 | 0.6482 |
| 1e-06 | none | platt | 3 | 28.826 | 0.2761 | 0.0003 | 66.6387 | 0.2501 | 0.8252 |
| 1e-06 | none | temperature | 3 | 29.867 | 0.2374 | 0.0004 | 68.4942 | 0.2360 | 0.6972 |
