# Phase B0: ADIW causal decomposition

Primary metric: mean target accuracy over the last 10 classifier epochs. Target test labels were not used for selection.

## Method means

| Method | n | Accuracy (%) | Used-weight SD | normalized ESS | weighted MMD | post-hoc AUC |
|---|---:|---:|---:|---:|---:|---:|
| adiw_mean | 10 | 60.987 | 0.0000 | 1.0000 | 0.1492 | 0.8598 |
| adiw_ones | 10 | 61.094 | 0.0000 | 1.0000 | 0.1475 | 0.8602 |
| adiw_original | 10 | 61.790 | 0.0383 | 0.9985 | 0.1521 | 0.8563 |
| adiw_shuffle | 10 | 60.472 | 0.0374 | 0.9986 | 0.1487 | 0.8598 |
| current_no_iw | 10 | 53.701 | 0.0000 | 1.0000 | 0.1750 | 0.8559 |
| matched_no_iw | 10 | 60.294 | 0.0000 | 1.0000 | 0.1463 | 0.8577 |

## Paired causal contrasts

| Contrast | mean pp | t 95% CI | bootstrap 95% CI |
|---|---:|---:|---:|
| adiw_original − adiw_shuffle | 1.319 | [-0.158, 2.795] | [0.132, 2.546] |
| adiw_original − adiw_mean | 0.803 | [-0.855, 2.461] | [-0.616, 2.104] |
| adiw_original − adiw_ones | 0.697 | [-1.198, 2.591] | [-0.910, 2.161] |
| adiw_ones − current_no_iw | 7.393 | [5.895, 8.891] | [6.168, 8.619] |
| matched_no_iw − current_no_iw | 6.593 | [5.289, 7.897] | [5.485, 7.619] |

Results are causal only for the preregistered within-seed interventions. Fewer than 10 paired seeds is a smoke/provisional result.
