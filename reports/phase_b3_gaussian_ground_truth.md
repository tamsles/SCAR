# Phase B3: Gaussian ground-truth benchmark

Oracle methods are diagnostic upper bounds and are not formal weakly supervised methods.

| Shift | Method | n | Accuracy | ratio MSE | log-ratio MSE | corr. | MMD | excess risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| class_prior_shift_only | conditional_estimated_ratio | 20 | 98.883 | 0.0331 | 0.0366 | 0.941 | 0.0749 | 0.0016 |
| class_prior_shift_only | global_estimated_ratio | 20 | 98.876 | 0.0218 | 0.0248 | 0.968 | 0.0161 | 0.0013 |
| class_prior_shift_only | no_iw | 20 | 98.889 | 0.2957 | 0.4176 | 0.000 | 0.2419 | 0.0014 |
| class_prior_shift_only | oracle_conditional_ratio | 20 | 98.926 | 0.0000 | 0.0000 | 1.000 | 0.0726 | 0.0000 |
| class_prior_shift_only | oracle_global_ratio | 20 | 98.889 | 0.0000 | 0.0000 | 1.000 | 0.0324 | 0.0012 |
| heterogeneous_conditional_shift | conditional_estimated_ratio | 20 | 77.261 | 3.2287 | 0.3743 | 0.804 | 0.0240 | 0.0214 |
| heterogeneous_conditional_shift | global_estimated_ratio | 20 | 76.658 | 0.4694 | 0.4754 | 0.940 | 0.0253 | 0.0472 |
| heterogeneous_conditional_shift | no_iw | 20 | 77.180 | 3.5111 | 4.3713 | 0.000 | 0.2694 | 0.0200 |
| heterogeneous_conditional_shift | oracle_conditional_ratio | 20 | 77.549 | 0.0000 | 0.0000 | 1.000 | 0.0320 | 0.0000 |
| heterogeneous_conditional_shift | oracle_global_ratio | 20 | 76.368 | 0.0000 | 0.0000 | 1.000 | 0.0365 | 0.0532 |
| homogeneous_conditional_shift | conditional_estimated_ratio | 20 | 98.183 | 1.3419 | 0.2952 | 0.886 | 0.0447 | 0.0042 |
| homogeneous_conditional_shift | global_estimated_ratio | 20 | 96.266 | 1.9639 | 0.3094 | 0.878 | 0.0411 | 0.0490 |
| homogeneous_conditional_shift | no_iw | 20 | 96.556 | 5.1446 | 2.3515 | 0.000 | 0.2110 | 0.0428 |
| homogeneous_conditional_shift | oracle_conditional_ratio | 20 | 98.417 | 0.0000 | 0.0000 | 1.000 | 0.0552 | 0.0000 |
| homogeneous_conditional_shift | oracle_global_ratio | 20 | 95.969 | 0.0000 | 0.0000 | 1.000 | 0.0590 | 0.0577 |
| no_shift | conditional_estimated_ratio | 20 | 98.633 | 0.0171 | 0.0168 | 0.000 | 0.0174 | -0.0000 |
| no_shift | global_estimated_ratio | 20 | 98.621 | 0.0152 | 0.0147 | 0.000 | 0.0158 | 0.0001 |
| no_shift | no_iw | 20 | 98.621 | 0.0000 | 0.0000 | 0.000 | 0.0333 | 0.0000 |
| no_shift | oracle_conditional_ratio | 20 | 98.621 | 0.0000 | 0.0000 | 0.000 | 0.0308 | 0.0000 |
| no_shift | oracle_global_ratio | 20 | 98.621 | 0.0000 | 0.0000 | 0.000 | 0.0321 | 0.0000 |
