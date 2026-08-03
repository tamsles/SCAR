# Phase B4: class-heterogeneous MNIST shifts

A practical conditional advantage requires paired mean >1 pp and the 95% t-CI lower bound >0 in a heterogeneous setting.

| Setting | Conditional | mean Δ pp | t 95% CI | bootstrap 95% CI | criterion met |
|---|---|---:|---:|---:|---|
| H0_no_shift | unified | -17.315 | [-34.988, 0.358] | [-29.919, -7.301] | False |
| H1_homogeneous | unified | -1.926 | [-4.760, 0.907] | [-3.859, -0.219] | False |
| H2_magnitude_heterogeneous | unified | -3.363 | [-5.882, -0.845] | [-4.915, -1.811] | False |
| H3_sparse_heterogeneous | unified | -11.530 | [-20.682, -2.378] | [-17.702, -6.072] | False |
| H4_direction_heterogeneous | unified | -0.574 | [-2.906, 1.757] | [-2.001, 0.852] | False |
| H0_no_shift | multihead | -9.439 | [-13.335, -5.543] | [-11.902, -7.011] | False |
| H1_homogeneous | multihead | -1.578 | [-4.476, 1.320] | [-3.275, 0.406] | False |
| H2_magnitude_heterogeneous | multihead | -4.582 | [-7.657, -1.507] | [-6.364, -2.381] | False |
| H3_sparse_heterogeneous | multihead | -7.286 | [-11.403, -3.168] | [-9.909, -4.663] | False |
| H4_direction_heterogeneous | multihead | -2.227 | [-5.017, 0.564] | [-4.004, -0.450] | False |
