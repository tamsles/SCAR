# SCARCE with Dynamic Importance Weighting

This repository contains the official SCARCE implementation and an integrated
DIW-SCARCE training path based on
[TongtongFANG/DIW](https://github.com/TongtongFANG/DIW).

SCARCE learns from complementary labels: each training label identifies a
class to which the sample does not belong. The integrated `DIW` mode uses a
small clean validation subset to estimate a dynamic weight for every
complementary-label training sample, then applies those weights inside the
SCARCE risk.

## Requirements

- Python 3.6+
- NumPy 1.19+
- PyTorch 1.7+
- torchvision 0.8+
- SciPy 1.5+
- pandas 1.1+

`-kmm_solver auto` uses CVXOPT when it is installed and otherwise falls back
to SciPy's SLSQP solver. Install `cvxopt>=1.2` to use the solver from the
original DIW implementation directly.

## Methods

### SCARCE

The original path is unchanged at the command-line level and uses all
training samples to generate complementary labels.

```bash
python main.py -ds mnist -gen random -me SCARCE -mo mlp -op adam -lr 1e-3 -wd 1e-5 -bs 256 -ep 200 -seed 0 -gpu 0
```

### DIW-SCARCE

`-me DIW` performs the following steps for every batch after warm-up:

1. Hold out `-num_val` clean examples with ordinary labels.
2. Measure pointwise complementary NLL on the training batch.
3. Measure ordinary cross-entropy on a clean validation batch.
4. Match the two loss distributions with Kernel Mean Matching.
5. Apply the resulting sample weights to the SCARCE risk.

```bash
python main.py -ds fashion -gen random -me DIW -mo lenet -op sgd -lr 3e-4 -wd 1e-4 -bs 256 -ep 400 -seed 100 -gpu 0 -run_times 1 -num_val 1000 -val_bs 256 -diw_warmup 1 -kmm_solver auto
```

This is an adaptation of DIW to complementary-label learning. It is not the
same experiment as the original DIW repository, which trains on noisy ordinary
labels and uses ordinary cross-entropy for both data groups.

### ADIW-SCAR with Two Weak Domains

`main_adiw_scar.py` implements the source/target formulation in
`20260622_TaichuLiu.pptx`. Both domains expose only inputs and complementary
label vectors during training. For every class, the implementation:

1. Splits each mini-batch into the observed complementary branch and the
   unlabeled branch.
2. Uses the current per-class logistic loss as the ADIW transformation.
3. Updates target-to-source branch ratios with warm-started ADIW-KM projected
   gradient steps.
4. Builds the self-normalized, target-coefficient source SCAR risk.
5. Combines it with the target-only SCAR risk using `eta`.

The original `main.py -me DIW` mode uses a clean ordinary-label validation
subset. It is retained as a separate baseline and is not the two-weak-domain
method described above.

One-epoch rotation-shift example:

```bash
python main_adiw_scar.py -ds mnist -mo mlp -shift rotation -target_weak_size 1000 -source_size 0 -bs 256 -target_bs 256 -ep 1 -run_times 1 -target_prior uniform -adiw_steps 1 -adiw_lr 1
```

Relevant arguments:

- `shift`: `none`, `rotation` (support-like shift), `label_permutation`
  (input-output-relation shift), or `joint` (rotation plus label permutation)
- `source_gen`, `target_gen`: `single` or generalized `independent` SCAR labels
- `source_scar_rate`, `target_scar_rate`: scalar or class-wise independent
  SCAR selection probabilities
- `target_prior`: `uniform`, or `oracle` for synthetic diagnostics only
- `eta`: target-only risk fraction; a negative value uses
  `n_target / (n_source + n_target)`
- `adiw_steps`, `adiw_lr`: number and step size of warm-started PGD updates
- `risk_correction`: `abs` (SCARCE default), `relu`, or `none`

ADIW-SCAR results are written under `result/adiw_scar`.

### Single-domain SCARCE baselines

`main_domain_scarce.py` trains SCARCE without ADIW. `train` maps to the
unshifted source split, `test` maps to the shifted target weak split, and
`pooled` directly concatenates both splits with one ordinary SCAR risk and no
domain weighting. Ordinary labels are never returned by the training dataset;
all models are evaluated on the shifted target test set.

```bash
python main_domain_scarce.py -domain train -ds mnist -shift rotation -ep 200 -run_times 1 -seed 0
python main_domain_scarce.py -domain test  -ds mnist -shift rotation -ep 200 -run_times 1 -seed 0
python main_domain_scarce.py -domain pooled -ds mnist -shift rotation -ep 200 -run_times 1 -seed 0
```

Results are written under `result/domain_scarce`. The two commands initialize
separate models and do not share parameters or optimizer state.

After matched seeds `0` through `4` finish for ADIW-SCAR and both single-domain
baselines, generate the ablation table with:

```bash
python summarize_ablation.py
```

The script requires all five seeds and writes mean accuracy, standard
deviation, per-seed values, and the difference from ADIW-SCAR to
`result/ablation_summary.csv`.

The four-condition shift experiment uses independent SCAR labels and compares
ADIW-SCAR with direct pooled SCAR under input, conditional-label, weak-label
mechanism, and combined shifts. On Wisteria, submit it with:

```bash
bash wisteria/submit_shift_ablation.sh
```

After all jobs finish, run `python summarize_shift_ablation.py`.

## DIW Arguments

- `num_val`: number of clean validation samples; default `1000`
- `val_bs`: validation batch size; defaults to the training batch size
- `diw_warmup`: number of initial epochs using unit weights; default `1`
- `diw_kernel_quantile`: loss-distance quantile used as RBF gamma; default `0.01`
- `diw_max_weight`: KMM upper bound for each sample weight; default `50`
- `kmm_solver`: `auto`, `cvxopt`, or `scipy`

Results are written under `result/total` and `result/detail`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citations

```bibtex
@inproceedings{wang2024learning,
    author = {Wang, Wei and Ishida, Takashi and Zhang, Yu-Jie and Niu, Gang and Sugiyama, Masashi},
    title = {Learning with Complementary Labels Revisited: The Selected-Completely-at-Random Setting Is More Practical},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    year = {2024}
}

@inproceedings{fang2020rethinking,
    author = {Fang, Tongtong and Lu, Nan and Niu, Gang and Sugiyama, Masashi},
    title = {Rethinking Importance Weighting for Deep Learning under Distribution Shift},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2020}
}

@article{fang2026accelerated,
    author = {Fang, Tongtong and Lu, Nan and Niu, Gang and Fukumizu, Kenji and Sugiyama, Masashi},
    title = {Accelerated Dynamic Importance Weighting with Versatile Divergence-Minimizing Estimators},
    journal = {arXiv preprint arXiv:2605.25499},
    year = {2026}
}
```
