# IWPU Reproduction

Independent reproduction code for **Importance-weighted Positive-unlabeled
Learning for Distribution Shift Adaptation** (Kumagai et al., AISTATS 2025).
**This is a from-paper reimplementation, not the authors' official code.**
Results may differ from those reported in the paper.

## Contents

- `src/iwpu/`: models, PU losses, importance weighting, data and training
- `scripts/`: experiment runners and result aggregation
- `configs/paper.yaml`: experiment settings
- `tests/`: unit tests
- `wisteria/`: cluster setup and job scripts

Includes IWPU, eight comparison baselines, and ablations on MNIST,
Fashion-MNIST, CIFAR-10, Diabetes, and Food Stamps.

## Setup and run

Use Python 3.10 or later. Run commands from this directory:

```bash
pip install -e .
python scripts/run_experiment.py --config configs/paper.yaml --task mnist_io --method iwpu --target-size 10,50 --seed 0 --device cuda
```

Experiments were run on Wisteria. See [WISTERIA.md](WISTERIA.md) for data
preparation and cluster setup, including the separate TableShift environment
needed for tabular tasks. Adapt the account and filesystem paths before use.

Datasets, checkpoints, and raw results are not included.
Implementation assumptions are recorded in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Paper: https://proceedings.mlr.press/v258/kumagai25a.html
