# SCAR

Experiment code for complementary-label learning under distribution shift.
Includes SCARCE baselines, importance weighting, and conditional density-ratio estimation.

## Structure

- `models/`, `ratio_estimation/`: ratio models and training
- `experiments/`, `scripts/`: experiment and analysis scripts
- `configs/`: experiment settings
- `datasets/`, `src/`: data utilities and diagnostics
- `legacy/SCARCE/`: earlier SCARCE and ADIW experiments
- `wisteria/`: cluster job scripts
- `reports/`: experiment summaries
- `tests/`: unit tests

## Setup

```bash
pip install -r requirements.txt
```

Image experiments also require `torchvision`, `scipy`, and `pandas`.

## Run

Synthetic example:

```bash
python train.py --config configs/ratio_synthetic.json --device cpu
```

List the next-round smoke experiments:

```bash
python scripts/run_next_round.py --config configs/next_round.yaml --phase smoke --experiment-mode smoke --list-tasks
```

For image experiments, pass `--scarce-repo legacy/SCARCE` to the relevant runner.
See the [legacy README](legacy/SCARCE/README.md) for the earlier experiments.

## Tests

```bash
python -m unittest discover -s tests -v
```

Datasets, checkpoints, and raw outputs are not included.
