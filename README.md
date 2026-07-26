# Direct conditional density-ratio estimation for SCAR

This repository implements a PyTorch density-ratio estimator for multiclass
complementary-label learning under source/target distribution shift. It trains
the `2q` ratios
`p_target(x | bar_y[k]=b) / p_source(x | bar_y[k]=b)` directly, using only
`x`, the binary complementary-label vector, and the source/target domain.
Ordinary labels are not accepted by the ratio-training API.

Four comparable architectures are available through `ratio_arch`:

- `unified`: the original class/state-conditioned shared estimator.
- `separate`: `2q` private backbones, encoders, heads, optimizer states, and
  warm-start parameter states.
- `multihead`: one shared backbone/ratio encoder and `2q` scalar heads.
- `fusion`: one global ratio, `2q` conditional heads, and `2q` learned
  sample-dependent gates combined geometrically in log space.

## Data and model flow

Source and target loaders return `(x, bar_y)`:

- `x`: `[B, ...]`, image or vector input.
- `bar_y`: `[B, q]`, binary complementary-label indicators.

`LabelConditionedRatioEstimator` runs a caller-supplied backbone once:

```text
x -> shared backbone -> h [B,F]
                       + class embedding [q,C]
                       + state embedding [2,S]
                       -> shared ratio head -> logits [B,q,2]
```

`forward_selected(x, bar_y)` gathers state `bar_y[i,k]` and returns `[B,q]`.
The `multihead` alternative still shares one backbone but directly emits
`[B,q,2]`.

The domain classifier uses source label `d=0` and target label `d=1`. For each
branch,

```text
log r_{k,z}(x) = logit_{k,z}(x) + log(pi_source(k,z)/pi_target(k,z))
r_{k,z}(x)     = exp(clamp(log r_{k,z}(x)))
```

The implementation clips log-ratios and ratios, performs the exponential in
float32, and replaces non-finite results safely. With genuinely balanced
source/target sampling, set `balanced_domain_sampling` to `true`, making the
prior correction exactly one.

## Train

Run the synthetic example:

```bash
python train.py --config configs/ratio_synthetic.json --device auto
```

Run each direct-conditional architecture with the same configuration/data seed:

```bash
python train.py --config configs/conditional_synthetic.json ratio_arch=separate
python train.py --config configs/conditional_synthetic.json ratio_arch=multihead
python train.py --config configs/conditional_synthetic.json ratio_arch=fusion
```

The fair multi-seed architecture/no-IW/oracle comparison is:

```bash
python scripts/run_ratio_ablation.py \
  --config configs/conditional_synthetic.json \
  --output-dir results/conditional_ratio_ablation
```

Or the short demo:

```bash
python demo_synthetic.py
```

The JSON file configures `num_classes`, `feature_dim`, embedding dimensions,
head dimensions, `conditioned|multihead`, branch balancing, support threshold,
weight bounds, prior behavior, ratio clipping, self-normalization, joint
training guard, optimizer, data, and runtime settings.

To reuse an image or project-specific backbone:

```python
from train import build_estimator

backbone = ExistingBackbone(...)  # must return [B, config["feature_dim"]]
ratio_estimator = build_estimator(config, backbone=backbone)
```

If an existing loader returns dictionaries, the trainer supports configurable
`x_key` and `bar_y_key`. Other formats can be adapted in
`ratio_estimation/data.py` without changing the model.

## Importance weights and downstream risk

```python
weights = ratio_estimator.compute_importance_weights(
    x,
    bar_y,
    detach=True,
    clip_min=None,
    clip_max=None,
    self_normalize=False,
)  # [B,q]

classwise_loss = complementary_loss(classifier(x), bar_y)  # [B,q]
final_loss = (weights * classwise_loss).mean()
```

Each class uses its own `r[k,bar_y_k](x)`. Do not reduce weights to `[B]`
before multiplying class-wise loss. `weighted_classwise_risk` provides this
integration and detaches ratios by default. Even if `joint_training=True`, it
requires explicit `allow_ratio_gradients=True` before classifier gradients can
flow into ratios, guarding against degenerate weight shrinkage.

Optional self-normalization uses `scatter_add` to divide each observed
`(k,z)` branch by its own batch mean. It is disabled by default because it
changes the raw importance-weighted objective.

The new conditional experiment configuration enables it by default. Processing
is finite-check, clipping, then independent `(k,b)` source-branch
normalization. It records mean/std/min/max, 5/50/95% quantiles, clipping rate,
ESS, source/target counts, branch objective, and skipped updates. A branch with
fewer than `ratio.min_condition_samples` observations in either domain is
skipped for that iteration. Its parameters remain warm-started; until the first
valid update, classifier weighting falls back to one.

## Branch balancing

- `none`: average all supported selected entries.
- `inverse_frequency`: inverse branch-frequency weights with
  `max_branch_weight` clipping.
- `balanced_bce` (default): average source and target BCE within each branch,
  then average supported branches.

`compute_branch_statistics` returns source/target `[q,2]` counts, configures
prior correction, warns about insufficient support, and excludes unsupported
branches from loss. It never builds branch-specific datasets.

Because `balanced_bce` normalizes source and target separately inside every
branch, its effective discriminator prior is 1:1 and the correction is
automatically one. Count-based prior correction is used for unbalanced
`none`/`inverse_frequency` training unless balanced sampling is declared.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover shapes, positivity/finite values, identical distributions, a known
conditional shift, one-backbone-call vectorization, gradients, missing
branches, prior correction, self-normalization, batch adaptation, and
class-wise risk integration.

They additionally cover all four architecture smoke runs, exact state
selection, strict conditional masks, independent optimizer/parameter state,
learned fusion gates, ordinary-label leakage traps, empty-branch fallback, and
classifier-to-ratio gradient detachment.

## Ablations and baselines

Run the multi-seed comparison:

```bash
python -m experiments.ablation \
  --config configs/ablation_synthetic.json \
  --output-dir outputs/ablation \
  --device auto \
  --seeds 7,17,29
```

The suite compares the proposed conditioned estimator against `multihead`,
removed class/state conditioning, removed prior correction, all branch
balancing modes, a pooled marginal-ratio baseline, unit weights, and the known
synthetic oracle. It reports conditional log-ratio RMSE/MAE/correlation,
branch-wise feature-moment RMSE, source ratio calibration, ESS, parameter
count, runtime, and discriminator validation metrics. Outputs are
`raw_results.json`, `summary.csv`, and `REPORT.md`.

Setting `class_embedding_dim` or `state_embedding_dim` to zero explicitly
removes that conditioning input for controlled ablations.

## Wisteria

`wisteria/job_ratio_estimator.sh` loads the verified PyTorch/CUDA modules,
requires a visible GPU, runs the complete unit test suite, and starts
`train.py`. Submit from the repository root with:

```bash
bash wisteria/submit.sh
```

Outputs are written to `outputs/wisteria/`; scheduler and training logs go to
the job output and `logs/`.

Submit the three-seed ablation/baseline suite with:

```bash
bash wisteria/submit_ablation.sh
```

It runs both the mixed covariate-shift and heterogeneous branch-shift
scenarios. Reports are written to `outputs/ablation_mixed/REPORT.md` and
`outputs/ablation_branch_shift/REPORT.md`.

### Original SCAR baseline comparison

When the original SCARCE repository is available, train the frozen neural
ratio estimator and feed its `[B,q]` weights into the original
`combined_adiw_scar_loss`:

```bash
python -m experiments.scarce_ratio_comparison \
  --scarce-repo /path/to/SCARCE \
  --ratio-arch fusion \
  --dataset mnist \
  --shift rotation \
  --seed 0 \
  --ratio-epochs 10 \
  --classifier-epochs 200 \
  --device cuda
```

The bridge reuses the original MNIST dual-domain loader, MLP feature layer,
SCAR branch risk, classifier, optimizer, and target evaluation. The Wisteria
launcher submits matched seeds 0--4 for the requested architectures:

```bash
bash wisteria/submit_scarce_ratio_comparison.sh
```

To submit selected architectures while keeping the same seed launcher:

```bash
RATIO_ARCHS="separate multihead fusion" \
  bash wisteria/submit_scarce_ratio_comparison.sh
```

### Multi-GPU debugging

Aquarius has eight A100 GPUs per node, while its shared queues accept at most
four GPUs per job. The multi-GPU launcher maps one independent
`(ratio_arch, seed)` task to one GPU and defaults to the complete 20-task
matrix. It requests five four-GPU jobs, so no GPU is reserved without a task:

```bash
SCARCE_REPO=/path/to/SCARCE \
RUN_MODE=debug \
bash wisteria/submit_scarce_ratio_multigpu.sh
```

Debug mode uses one ratio epoch, two classifier epochs, and 256 target weak
examples. Inspect scheduler and per-task state with:

```bash
bash wisteria/monitor_scarce_multigpu.sh
```

After all 20 `.ok` files are present and no `.failed` file remains, submit the
full 10/200-epoch experiment with the same GPU layout:

```bash
SCARCE_REPO=/path/to/SCARCE \
RUN_MODE=full \
bash wisteria/submit_scarce_ratio_multigpu.sh
```

Use `GPUS_PER_JOB=1|2|4`, `SEEDS`, or `RATIO_ARCHS` to reduce the matrix.
`DRY_RUN=1` generates the PJM scripts without submitting them.

`experiments/summarize_scarce_methods.py` combines the new JSON records with
the original `trainSCAR`, `testSCAR`, pooled/`ftSCAR`, and ADIW-SCAR table.
The completed Wisteria results and interpretation are recorded in
`ABLATION_RESULTS.md`.

### Matched next-round diagnostics

The next-round runner uses the JSON-compatible YAML configuration in
`configs/next_round.yaml`. It provides matched `no_iw`, `pooled_scar`, `adiw`,
`unified`, `multihead`, `fusion`, `separate_full`, and
`separate_matched_capacity` tasks. List or run one deterministic task with:

```bash
python scripts/run_next_round.py \
  --config configs/next_round.yaml \
  --phase smoke \
  --experiment-mode smoke \
  --list-tasks

python scripts/run_next_round.py \
  --config configs/next_round.yaml \
  --phase smoke \
  --experiment-mode smoke \
  --task-index 0 \
  --scarce-repo /path/to/SCARCE \
  --results-root results/next_round_smoke \
  --device cuda
```

Every run writes `config.yaml`, `summary.json`, a checkpoint, classifier and
per-class trajectories, raw/clipped/normalized/classifier-used ratio
statistics, discriminator calibration, and fusion gate diagnostics under:

```text
results/<dataset>/<setting>/<method>/seed_<seed>/
```

Submit the bounded smoke matrix on Wisteria using every task-bearing GPU:

```bash
PHASE=smoke EXPERIMENT_MODE=smoke GPUS_PER_JOB=4 \
  RESULTS_ROOT=results/next_round_smoke \
  bash wisteria/submit_next_round.sh
```

After smoke succeeds, Phase A1 contains 90 diagnostic runs. The first rotation
block is the matched 30-degree six-method baseline; add the five
`A0_separate_standard` tasks to complete the seven-method main table.
`OFFSET` and `MAX_JOBS` keep submissions within the Aquarius project limit:

```bash
PHASE=A1 EXPERIMENT_MODE=diagnostic OFFSET=0 MAX_JOBS=16 \
  RESULTS_ROOT=results/next_round_phase_a \
  bash wisteria/submit_next_round.sh

PHASE=A0_separate_standard EXPERIMENT_MODE=diagnostic \
  RESULTS_ROOT=results/next_round_phase_a \
  bash wisteria/submit_next_round.sh
```

Summarize completed runs and generate the seven requested CSV tables plus 14
diagnostic figures with:

```bash
python scripts/summarize_next_round.py \
  --results results/next_round_phase_a
```

## Assumptions and limitations

- Ratios are only identifiable for branches with both source and target
  support; unsupported branches are reported and excluded.
- Count-based correction assumes loader counts represent the discriminator's
  effective sampling distribution. Pass explicit priors when a custom sampler
  changes that distribution.
- The bundled dataset is synthetic. Real image/vector loaders should preserve
  the `(x, bar_y)` contract, and a custom backbone must return `[B,F]`.
- MNIST, FashionMNIST, and CIFAR-10 classifier experiments use the existing
  SCARCE bridge (`--dataset mnist|fashionmnist|cifar10`), so that repository
  must be provided with `--scarce-repo`. Its data module remains responsible
  for the exact input-output-relation and support-shift implementations.
- Source and target are forwarded separately, so each domain batch invokes the
  shared backbone once; there is never a per-class or per-state backbone call.
