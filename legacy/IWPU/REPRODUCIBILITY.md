# Reproducibility statement

## Classification of this project

This repository targets a high-fidelity, from-paper reimplementation. It is not
an exact official reproduction.

That conclusion is based on primary artifacts. The PMLR asset directory
contains only the paper PDF. In the PDF's reproducibility checklist, the
authors mark anonymized source code as unavailable and mark the code, data, and
instructions needed to reproduce the main results as unavailable because the
code is proprietary. The authors do provide mathematical derivations,
Algorithm 1, task construction, model families, hyperparameter grids, and
appendix results.

Consequently, this project can test whether the reported idea and broad trends
can be recovered. It cannot establish bitwise identity, checkpoint identity,
identical sample membership, or exact equality with every published mean.

## What the paper specifies

### Data protocol

- Seven main tasks: MNIST, Fashion-MNIST, and CIFAR-10 each under input-output
  relation and support shifts, plus TableShift Diabetes under a race shift.
- One appendix task: TableShift Food Stamps under a geographic-region shift.
- Source training PU counts: 1,000 positives and 5,000 unlabeled.
- Target training PU counts: `(10, 50)`, `(20, 100)`, or `(40, 200)`, ordered as
  `(positive, unlabeled)`.
- Target validation counts: 20 positives and 100 unlabeled.
- Target evaluation counts: 2,500 positives and 2,500 negatives.
- Source and target class priors are both 0.5 in the main comparison.
- Train, validation, and evaluation sets do not overlap.
- Each target-size condition is repeated ten times with changing random seeds.

### Image-task label construction

The positive and negative class lists below remove ambiguity from the paper's
verbal description.

| Task | Source positive | Source negative | Target positive | Target negative |
|---|---|---|---|---|
| MNIST IO | 0,2,5,7,9 | 1,3,4,6,8 | 1,3,5,7,9 | 0,2,4,6,8 |
| MNIST support | 1,3,5 | 0,2,4 | 5,7,9 | 4,6,8 |
| FMNIST IO | 0,2,7,8,9 | 1,3,4,5,6 | 1,5,7,8,9 | 0,2,3,4,6 |
| FMNIST support | 1,5,7 | 0,2,3 | 7,8,9 | 3,4,6 |
| CIFAR-10 IO | 2,3,8,9 | 0,1,4,5,6,7 | 0,1,8,9 | 2,3,4,5,6,7 |
| CIFAR-10 support | 0,1,8 | 2,3,4,5 | 1,8,9 | 4,5,6,7 |

For IO tasks, these source lists are obtained by applying the paper's stated
label flips to the target definition. For support tasks, only the listed
classes belong to each domain.

### Optimization and selection

- PyTorch implementation, sigmoid loss, Adam optimizer.
- Absolute-value risk correction is stated for all comparison methods.
- Source and target mini-batch sizes are both 256.
- Maximum training length is 200 epochs.
- Classifier/feature learning rate is `1e-4`.
- Importance-weight-model learning rate is `1e-3`.
- Target-validation empirical PU risk selects hyperparameters and the
  early-stopping checkpoint.
- Relative density-ratio parameter `alpha`: `0.1, 0.5, 0.9` for IWPU and GIW.
- Source/target weighting parameter `beta`: `0.1, 0.3, 0.5, 0.7, 0.9` for IWPU,
  GIW, mtPU, mtsPU, and DAPU. FoodStamp additionally evaluates `0.0`.
- MMD coefficient: `1, 1e-1, 1e-2, 1e-3` for DAPU and UDAPU.
- DAPU/UDAPU use five RBF-kernel mixtures.

### Model families

- MNIST, Fashion-MNIST, and Diabetes use a three-layer ReLU feed-forward
  feature extractor. The stated node size is 128 for the image tasks and 32 for
  Diabetes.
- CIFAR-10 uses two convolutional blocks: 6 then 16 filters, 5x5 convolution,
  ReLU, and 2x2 max pooling, followed by fully connected dimensions 120 and 84.
- The classifier `u` is a one-layer feed-forward network.
- The importance model `v` is a two-layer feed-forward network with output
  `(1 / alpha) * sigmoid(.)`.
- Comparison methods share the same classifier architecture; mtPU uses two
  classifier heads and GIW uses the same importance-model family as IWPU.

## Local assumptions introduced by this project

Every assumption must be preserved in resolved run metadata.

| Assumption | Local choice | Reason |
|---|---|---|
| seed identities | integers 0 through 9 | paper gives only the count of seeds |
| early-stopping patience | 20 epochs | patience is not reported |
| early-stopping tie break | earliest minimum validation PU risk | not reported |
| Adam betas/epsilon/weight decay | framework defaults; no weight decay | not reported |
| image preprocessing | tensor conversion and scaling to `[0,1]`; no augmentation | not reported |
| raw image split policy | official train split for PU train/validation, official test split for evaluation | not reported |
| sample selection | seeded, without replacement, disjoint partitions | membership and sampler are unavailable |
| TableShift implementation | official pinned engineering commit | paper gives no commit or package version |
| tabular preprocessing | preserve TableShift's own transformed feature values; no additional source/target min-max fit | paper gives only final feature dimensions |
| validation selection | choose the minimum risk, then deterministic grid-order tie break | tie handling is not reported |
| deterministic mode | requested where PyTorch supports it | paper does not report deterministic settings |

The config annotates these values with `source: local_assumption`; they must not
be silently presented as paper hyperparameters.

## Missing information that prevents strict reproduction

- proprietary official implementation;
- original random seed values and exact sampled row/image identities;
- complete dependency versions, CUDA/cuDNN versions, and PyTorch version;
- initialization details and precise alternating-update scheduling beyond the
  pseudocode;
- image normalization, augmentation, and raw train/test-pool policy;
- TableShift version/commit and exact transformed feature pipeline;
- early-stopping patience, checkpoint tie-breaking, and hyperparameter-tie
  resolution;
- Adam beta, epsilon, and weight-decay settings;
- exact construction and thresholding of GIW pseudo-PN labels;
- RBF bandwidth construction for the five-kernel MMD mixture;
- whether all reported grids were shared identically across every task;
- original hardware model details beyond Intel Xeon and NVIDIA A100.

These gaps mean exact table matching is not a sound acceptance criterion.

## Acceptance criteria

Use three levels of verification:

1. **Implementation correctness**: unit tests validate PU-risk identities,
   density-ratio bounds, gradient isolation between alternating updates, method
   routing, disjoint sampling, and deterministic seed handling.
2. **Protocol fidelity**: resolved configs match the paper counts, task label
   maps, architectures, grids, and aggregation over ten seeds.
3. **Empirical agreement**: compare mean accuracy and standard deviation with
   the paper while reporting absolute deltas and uncertainty. Treat direction
   of improvement and confidence intervals as stronger evidence than exact
   decimal equality.

Do not tune against the final target test labels. Hyperparameters and stopping
must use only the target validation PU risk described by the paper.

## Run provenance checklist

Each run record should include:

- git commit and dirty-worktree status;
- full resolved YAML;
- task, method, target size, alpha/beta/lambda values, and seed;
- dataset URL/version, TableShift commit, and cached-file checksum where
  possible;
- train/validation/test sample indices or hashes;
- Python, PyTorch, torchvision, NumPy, CUDA, cuDNN, GPU, and driver versions;
- deterministic flags and any nondeterministic-operation warnings;
- epoch-wise train objectives and target validation PU risk;
- selected epoch, final target accuracy, elapsed time, and peak GPU memory;
- failure/retry status and any reduced-batch or gradient-accumulation fallback.

The grid report should aggregate exactly ten successful, distinct seeds per
cell. Failed seeds must remain visible rather than being silently omitted.

## Hardware interpretation

The paper used an NVIDIA A100 on Linux. An RTX 3070 run is an independent
hardware reproduction. It should preserve mathematical batch size 256 whenever
possible, but the device has materially less memory and throughput. If gradient
accumulation is used, record both micro-batch and effective batch size. If batch
size itself changes, report that run as a protocol deviation.

CUDA kernels and floating-point reduction order can differ across A100 and RTX
3070, so setting a seed does not imply cross-GPU identity. Accuracy should be
reported over ten seeds; paper wall-clock results should not be used as an RTX
3070 performance target.

## Primary-source inventory

- Paper record: https://proceedings.mlr.press/v258/kumagai25a.html
- Paper and integrated appendices:
  https://raw.githubusercontent.com/mlresearch/v258/main/assets/kumagai25a/kumagai25a.pdf
- PMLR assets: https://github.com/mlresearch/v258/tree/main/assets/kumagai25a
- OpenReview: https://openreview.net/forum?id=CTYgrczjj2
- Author publication page: https://www.kecl.ntt.co.jp/as/members/iwata/
- MNIST: https://yann.lecun.org/exdb/mnist/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- TableShift code: https://github.com/mlfoundations/tableshift
- TableShift task definitions: https://tableshift.org/datasets.html
