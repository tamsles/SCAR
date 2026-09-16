"""Training engine for IW-PU and the paper's executable baselines.

The proposed method follows Algorithm 1 exactly: each stochastic iteration
first updates ``v`` with ``h`` detached, then updates ``h`` and ``u`` with the
new importance weights detached.  Within each domain batch, positive and
unlabeled examples are sampled independently with replacement, because the
target-positive set can be much smaller than the reported batch size.  The
paper defines ``B_tr`` and ``B_te`` as source and target PU-batch sizes, not
as the size of each P/U component.  Because it does not report the component
allocation, this implementation splits each domain batch equally between P
and U (128+128 for the paper's batch size 256).

The paper does not specify which samples enter its MMD term.  DAPU and UDAPU
therefore apply :func:`iwpu.mmd.multi_rbf_mmd` to source- and target-unlabeled
features; this is an explicit reproduction assumption.  It also omits the
phase lengths and optimizer-state handling for ftPU and 2step.  Here each
pretraining phase uses ``pretrain_epochs`` (or ``max_epochs`` when unset), and
ftPU starts a fresh Adam optimizer for target fine-tuning.

GIW is necessarily approximate because its official pseudo-labeling code is
not available.  We independently pretrain trPU and tePU classifiers, predict
every example in the corresponding concatenated P+U set (including observed
positives rather than forcing their pseudo-labels), and then apply supervised
dynamic relative-ratio estimation.  This choice follows the paper's statement
that the classifiers are applied to the PU data without adding an unreported
positive-label override.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn

from .losses import importance_ratio_loss, pu_risk, sigmoid_loss
from .mmd import multi_rbf_mmd
from .models import IWPUModel, build_model
from .utils import seed_everything


@dataclass(slots=True)
class TrainSettings:
    """Hyperparameters for one method/seed run.

    ``steps_per_epoch`` is deliberately explicit: the paper reports epochs and
    batch sizes but does not define an epoch when tiny target sets are sampled
    with replacement.
    """

    steps_per_epoch: int = 1
    alpha: float = 0.5
    beta: float = 0.5
    mmd_lambda: float = 1.0
    max_epochs: int = 200
    patience: int = 20
    source_batch_size: int = 256
    target_batch_size: int = 256
    lr_classifier: float = 1.0e-4
    lr_importance: float = 1.0e-3
    device: str | torch.device = "cpu"
    seed: int = 0
    correct_iw: bool = True
    correct_cl: bool = True
    dataset: str | None = None
    input_dim: int | None = None
    pi_src: float | None = None
    pi_tgt: float | None = None
    pretrain_epochs: int | None = None
    min_delta: float = 0.0
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        for name in (
            "steps_per_epoch",
            "max_epochs",
            "source_batch_size",
            "target_batch_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.source_batch_size < 2 or self.target_batch_size < 2:
            raise ValueError("source_batch_size and target_batch_size must be at least 2")
        if (
            not isinstance(self.patience, Integral)
            or isinstance(self.patience, bool)
            or self.patience < 0
        ):
            raise ValueError("patience must be a non-negative integer")
        if self.pretrain_epochs is not None and (
            not isinstance(self.pretrain_epochs, Integral)
            or isinstance(self.pretrain_epochs, bool)
            or self.pretrain_epochs < 0
        ):
            raise ValueError("pretrain_epochs must be a non-negative integer or None")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(self.deterministic_algorithms, bool):
            raise ValueError("deterministic_algorithms must be a bool")
        _probability(self.alpha, "alpha", open_zero=True)
        _probability(self.beta, "beta")
        for name in ("lr_classifier", "lr_importance", "adam_eps"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be a positive finite scalar")
        for name in ("mmd_lambda", "min_delta", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite scalar")
        if (
            len(self.adam_betas) != 2
            or any(not 0.0 <= float(beta) < 1.0 for beta in self.adam_betas)
        ):
            raise ValueError("adam_betas must contain two values in [0, 1)")


def _probability(value: float, name: str, *, open_zero: bool = False) -> float:
    if not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    valid = 0.0 < value <= 1.0 if open_zero else 0.0 <= value <= 1.0
    if not valid:
        interval = "(0, 1]" if open_zero else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return value


def _canonical_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    key = method.strip().lower().replace("-", "_").replace(" ", "")
    aliases = {
        "ours": "iwpu",
        "iwpu": "iwpu",
        "tepu": "tepu",
        "trpu": "trpu",
        "ftpu": "ftpu",
        "mtpu": "mtpu",
        "mtspu": "mtspu",
        "dapu": "dapu",
        "udapu": "udapu",
        "giw": "giw",
        "2step": "two_step",
        "2_step": "two_step",
        "twostep": "two_step",
        "two_step": "two_step",
    }
    if key not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"unsupported method {method!r}; expected one of {supported}")
    return aliases[key]


def _resolve_device(specification: str | torch.device) -> torch.device:
    if isinstance(specification, torch.device):
        return specification
    if not isinstance(specification, str):
        raise TypeError("device must be a string or torch.device")
    if specification.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def _metadata(bundle: Any) -> dict[str, Any]:
    value = getattr(bundle, "metadata", {})
    return value if isinstance(value, dict) else {}


def _resolve_dataset(bundle: Any, settings: TrainSettings) -> str:
    if settings.dataset:
        return settings.dataset
    metadata = _metadata(bundle)
    dataset = metadata.get("dataset") or metadata.get("task")
    if isinstance(dataset, str) and dataset.lower() not in {"tableshift", "tabular"}:
        return dataset
    identifier = str(metadata.get("tableshift_id", "")).lower()
    if "food" in identifier:
        return "foodstamp"
    if "diabetes" in identifier:
        return "diabetes"
    input_dim = int(getattr(bundle, "input_dim"))
    if input_dim == 239:
        return "foodstamp"
    if input_dim == 142:
        return "diabetes"
    raise ValueError(
        "could not infer dataset from bundle.metadata; set TrainSettings.dataset"
    )


def _resolve_priors(bundle: Any, settings: TrainSettings) -> tuple[float, float]:
    metadata = _metadata(bundle)
    pi_src = settings.pi_src
    pi_tgt = settings.pi_tgt
    if pi_src is None:
        pi_src = metadata.get("pi_src", metadata.get("source_positive_prior", 0.5))
    if pi_tgt is None:
        pi_tgt = metadata.get("pi_tgt", metadata.get("target_positive_prior", 0.5))
    return _probability(pi_src, "pi_src"), _probability(pi_tgt, "pi_tgt")


_TRAINING_BUNDLE_TENSORS = (
    "src_pos",
    "src_unl",
    "tgt_pos",
    "tgt_unl",
    "val_pos",
    "val_unl",
)


def _validate_training_bundle(bundle: Any) -> None:
    for name in _TRAINING_BUNDLE_TENSORS:
        value = getattr(bundle, name, None)
        if not torch.is_tensor(value):
            raise TypeError(f"bundle.{name} must be a torch.Tensor")
        if value.shape[0] == 0:
            raise ValueError(f"bundle.{name} must not be empty")


def _validate_test_bundle(bundle: Any) -> None:
    for name in ("test_x", "test_y"):
        value = getattr(bundle, name, None)
        if not torch.is_tensor(value):
            raise TypeError(f"bundle.{name} must be a torch.Tensor")
        if value.shape[0] == 0:
            raise ValueError(f"bundle.{name} must not be empty")
    if bundle.test_x.shape[0] != bundle.test_y.shape[0]:
        raise ValueError("bundle.test_x and bundle.test_y must have equal length")
    labels = bundle.test_y
    if torch.any((labels != -1) & (labels != 1)):
        raise ValueError("bundle.test_y must use -1/+1 encoding")


def _replacement_batch(
    values: Tensor,
    size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Draw a replacement batch and return its CPU source indices."""

    indices = torch.randint(values.shape[0], (size,), generator=generator)
    return values.index_select(0, indices).to(device), indices


def _pu_component_sizes(domain_batch_size: int) -> tuple[int, int]:
    """Split one reported domain-level PU batch approximately equally.

    Algorithm 1 samples ``B_tr`` observations from the source PU data and
    ``B_te`` observations from the target PU data, but the paper does not say
    how those observations are allocated between P and U.  Separate component
    samples are required by the displayed empirical PU risks.  The equal split
    is therefore an explicit local assumption; an odd total gives the extra
    observation to U.
    """

    if domain_batch_size < 2:
        raise ValueError("domain_batch_size must be at least 2")
    positive = domain_batch_size // 2
    return positive, domain_batch_size - positive


def _draw_four(
    bundle: Any,
    settings: TrainSettings,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    src_pos_size, src_unl_size = _pu_component_sizes(settings.source_batch_size)
    tgt_pos_size, tgt_unl_size = _pu_component_sizes(settings.target_batch_size)
    src_pos, src_pos_idx = _replacement_batch(
        bundle.src_pos, src_pos_size, generator, device
    )
    src_unl, src_unl_idx = _replacement_batch(
        bundle.src_unl, src_unl_size, generator, device
    )
    tgt_pos, tgt_pos_idx = _replacement_batch(
        bundle.tgt_pos, tgt_pos_size, generator, device
    )
    tgt_unl, tgt_unl_idx = _replacement_batch(
        bundle.tgt_unl, tgt_unl_size, generator, device
    )
    return (
        src_pos,
        src_unl,
        tgt_pos,
        tgt_unl,
        src_pos_idx,
        src_unl_idx,
        tgt_pos_idx,
        tgt_unl_idx,
    )


def _adam(parameters: Any, lr: float, settings: TrainSettings) -> torch.optim.Adam:
    return torch.optim.Adam(
        parameters,
        lr=lr,
        betas=settings.adam_betas,
        eps=settings.adam_eps,
        weight_decay=settings.weight_decay,
    )


def _plain_pu(
    model: IWPUModel,
    positive: Tensor,
    unlabeled: Tensor,
    pi: float,
    correct: bool,
    *,
    source_head: bool = False,
) -> Tensor:
    positive_features = model.encode(positive)
    unlabeled_features = model.encode(unlabeled)
    head: nn.Module = getattr(model, "u_source") if source_head else model.u
    return pu_risk(
        head(positive_features), head(unlabeled_features), pi, correct=correct
    )


def _weighted_source_pu(
    model: IWPUModel,
    positive: Tensor,
    unlabeled: Tensor,
    pi_src: float,
    correct: bool,
    positive_weights: tuple[Tensor, Tensor],
    unlabeled_weights: Tensor,
) -> Tensor:
    return pu_risk(
        model(positive),
        model(unlabeled),
        pi_src,
        correct=correct,
        pos_weights=positive_weights,
        unl_weights=unlabeled_weights,
    )


def _importance_objective(
    model: IWPUModel,
    src_pos: Tensor,
    src_unl: Tensor,
    tgt_pos: Tensor,
    tgt_unl: Tensor,
    pi_src: float,
    pi_tgt: float,
    settings: TrainSettings,
    *,
    detach_features: bool,
) -> Tensor:
    """Evaluate Eq. (19), optionally stopping its feature gradients."""

    return importance_ratio_loss(
        model.importance(tgt_pos, +1, detach_features=detach_features),
        model.importance(tgt_unl, -1, detach_features=detach_features),
        model.importance(tgt_pos, -1, detach_features=detach_features),
        model.importance(src_pos, +1, detach_features=detach_features),
        model.importance(src_unl, -1, detach_features=detach_features),
        model.importance(src_pos, -1, detach_features=detach_features),
        pi_src,
        pi_tgt,
        settings.alpha,
        correct=settings.correct_iw,
    )


def _importance_update(
    model: IWPUModel,
    optimizer: torch.optim.Optimizer,
    src_pos: Tensor,
    src_unl: Tensor,
    tgt_pos: Tensor,
    tgt_unl: Tensor,
    pi_src: float,
    pi_tgt: float,
    settings: TrainSettings,
) -> Tensor:
    """Perform Algorithm 1 lines 4-5, stopping all gradients through ``h``."""

    optimizer.zero_grad(set_to_none=True)
    loss = _importance_objective(
        model,
        src_pos,
        src_unl,
        tgt_pos,
        tgt_unl,
        pi_src,
        pi_tgt,
        settings,
        detach_features=True,
    )
    _ensure_finite(loss, "importance-ratio loss")
    loss.backward()
    optimizer.step()
    return loss.detach()


def _dynamic_iwpu_loss(
    model: IWPUModel,
    src_pos: Tensor,
    src_unl: Tensor,
    tgt_pos: Tensor,
    tgt_unl: Tensor,
    pi_src: float,
    pi_tgt: float,
    settings: TrainSettings,
) -> Tensor:
    """Algorithm 1 lines 6-7 with the current weights fully detached."""

    source_risk = _weighted_source_pu(
        model,
        src_pos,
        src_unl,
        pi_src,
        settings.correct_cl,
        (
            model.importance(src_pos, +1, detach_output=True),
            model.importance(src_pos, -1, detach_output=True),
        ),
        model.importance(src_unl, -1, detach_output=True),
    )
    target_risk = _plain_pu(
        model, tgt_pos, tgt_unl, pi_tgt, settings.correct_cl
    )
    return settings.beta * target_risk + (1.0 - settings.beta) * source_risk


def _classifier_objective(
    method: str,
    model: IWPUModel,
    src_pos: Tensor,
    src_unl: Tensor,
    tgt_pos: Tensor,
    tgt_unl: Tensor,
    pi_src: float,
    pi_tgt: float,
    settings: TrainSettings,
    fixed_weights: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    source_risk: Tensor
    target_risk: Tensor
    if method == "iwpu":
        return _dynamic_iwpu_loss(
            model, src_pos, src_unl, tgt_pos, tgt_unl, pi_src, pi_tgt, settings
        )
    if method == "tepu" or method == "ftpu":
        return _plain_pu(model, tgt_pos, tgt_unl, pi_tgt, settings.correct_cl)
    if method == "trpu":
        return _plain_pu(model, src_pos, src_unl, pi_src, settings.correct_cl)
    if method == "two_step":
        if fixed_weights is None:
            raise RuntimeError("two_step requires fixed source weights")
        source_risk = _weighted_source_pu(
            model,
            src_pos,
            src_unl,
            pi_src,
            settings.correct_cl,
            (fixed_weights[0], fixed_weights[1]),
            fixed_weights[2],
        )
        target_risk = _plain_pu(
            model, tgt_pos, tgt_unl, pi_tgt, settings.correct_cl
        )
        return settings.beta * target_risk + (1.0 - settings.beta) * source_risk

    source_risk = _plain_pu(
        model,
        src_pos,
        src_unl,
        pi_src,
        settings.correct_cl,
        source_head=(method == "mtpu"),
    )
    if method == "udapu":
        mmd = multi_rbf_mmd(model.encode(src_unl), model.encode(tgt_unl))
        return source_risk + settings.mmd_lambda * mmd

    target_risk = _plain_pu(
        model, tgt_pos, tgt_unl, pi_tgt, settings.correct_cl
    )
    combined = settings.beta * target_risk + (1.0 - settings.beta) * source_risk
    if method == "dapu":
        # Unlabeled-to-unlabeled MMD is an explicit choice where the paper is
        # underspecified; it avoids leaking target-positive labels into MMD.
        combined = combined + settings.mmd_lambda * multi_rbf_mmd(
            model.encode(src_unl), model.encode(tgt_unl)
        )
    return combined


def _ensure_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value):
        raise FloatingPointError(f"{name} became non-finite")


def _state_on_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validation_risk(
    model: IWPUModel,
    bundle: Any,
    pi_tgt: float,
    correct: bool,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        risk = pu_risk(
            model(bundle.val_pos.to(device)),
            model(bundle.val_unl.to(device)),
            pi_tgt,
            correct=correct,
        )
    model.train(was_training)
    return float(risk.detach().cpu())


def _test_accuracy(model: IWPUModel, bundle: Any, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(bundle.test_x.to(device))
        labels = bundle.test_y.to(device).reshape_as(logits)
        predictions = torch.where(logits >= 0, 1.0, -1.0).to(labels.dtype)
        accuracy = (predictions == labels).float().mean()
    model.train(was_training)
    return float(accuracy.detach().cpu())


def evaluate_test_accuracy(
    model: IWPUModel,
    bundle: Any,
    device: str | torch.device,
) -> float:
    """Evaluate a selected model on the held-out test set exactly once.

    Keeping this operation separate lets experiment runners tune exclusively
    on validation PU risk without computing or exposing candidate test scores.
    """

    if not isinstance(model, IWPUModel):
        raise TypeError("model must be an IWPUModel")
    _validate_test_bundle(bundle)
    return _test_accuracy(model, bundle, _resolve_device(device))


def _run_ftpu_pretraining(
    model: IWPUModel,
    bundle: Any,
    pi_src: float,
    settings: TrainSettings,
    generator: torch.Generator,
    device: torch.device,
    history: list[dict[str, Any]],
) -> int:
    epochs = settings.max_epochs if settings.pretrain_epochs is None else settings.pretrain_epochs
    positive_size, unlabeled_size = _pu_component_sizes(settings.source_batch_size)
    optimizer = _adam(
        list(model.h.parameters()) + list(model.u.parameters()),
        settings.lr_classifier,
        settings,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(settings.steps_per_epoch):
            src_pos, _ = _replacement_batch(
                bundle.src_pos, positive_size, generator, device
            )
            src_unl, _ = _replacement_batch(
                bundle.src_unl, unlabeled_size, generator, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _plain_pu(
                model, src_pos, src_unl, pi_src, settings.correct_cl
            )
            _ensure_finite(loss, "ftPU source-pretraining loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append(
            {
                "phase": "source_pretrain",
                "epoch": epoch,
                "train_loss": total / settings.steps_per_epoch,
                "val_risk": None,
            }
        )
    return epochs


def _run_ratio_pretraining(
    model: IWPUModel,
    bundle: Any,
    pi_src: float,
    pi_tgt: float,
    settings: TrainSettings,
    generator: torch.Generator,
    device: torch.device,
    history: list[dict[str, Any]],
) -> int:
    """Fit the complete ratio model ``m=v([h(x),y])`` for traditional 2step.

    Unlike Algorithm 1's dynamic ratio update, a standalone minimization of
    Eq. (19) has no classifier step that will learn ``h``.  Freezing a random
    ``h`` here would therefore estimate ratios in a random feature space and
    artificially weaken the baseline.  We train both parts of ``m`` using the
    paper's respective feature and importance-model learning rates, then
    materialize the source-example weights before classifier learning.
    """

    epochs = settings.max_epochs if settings.pretrain_epochs is None else settings.pretrain_epochs
    optimizer = _adam(
        [
            {"params": list(model.h.parameters()), "lr": settings.lr_classifier},
            {"params": list(model.v.parameters()), "lr": settings.lr_importance},
        ],
        settings.lr_importance,
        settings,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(settings.steps_per_epoch):
            src_pos, src_unl, tgt_pos, tgt_unl, *_ = _draw_four(
                bundle, settings, generator, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _importance_objective(
                model,
                src_pos,
                src_unl,
                tgt_pos,
                tgt_unl,
                pi_src,
                pi_tgt,
                settings,
                detach_features=False,
            )
            _ensure_finite(loss, "two-step importance-ratio loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append(
            {
                "phase": "ratio_pretrain",
                "epoch": epoch,
                "train_loss": total / settings.steps_per_epoch,
                "val_risk": None,
            }
        )
    return epochs


def _fixed_source_weights(
    model: IWPUModel, bundle: Any, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Materialize 2step weights so later changes to ``h`` cannot alter them."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        positive = bundle.src_pos.to(device)
        unlabeled = bundle.src_unl.to(device)
        result = (
            model.importance(positive, +1).detach(),
            model.importance(positive, -1).detach(),
            model.importance(unlabeled, -1).detach(),
        )
    model.train(was_training)
    return result


def _replacement_labeled_batch(
    examples: Tensor,
    labels: Tensor,
    size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Independently sample a pseudo-labeled domain with replacement."""

    indices = torch.randint(examples.shape[0], (size,), generator=generator)
    return (
        examples.index_select(0, indices).to(device),
        labels.index_select(0, indices).to(device),
    )


def _pretrain_giw_pu_classifier(
    dataset: str,
    input_dim: int,
    positive: Tensor,
    unlabeled: Tensor,
    pi: float,
    domain_batch_size: int,
    phase: str,
    settings: TrainSettings,
    generator: torch.Generator,
    device: torch.device,
    history: list[dict[str, Any]],
) -> tuple[Tensor, Tensor, int]:
    """Fit one independent PU classifier and predict all of its P+U data.

    Observed positives are deliberately *not* forced to ``+1``: the paper says
    GIW creates pseudo-PN data by applying trPU/tePU, and does not report such
    an override.  The returned tensors are on CPU so the temporary classifier
    can be released before the main GIW optimization.
    """

    classifier = build_model(dataset, input_dim, settings.alpha).to(device)
    optimizer = _adam(
        list(classifier.h.parameters()) + list(classifier.u.parameters()),
        settings.lr_classifier,
        settings,
    )
    epochs = settings.max_epochs if settings.pretrain_epochs is None else settings.pretrain_epochs
    positive_size, unlabeled_size = _pu_component_sizes(domain_batch_size)
    for epoch in range(1, epochs + 1):
        classifier.train()
        total = 0.0
        for _ in range(settings.steps_per_epoch):
            positive_batch, _ = _replacement_batch(
                positive, positive_size, generator, device
            )
            unlabeled_batch, _ = _replacement_batch(
                unlabeled, unlabeled_size, generator, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _plain_pu(
                classifier,
                positive_batch,
                unlabeled_batch,
                pi,
                settings.correct_cl,
            )
            _ensure_finite(loss, f"{phase} pseudo-label pretraining loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "train_loss": total / settings.steps_per_epoch,
                "val_risk": None,
            }
        )

    examples = torch.cat((positive, unlabeled), dim=0)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(examples.to(device))
        pseudo_labels = torch.where(logits >= 0, 1.0, -1.0).cpu()
    return examples.cpu(), pseudo_labels, epochs


def _prepare_giw_pseudo_domains(
    dataset: str,
    input_dim: int,
    bundle: Any,
    pi_src: float,
    pi_tgt: float,
    settings: TrainSettings,
    generator: torch.Generator,
    device: torch.device,
    history: list[dict[str, Any]],
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], int]:
    """Create source and target pseudo-PN datasets using independent models."""

    source_x, source_y, source_epochs = _pretrain_giw_pu_classifier(
        dataset,
        input_dim,
        bundle.src_pos,
        bundle.src_unl,
        pi_src,
        settings.source_batch_size,
        "giw_trpu_pretrain",
        settings,
        generator,
        device,
        history,
    )
    target_x, target_y, target_epochs = _pretrain_giw_pu_classifier(
        dataset,
        input_dim,
        bundle.tgt_pos,
        bundle.tgt_unl,
        pi_tgt,
        settings.target_batch_size,
        "giw_tepu_pretrain",
        settings,
        generator,
        device,
        history,
    )
    return (source_x, source_y), (target_x, target_y), source_epochs + target_epochs


def _giw_importance_update(
    model: IWPUModel,
    optimizer: torch.optim.Optimizer,
    source_x: Tensor,
    source_y: Tensor,
    target_x: Tensor,
    target_y: Tensor,
    settings: TrainSettings,
) -> Tensor:
    """Supervised relative joint-ratio update used by the GIW approximation."""

    optimizer.zero_grad(set_to_none=True)
    target_weights = model.importance(
        target_x, target_y, detach_features=True
    )
    source_weights = model.importance(
        source_x, source_y, detach_features=True
    )
    loss = (
        settings.alpha * target_weights.square() - 2.0 * target_weights
    ).mean() + (1.0 - settings.alpha) * source_weights.square().mean()
    _ensure_finite(loss, "GIW joint-ratio loss")
    loss.backward()
    optimizer.step()
    return loss.detach()


def _giw_classifier_objective(
    model: IWPUModel,
    source_x: Tensor,
    source_y: Tensor,
    target_x: Tensor,
    target_y: Tensor,
    settings: TrainSettings,
) -> Tensor:
    """Importance-weighted supervised classifier objective for GIW."""

    source_weights = model.importance(
        source_x, source_y, detach_output=True
    )
    source_risk = (
        source_weights * sigmoid_loss(model(source_x), source_y)
    ).mean()
    target_risk = sigmoid_loss(model(target_x), target_y).mean()
    return settings.beta * target_risk + (1.0 - settings.beta) * source_risk


def train_one(
    bundle: Any,
    method: str,
    settings: TrainSettings,
    *,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Train one IW-PU method and return its selected target-domain model.

    Parameters
    ----------
    bundle:
        A ``PUDataBundle``-like object with ``src_pos``, ``src_unl``,
        ``tgt_pos``, ``tgt_unl``, ``val_pos``, ``val_unl``, ``test_x``,
        ``test_y``, ``input_dim``, and ``metadata`` attributes.
    method:
        One of ``iwpu`` (alias ``ours``), ``tepu``, ``trpu``, ``ftpu``,
        ``mtspu``, ``mtpu``, ``dapu``, ``udapu``, ``giw``, or ``two_step``
        (alias ``2step``).

    Returns
    -------
    dict
        Contains ``model``, ``best_val_risk``, total optimization ``epochs``,
        wall-clock ``runtime`` in seconds, and ``history``.  ``test_accuracy``
        is included only when ``evaluate_test`` is true.  Hyperparameter
        search must pass false and call :func:`evaluate_test_accuracy` only for
        the model selected by validation risk.
        Early stopping and model selection always use corrected or uncorrected
        target-validation PU risk according to ``settings.correct_cl``.
    """

    if not isinstance(settings, TrainSettings):
        raise TypeError("settings must be a TrainSettings instance")
    if not isinstance(evaluate_test, bool):
        raise TypeError("evaluate_test must be a bool")
    _validate_training_bundle(bundle)
    canonical = _canonical_method(method)
    device = _resolve_device(settings.device)
    dataset = _resolve_dataset(bundle, settings)
    input_dim = settings.input_dim or int(getattr(bundle, "input_dim"))
    pi_src, pi_tgt = _resolve_priors(bundle, settings)

    seed_everything(
        int(settings.seed), deterministic=settings.deterministic_algorithms
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(settings.seed))

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(dataset, input_dim, settings.alpha).to(device)
    if canonical == "mtpu":
        # The paper's mtPU shares h but has distinct source/target heads.
        model.u_source = copy.deepcopy(model.u).to(device)  # type: ignore[attr-defined]

    history: list[dict[str, Any]] = []
    phase_epochs = 0
    fixed_full_weights: tuple[Tensor, Tensor, Tensor] | None = None
    giw_source: tuple[Tensor, Tensor] | None = None
    giw_target: tuple[Tensor, Tensor] | None = None

    if canonical == "ftpu":
        phase_epochs += _run_ftpu_pretraining(
            model, bundle, pi_src, settings, generator, device, history
        )
    elif canonical == "two_step":
        phase_epochs += _run_ratio_pretraining(
            model, bundle, pi_src, pi_tgt, settings, generator, device, history
        )
        fixed_full_weights = _fixed_source_weights(model, bundle, device)
    elif canonical == "giw":
        giw_source, giw_target, giw_pretrain_epochs = _prepare_giw_pseudo_domains(
            dataset,
            input_dim,
            bundle,
            pi_src,
            pi_tgt,
            settings,
            generator,
            device,
            history,
        )
        phase_epochs += giw_pretrain_epochs

    classifier_parameters = list(model.h.parameters()) + list(model.u.parameters())
    if canonical == "mtpu":
        classifier_parameters += list(model.u_source.parameters())  # type: ignore[attr-defined]
    classifier_optimizer = _adam(
        classifier_parameters, settings.lr_classifier, settings
    )
    importance_optimizer = (
        _adam(model.v.parameters(), settings.lr_importance, settings)
        if canonical in {"iwpu", "giw"}
        else None
    )

    best_risk = math.inf
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    trained_epochs = 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        total_classifier = 0.0
        total_importance = 0.0
        for _ in range(settings.steps_per_epoch):
            if canonical == "giw":
                if (
                    giw_source is None
                    or giw_target is None
                    or importance_optimizer is None
                ):
                    raise RuntimeError("GIW pseudo domains or optimizer are missing")
                source_x, source_y = _replacement_labeled_batch(
                    giw_source[0],
                    giw_source[1],
                    settings.source_batch_size,
                    generator,
                    device,
                )
                target_x, target_y = _replacement_labeled_batch(
                    giw_target[0],
                    giw_target[1],
                    settings.target_batch_size,
                    generator,
                    device,
                )
                ratio_loss = _giw_importance_update(
                    model,
                    importance_optimizer,
                    source_x,
                    source_y,
                    target_x,
                    target_y,
                    settings,
                )
                total_importance += float(ratio_loss.cpu())

                classifier_optimizer.zero_grad(set_to_none=True)
                classifier_loss = _giw_classifier_objective(
                    model,
                    source_x,
                    source_y,
                    target_x,
                    target_y,
                    settings,
                )
                _ensure_finite(classifier_loss, "GIW classifier loss")
                classifier_loss.backward()
                classifier_optimizer.step()
                total_classifier += float(classifier_loss.detach().cpu())
                continue

            (
                src_pos,
                src_unl,
                tgt_pos,
                tgt_unl,
                src_pos_idx,
                src_unl_idx,
                _,
                _,
            ) = _draw_four(bundle, settings, generator, device)

            if canonical == "iwpu":
                if importance_optimizer is None:
                    raise RuntimeError("missing importance optimizer")
                ratio_loss = _importance_update(
                    model,
                    importance_optimizer,
                    src_pos,
                    src_unl,
                    tgt_pos,
                    tgt_unl,
                    pi_src,
                    pi_tgt,
                    settings,
                )
                total_importance += float(ratio_loss.cpu())

            batch_fixed_weights = None
            if fixed_full_weights is not None:
                pos_indices = src_pos_idx.to(device)
                unl_indices = src_unl_idx.to(device)
                batch_fixed_weights = (
                    fixed_full_weights[0].index_select(0, pos_indices),
                    fixed_full_weights[1].index_select(0, pos_indices),
                    fixed_full_weights[2].index_select(0, unl_indices),
                )

            classifier_optimizer.zero_grad(set_to_none=True)
            classifier_loss = _classifier_objective(
                canonical,
                model,
                src_pos,
                src_unl,
                tgt_pos,
                tgt_unl,
                pi_src,
                pi_tgt,
                settings,
                batch_fixed_weights,
            )
            _ensure_finite(classifier_loss, "classifier loss")
            classifier_loss.backward()
            classifier_optimizer.step()
            total_classifier += float(classifier_loss.detach().cpu())

        trained_epochs = epoch
        val_risk = _validation_risk(
            model, bundle, pi_tgt, settings.correct_cl, device
        )
        record: dict[str, Any] = {
            "phase": "classifier",
            "epoch": phase_epochs + epoch,
            "train_loss": total_classifier / settings.steps_per_epoch,
            "val_risk": val_risk,
        }
        if canonical in {"iwpu", "giw"}:
            record["importance_loss"] = total_importance / settings.steps_per_epoch
        history.append(record)

        improved = val_risk < best_risk - settings.min_delta
        if improved:
            best_risk = val_risk
            best_state = _state_on_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break

    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_memory_bytes: int | None = int(
            torch.cuda.max_memory_allocated(device)
        )
    else:
        peak_cuda_memory_bytes = None
    runtime = time.perf_counter() - started
    result: dict[str, Any] = {
        "model": model,
        "best_val_risk": float(best_risk),
        "epochs": phase_epochs + trained_epochs,
        "runtime": float(runtime),
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "history": history,
        "method": canonical,
        "approximation_due_to_missing_official_code": canonical == "giw",
        "metadata": {
            "approximation_due_to_missing_official_code": canonical == "giw",
            "giw_observed_positives_forced_positive": False if canonical == "giw" else None,
            "batch_protocol": {
                "source_domain_total": settings.source_batch_size,
                "target_domain_total": settings.target_batch_size,
                "source_positive_unlabeled": _pu_component_sizes(
                    settings.source_batch_size
                ),
                "target_positive_unlabeled": _pu_component_sizes(
                    settings.target_batch_size
                ),
                "allocation_source": "local_assumption_equal_split",
            },
            "two_step_ratio_features_trained": canonical == "two_step",
            "two_step_source_weights_materialized": canonical == "two_step",
            "two_step_ratio_phase_epochs": (
                phase_epochs if canonical == "two_step" else None
            ),
            "classifier_phase_epochs": trained_epochs,
        },
    }
    if evaluate_test:
        result["test_accuracy"] = evaluate_test_accuracy(model, bundle, device)
    return result


__all__ = ["TrainSettings", "evaluate_test_accuracy", "train_one"]
