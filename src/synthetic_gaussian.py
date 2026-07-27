"""Ground-truth Gaussian-mixture benchmark for Phase B3."""

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SHIFT_SETTINGS = (
    "no_shift",
    "homogeneous_conditional_shift",
    "heterogeneous_conditional_shift",
    "class_prior_shift_only",
)


def gaussian_parameters(
    setting: str, q: int = 3
) -> Dict[str, np.ndarray]:
    if q != 3:
        raise ValueError("the preregistered benchmark currently uses q=3")
    if setting not in SHIFT_SETTINGS:
        raise ValueError("unknown Gaussian shift: {}".format(setting))
    source_means = np.asarray(
        [[-2.2, -1.2], [2.2, -1.1], [0.0, 2.4]], dtype=np.float64
    )
    source_covariances = np.asarray(
        [
            [[0.75, 0.10], [0.10, 0.65]],
            [[0.70, -0.08], [-0.08, 0.80]],
            [[0.80, 0.00], [0.00, 0.70]],
        ],
        dtype=np.float64,
    )
    source_priors = np.asarray([0.34, 0.33, 0.33], dtype=np.float64)
    target_means = source_means.copy()
    target_covariances = source_covariances.copy()
    target_priors = source_priors.copy()
    if setting == "homogeneous_conditional_shift":
        target_means += np.asarray([1.0, 0.7])
    elif setting == "heterogeneous_conditional_shift":
        target_means += np.asarray(
            [[1.4, 0.1], [-0.9, 1.2], [0.4, -1.3]]
        )
        target_covariances = np.asarray(
            [
                [[1.05, 0.30], [0.30, 0.55]],
                [[0.50, -0.18], [-0.18, 1.15]],
                [[1.20, 0.00], [0.00, 0.45]],
            ],
            dtype=np.float64,
        )
    elif setting == "class_prior_shift_only":
        target_priors = np.asarray([0.58, 0.30, 0.12], dtype=np.float64)
    source_complement_propensity = np.asarray(
        [0.72, 0.82, 0.76], dtype=np.float64
    )
    target_complement_propensity = (
        source_complement_propensity.copy()
        if setting == "no_shift"
        else np.asarray([0.65, 0.87, 0.70], dtype=np.float64)
    )
    return {
        "source_means": source_means,
        "source_covariances": source_covariances,
        "source_priors": source_priors,
        "target_means": target_means,
        "target_covariances": target_covariances,
        "target_priors": target_priors,
        "source_complement_propensity": source_complement_propensity,
        "target_complement_propensity": target_complement_propensity,
    }


def _sample_domain(
    random_state: np.random.RandomState,
    size: int,
    means: np.ndarray,
    covariances: np.ndarray,
    priors: np.ndarray,
    complement_propensity: np.ndarray,
) -> Dict[str, torch.Tensor]:
    labels = random_state.choice(len(priors), size=size, p=priors)
    features = np.empty((size, means.shape[1]), dtype=np.float32)
    complements = np.zeros((size, len(priors)), dtype=np.float32)
    for index, label in enumerate(labels):
        features[index] = random_state.multivariate_normal(
            means[label], covariances[label]
        )
        for class_index in range(len(priors)):
            if class_index != label:
                complements[index, class_index] = float(
                    random_state.rand()
                    < complement_propensity[class_index]
                )
    return {
        "x": torch.from_numpy(features),
        "y": torch.from_numpy(labels.astype(np.int64)),
        "bar_y": torch.from_numpy(complements),
    }


def generate_gaussian_shift(
    setting: str,
    seed: int,
    source_size: int = 2000,
    target_weak_size: int = 1000,
    target_test_size: int = 2000,
    q: int = 3,
) -> Dict[str, Any]:
    parameters = gaussian_parameters(setting, q=q)
    random_state = np.random.RandomState(int(seed))
    source = _sample_domain(
        random_state,
        source_size,
        parameters["source_means"],
        parameters["source_covariances"],
        parameters["source_priors"],
        parameters["source_complement_propensity"],
    )
    target_weak = _sample_domain(
        random_state,
        target_weak_size,
        parameters["target_means"],
        parameters["target_covariances"],
        parameters["target_priors"],
        parameters["target_complement_propensity"],
    )
    target_test = _sample_domain(
        random_state,
        target_test_size,
        parameters["target_means"],
        parameters["target_covariances"],
        parameters["target_priors"],
        parameters["target_complement_propensity"],
    )
    return {
        "setting": setting,
        "parameters": parameters,
        "source": source,
        "target_weak": target_weak,
        "target_test": target_test,
        "q": q,
    }


def _component_log_density(
    x: torch.Tensor, means: np.ndarray, covariances: np.ndarray
) -> torch.Tensor:
    values = []
    x64 = x.double()
    dimension = x.shape[1]
    for mean, covariance in zip(means, covariances):
        mean_tensor = torch.as_tensor(mean, dtype=torch.double)
        covariance_tensor = torch.as_tensor(
            covariance, dtype=torch.double
        )
        inverse = torch.inverse(covariance_tensor)
        difference = x64 - mean_tensor
        quadratic = torch.sum(
            (difference @ inverse) * difference, dim=1
        )
        log_determinant = torch.logdet(covariance_tensor)
        values.append(
            -0.5
            * (
                dimension * math.log(2.0 * math.pi)
                + log_determinant
                + quadratic
            )
        )
    return torch.stack(values, dim=1)


def oracle_global_ratio(
    x: torch.Tensor, parameters: Dict[str, np.ndarray]
) -> torch.Tensor:
    source_components = _component_log_density(
        x,
        parameters["source_means"],
        parameters["source_covariances"],
    ) + torch.log(torch.as_tensor(parameters["source_priors"]))
    target_components = _component_log_density(
        x,
        parameters["target_means"],
        parameters["target_covariances"],
    ) + torch.log(torch.as_tensor(parameters["target_priors"]))
    return torch.exp(
        torch.logsumexp(target_components, dim=1)
        - torch.logsumexp(source_components, dim=1)
    ).float()


def oracle_joint_ratio(
    x: torch.Tensor,
    y: torch.Tensor,
    parameters: Dict[str, np.ndarray],
) -> torch.Tensor:
    source_log = _component_log_density(
        x,
        parameters["source_means"],
        parameters["source_covariances"],
    )
    target_log = _component_log_density(
        x,
        parameters["target_means"],
        parameters["target_covariances"],
    )
    indices = torch.arange(x.shape[0])
    prior_ratio = torch.as_tensor(
        parameters["target_priors"] / parameters["source_priors"]
    )
    return (
        prior_ratio[y.long()]
        * torch.exp(target_log[indices, y] - source_log[indices, y])
    ).float()


def _bar_probability(
    class_index: int,
    state: int,
    priors: np.ndarray,
    propensities: np.ndarray,
) -> np.ndarray:
    probability = np.empty_like(priors)
    for label in range(priors.size):
        observed_probability = (
            0.0 if label == class_index else propensities[class_index]
        )
        probability[label] = (
            observed_probability if state == 1 else 1.0 - observed_probability
        )
    return probability


def oracle_branch_ratios(
    x: torch.Tensor,
    bar_y: torch.Tensor,
    parameters: Dict[str, np.ndarray],
) -> torch.Tensor:
    """Return true ``p_te(x|bar_y_k=b)/p_tr(x|bar_y_k=b)``."""
    source_log_density = _component_log_density(
        x,
        parameters["source_means"],
        parameters["source_covariances"],
    )
    target_log_density = _component_log_density(
        x,
        parameters["target_means"],
        parameters["target_covariances"],
    )
    output = torch.empty_like(bar_y, dtype=torch.float64)
    for class_index in range(bar_y.shape[1]):
        for state in (0, 1):
            source_bar = _bar_probability(
                class_index,
                state,
                parameters["source_priors"],
                parameters["source_complement_propensity"],
            )
            target_bar = _bar_probability(
                class_index,
                state,
                parameters["target_priors"],
                parameters["target_complement_propensity"],
            )
            source_joint = (
                source_log_density
                + torch.log(
                    torch.as_tensor(
                        parameters["source_priors"] * source_bar
                    ).clamp_min(1.0e-12)
                )
            )
            target_joint = (
                target_log_density
                + torch.log(
                    torch.as_tensor(
                        parameters["target_priors"] * target_bar
                    ).clamp_min(1.0e-12)
                )
            )
            source_probability = float(
                np.dot(parameters["source_priors"], source_bar)
            )
            target_probability = float(
                np.dot(parameters["target_priors"], target_bar)
            )
            log_source_conditional = torch.logsumexp(
                source_joint, dim=1
            ) - math.log(max(source_probability, 1.0e-12))
            log_target_conditional = torch.logsumexp(
                target_joint, dim=1
            ) - math.log(max(target_probability, 1.0e-12))
            mask = bar_y[:, class_index].long() == state
            output[mask, class_index] = torch.exp(
                log_target_conditional[mask]
                - log_source_conditional[mask]
            )
    return output.float()


class GlobalDomainEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class ConditionalDomainEstimator(nn.Module):
    def __init__(self, q: int) -> None:
        super().__init__()
        self.q = q
        self.class_embedding = nn.Embedding(q, 6)
        self.state_embedding = nn.Embedding(2, 3)
        self.network = nn.Sequential(
            nn.Linear(2 + 6 + 3, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
        )
        self.register_buffer(
            "log_prior_correction", torch.zeros(q, 2)
        )

    def all_logits(self, x: torch.Tensor, bar_y: torch.Tensor) -> torch.Tensor:
        count = x.shape[0]
        classes = torch.arange(self.q, device=x.device)
        class_features = self.class_embedding(classes)[None].expand(
            count, -1, -1
        )
        state_features = self.state_embedding(bar_y.long())
        x_features = x[:, None, :].expand(-1, self.q, -1)
        combined = torch.cat(
            (x_features, class_features, state_features), dim=2
        )
        return self.network(combined).squeeze(2)


def fit_domain_estimators(
    source: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    q: int,
    seed: int,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 0.01,
) -> Tuple[GlobalDomainEstimator, ConditionalDomainEstimator, List[float]]:
    """Fit estimators with domain labels only; ordinary labels are untouched."""
    torch.manual_seed(int(seed))
    global_model = GlobalDomainEstimator().to(device)
    conditional_model = ConditionalDomainEstimator(q).to(device)
    parameters = list(global_model.parameters()) + list(
        conditional_model.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    source_x = source["x"].to(device)
    target_x = target["x"].to(device)
    source_bar = source["bar_y"].to(device)
    target_bar = target["bar_y"].to(device)
    with torch.no_grad():
        for class_index in range(q):
            for state in (0, 1):
                source_prior = (
                    source_bar[:, class_index].long() == state
                ).float().mean()
                target_prior = (
                    target_bar[:, class_index].long() == state
                ).float().mean()
                conditional_model.log_prior_correction[
                    class_index, state
                ] = torch.log(source_prior.clamp_min(1.0e-8)) - torch.log(
                    target_prior.clamp_min(1.0e-8)
                )
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        global_source = global_model(source_x)
        global_target = global_model(target_x)
        conditional_source = conditional_model.all_logits(
            source_x, source_bar
        )
        conditional_target = conditional_model.all_logits(
            target_x, target_bar
        )
        global_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                global_source, torch.zeros_like(global_source)
            )
            + F.binary_cross_entropy_with_logits(
                global_target, torch.ones_like(global_target)
            )
        )
        conditional_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                conditional_source,
                torch.zeros_like(conditional_source),
            )
            + F.binary_cross_entropy_with_logits(
                conditional_target,
                torch.ones_like(conditional_target),
            )
        )
        loss = global_loss + conditional_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return global_model, conditional_model, history


@torch.no_grad()
def estimated_ratios(
    global_model: GlobalDomainEstimator,
    conditional_model: ConditionalDomainEstimator,
    x: torch.Tensor,
    bar_y: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global_model.eval()
    conditional_model.eval()
    global_ratio = torch.exp(
        global_model(x.to(device)).double().clamp(-20.0, 20.0)
    ).float()
    states = bar_y.to(device).long()
    correction = conditional_model.log_prior_correction[
        torch.arange(conditional_model.q, device=device)[None, :],
        states,
    ]
    conditional_ratio = torch.exp(
        (
            conditional_model.all_logits(x.to(device), states)
            + correction
        )
        .double()
        .clamp(-20.0, 20.0)
    ).float()
    return global_ratio.cpu(), conditional_ratio.cpu()


class WeakGaussianClassifier(nn.Module):
    def __init__(self, q: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 48), nn.ReLU(), nn.Linear(48, q)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def weak_negative_loss(
    logits: torch.Tensor, bar_y: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1).clamp(
        1.0e-7, 1.0 - 1.0e-7
    )
    return -bar_y * torch.log1p(-probabilities)


def train_weak_classifier(
    source: Dict[str, torch.Tensor],
    target_weak: Dict[str, torch.Tensor],
    target_test: Dict[str, torch.Tensor],
    source_weights: torch.Tensor,
    seed: int,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """Train from complementary labels; target ordinary labels are evaluation-only."""
    torch.manual_seed(int(seed))
    q = source["bar_y"].shape[1]
    model = WeakGaussianClassifier(q).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
    source_x = source["x"].to(device)
    source_bar = source["bar_y"].to(device)
    target_x = target_weak["x"].to(device)
    target_bar = target_weak["bar_y"].to(device)
    test_x = target_test["x"].to(device)
    test_y = target_test["y"].to(device)
    weights = source_weights.to(device)
    if weights.ndim == 1:
        weights = weights[:, None].expand(-1, q)
    target_fraction = target_x.shape[0] / float(
        source_x.shape[0] + target_x.shape[0]
    )
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        source_losses = weak_negative_loss(model(source_x), source_bar)
        target_losses = weak_negative_loss(model(target_x), target_bar)
        source_risk = (source_losses * weights).sum() / (
            source_bar * weights
        ).sum().clamp_min(1.0)
        target_risk = target_losses.sum() / target_bar.sum().clamp_min(1.0)
        total = (
            (1.0 - target_fraction) * source_risk
            + target_fraction * target_risk
        )
        total.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            test_logits = model(test_x)
            true_risk = F.cross_entropy(test_logits, test_y)
            accuracy = (
                test_logits.argmax(dim=1) == test_y
            ).float().mean() * 100.0
            weak_estimate = weak_negative_loss(
                model(target_x), target_bar
            ).sum() / target_bar.sum().clamp_min(1.0)
            iw_estimate = (
                weak_negative_loss(model(source_x), source_bar) * weights
            ).sum() / (source_bar * weights).sum().clamp_min(1.0)
        rows.append(
            {
                "epoch": epoch,
                "target_accuracy": float(accuracy.item()),
                "true_target_risk": float(true_risk.item()),
                "estimated_weak_risk": float(weak_estimate.item()),
                "estimated_iw_risk": float(iw_estimate.item()),
                "classifier_loss": float(total.item()),
                "source_risk_contribution": float(
                    ((1.0 - target_fraction) * source_risk).item()
                ),
                "target_risk_contribution": float(
                    (target_fraction * target_risk).item()
                ),
            }
        )
    return {"model": model, "epoch_rows": rows}
