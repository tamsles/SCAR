"""ADIW-SCAR objectives for source/target complementary-label learning.

The implementation follows the branch decomposition in slides 17--28 of
``20260622_TaichuLiu.pptx``.  Every class is split into the observed
complementary branch (bar_y_k = 1) and its unlabeled branch (bar_y_k = 0).
ADIW estimates a target-to-source conditional ratio for each branch, and the
source SCAR risk is combined with a target-only SCAR risk.
"""

import math

import torch

from utils_algo import logistic_loss


def _as_matrix(values, name):
    if values.ndim == 1:
        values = values.unsqueeze(1)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "{} must be a non-empty vector or matrix".format(name)
        )
    if not torch.isfinite(values).all().item():
        raise ValueError("{} contains NaN or infinite values".format(name))
    return values


def _as_prior(values, num_classes, reference, name):
    prior = torch.as_tensor(
        values, device=reference.device, dtype=reference.dtype
    ).view(-1)
    if prior.shape[0] != num_classes:
        raise ValueError("{} must contain one value per class".format(name))
    if not torch.isfinite(prior).all().item():
        raise ValueError("{} contains NaN or infinite values".format(name))
    if ((prior < 0.0) | (prior > 1.0)).any().item():
        raise ValueError("{} must be between zero and one".format(name))
    return prior


def branch_prior_ratios(source_complement_prior, target_complement_prior):
    """Return rho_k^+ and rho_k^- from slide 22."""
    source = torch.as_tensor(source_complement_prior)
    target = torch.as_tensor(
        target_complement_prior, device=source.device, dtype=source.dtype
    )
    if source.shape != target.shape:
        raise ValueError("source and target priors must have the same shape")
    epsilon = torch.finfo(source.dtype).eps
    positive = target / source.clamp_min(epsilon)
    negative = (1.0 - target) / (1.0 - source).clamp_min(epsilon)
    return positive, negative


def joint_to_conditional_weights(joint_weights, branch_prior_ratio):
    """Recover r_k^s = omega_k^s / rho_k^s from slides 21--22."""
    ratio = torch.as_tensor(
        branch_prior_ratio,
        device=joint_weights.device,
        dtype=joint_weights.dtype,
    )
    epsilon = torch.finfo(joint_weights.dtype).eps
    return joint_weights / ratio.clamp_min(epsilon)


def _pairwise_squared_distance(left, right):
    difference = left[:, None, :] - right[None, :, :]
    return torch.sum(difference * difference, dim=2)


def rbf_gamma_from_quantile(source, target, quantile=0.5):
    """Use a distance quantile as the RBF median-style bandwidth."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    source = _as_matrix(source, "source")
    target = _as_matrix(target, "target")
    combined = torch.cat([source, target], dim=0)
    if combined.shape[0] < 2:
        return combined.new_tensor(1.0)

    squared = _pairwise_squared_distance(combined, combined)
    upper = torch.triu_indices(
        combined.shape[0], combined.shape[0], offset=1,
        device=combined.device,
    )
    distances = squared[upper[0], upper[1]]
    epsilon = torch.finfo(combined.dtype).eps
    distances = distances[distances > epsilon]
    if distances.numel() == 0:
        return combined.new_tensor(1.0)
    bandwidth_squared = torch.quantile(distances, quantile).clamp_min(epsilon)
    return 0.5 / bandwidth_squared


def mmd_weight_objective(source, target, weights, gamma):
    """Half squared MMD between target and weighted source distributions."""
    source = _as_matrix(source, "source")
    target = _as_matrix(target, "target")
    weights = weights.view(-1)
    if source.shape[0] != weights.shape[0]:
        raise ValueError("weights must contain one value per source sample")

    source_kernel = torch.exp(
        -gamma * _pairwise_squared_distance(source, source)
    )
    cross_kernel = torch.exp(
        -gamma * _pairwise_squared_distance(source, target)
    )
    num_source = float(source.shape[0])
    source_term = 0.5 * torch.dot(
        weights, torch.mv(source_kernel, weights)
    ) / (num_source * num_source)
    cross_term = torch.dot(
        weights, cross_kernel.mean(dim=1)
    ) / num_source
    return source_term - cross_term


def _project_to_box_with_sum(values, target_sum, max_weight):
    """Euclidean projection onto {0 <= w <= B, sum(w) = target_sum}."""
    num_values = values.numel()
    target_sum = max(0.0, min(float(target_sum), num_values * max_weight))
    if target_sum == 0.0:
        return torch.zeros_like(values)
    if target_sum == num_values * max_weight:
        return torch.full_like(values, max_weight)

    lower = torch.min(values - max_weight).item()
    upper = torch.max(values).item()
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        projected_sum = (
            values.sub(midpoint).clamp(0.0, max_weight).sum().item()
        )
        if projected_sum > target_sum:
            lower = midpoint
        else:
            upper = midpoint
    return values.sub(0.5 * (lower + upper)).clamp(0.0, max_weight)


def project_kmm_weights(
        values,
        max_weight=50.0,
        sum_tolerance=None):
    """Project weights onto the standard KMM box and mean constraints."""
    values = values.view(-1)
    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")

    num_values = values.numel()
    if sum_tolerance is None:
        sum_tolerance = (
            (math.sqrt(num_values) - 1.0) / math.sqrt(num_values)
        )
    if not 0.0 <= sum_tolerance <= 1.0:
        raise ValueError("sum_tolerance must be between zero and one")

    lower_sum = num_values * (1.0 - sum_tolerance)
    upper_sum = min(
        num_values * (1.0 + sum_tolerance),
        num_values * max_weight,
    )
    projected = values.clamp(0.0, max_weight)
    current_sum = projected.sum().item()
    if current_sum < lower_sum:
        projected = _project_to_box_with_sum(
            values, lower_sum, max_weight
        )
    elif current_sum > upper_sum:
        projected = _project_to_box_with_sum(
            values, upper_sum, max_weight
        )
    return projected


def adiw_kmm_pgd(
        source,
        target,
        initial_weights,
        step_size=1.0,
        num_steps=1,
        max_weight=50.0,
        kernel_quantile=0.5,
        sum_tolerance=None):
    """Perform warm-started ADIW-KM projected-gradient updates."""
    source = _as_matrix(source.detach(), "source")
    target = _as_matrix(target.detach(), "target")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target representations must align")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    weights = initial_weights.detach().to(
        device=source.device, dtype=source.dtype
    ).view(-1)
    if weights.shape[0] != source.shape[0]:
        raise ValueError("initial_weights must align with source")

    gamma = rbf_gamma_from_quantile(
        source, target, quantile=kernel_quantile
    )
    source_kernel = torch.exp(
        -gamma * _pairwise_squared_distance(source, source)
    )
    cross_mean = torch.exp(
        -gamma * _pairwise_squared_distance(source, target)
    ).mean(dim=1)
    num_source = float(source.shape[0])

    for _ in range(num_steps):
        gradient = (
            torch.mv(source_kernel, weights) / (num_source * num_source)
            - cross_mean / num_source
        )
        weights = project_kmm_weights(
            weights - step_size * gradient,
            max_weight=max_weight,
            sum_tolerance=sum_tolerance,
        )
    return weights.detach()


class BranchWeightMemory(object):
    """Global per-source, per-class ADIW weights with warm starts."""

    def __init__(self, num_source_samples, num_classes, device):
        if num_source_samples <= 0 or num_classes <= 1:
            raise ValueError("invalid source size or number of classes")
        self.num_classes = int(num_classes)
        self.weights = torch.ones(
            num_source_samples, num_classes, device=device
        )

    def estimate(
            self,
            source_indices,
            source_representations,
            source_complements,
            target_representations,
            target_complements,
            source_complement_prior,
            target_complement_prior,
            step_size=1.0,
            num_steps=1,
            max_weight=50.0,
            kernel_quantile=0.5,
            sum_tolerance=None):
        """Update every observed/unlabeled class branch in one mini-batch."""
        source_indices = source_indices.to(
            device=self.weights.device, dtype=torch.long
        ).view(-1)
        source_representations = _as_matrix(
            source_representations, "source_representations"
        )
        target_representations = _as_matrix(
            target_representations, "target_representations"
        )
        source_complements = source_complements.to(
            device=self.weights.device
        )
        target_complements = target_complements.to(
            device=self.weights.device
        )
        if source_representations.shape[1] != self.num_classes:
            raise ValueError("source representations must have K columns")
        if target_representations.shape[1] != self.num_classes:
            raise ValueError("target representations must have K columns")
        if source_complements.shape != source_representations.shape:
            raise ValueError("source complement vectors must be B by K")
        if target_complements.shape != target_representations.shape:
            raise ValueError("target complement vectors must be B by K")
        if source_indices.shape[0] != source_representations.shape[0]:
            raise ValueError("source indices must align with the batch")

        positive_rho, negative_rho = branch_prior_ratios(
            torch.as_tensor(
                source_complement_prior,
                device=self.weights.device,
                dtype=self.weights.dtype,
            ),
            torch.as_tensor(
                target_complement_prior,
                device=self.weights.device,
                dtype=self.weights.dtype,
            ),
        )
        batch_weights = self.weights[source_indices].clone()

        for class_index in range(self.num_classes):
            for observed in (False, True):
                source_mask = (
                    source_complements[:, class_index] > 0.5
                ) == observed
                target_mask = (
                    target_complements[:, class_index] > 0.5
                ) == observed
                if (
                        not source_mask.any().item()
                        or not target_mask.any().item()):
                    continue

                conditional = adiw_kmm_pgd(
                    source_representations[source_mask, class_index],
                    target_representations[target_mask, class_index],
                    batch_weights[source_mask, class_index],
                    step_size=step_size,
                    num_steps=num_steps,
                    max_weight=max_weight,
                    kernel_quantile=kernel_quantile,
                    sum_tolerance=sum_tolerance,
                )
                rho = (
                    positive_rho[class_index]
                    if observed else negative_rho[class_index]
                )
                joint_weights = conditional * rho
                batch_weights[source_mask, class_index] = (
                    joint_to_conditional_weights(joint_weights, rho)
                )

        self.weights[source_indices] = batch_weights.detach()
        return batch_weights.detach()


def loss_value_representation(outputs):
    """The scalar-per-class transformation z_theta,k(x)=ell(f_k(x))."""
    return logistic_loss(outputs)


def _branch_mean(values, mask, weights=None):
    selected = values[mask]
    if selected.numel() == 0:
        return values.new_zeros(())
    if weights is None:
        return selected.mean()
    selected_weights = weights[mask]
    denominator = selected_weights.sum()
    epsilon = torch.finfo(values.dtype).eps
    if denominator.detach().item() <= epsilon:
        return selected.mean()
    return torch.sum(selected * selected_weights) / denominator


def scar_branch_risk(
        outputs,
        complementary_vectors,
        target_class_prior,
        target_complement_prior,
        branch_weights=None,
        correction="abs"):
    """Compute the target-coefficient SCAR risk from branch expectations."""
    if outputs.ndim != 2:
        raise ValueError("outputs must be a B by K matrix")
    complementary_vectors = complementary_vectors.to(
        device=outputs.device, dtype=outputs.dtype
    )
    if complementary_vectors.shape != outputs.shape:
        raise ValueError("complementary_vectors must have shape B by K")
    if branch_weights is not None:
        branch_weights = branch_weights.to(
            device=outputs.device, dtype=outputs.dtype
        )
        if branch_weights.shape != outputs.shape:
            raise ValueError("branch_weights must have shape B by K")
        if (branch_weights < 0.0).any().item():
            raise ValueError("branch_weights must be non-negative")
    if correction not in ("abs", "none", "relu"):
        raise ValueError("correction must be abs, relu, or none")

    num_classes = outputs.shape[1]
    class_prior = _as_prior(
        target_class_prior, num_classes, outputs, "target_class_prior"
    )
    complement_prior = _as_prior(
        target_complement_prior,
        num_classes,
        outputs,
        "target_complement_prior",
    )
    positive_loss = logistic_loss(outputs)
    negative_loss = logistic_loss(-outputs)

    negative_risks = []
    positive_risks = []
    corrected_positive_risks = []
    for class_index in range(num_classes):
        observed = complementary_vectors[:, class_index] > 0.5
        unlabeled = ~observed
        weights = (
            None if branch_weights is None
            else branch_weights[:, class_index]
        )
        negative_observed = _branch_mean(
            negative_loss[:, class_index], observed, weights
        )
        positive_observed = _branch_mean(
            positive_loss[:, class_index], observed, weights
        )
        positive_unlabeled = _branch_mean(
            positive_loss[:, class_index], unlabeled, weights
        )

        negative_risk = (
            1.0 - class_prior[class_index]
        ) * negative_observed
        positive_risk = (
            complement_prior[class_index]
            + class_prior[class_index]
            - 1.0
        ) * positive_observed + (
            1.0 - complement_prior[class_index]
        ) * positive_unlabeled
        if correction == "abs":
            corrected_positive = torch.abs(positive_risk)
        elif correction == "relu":
            corrected_positive = torch.relu(positive_risk)
        else:
            corrected_positive = positive_risk

        negative_risks.append(negative_risk)
        positive_risks.append(positive_risk)
        corrected_positive_risks.append(corrected_positive)

    negative_risks = torch.stack(negative_risks)
    positive_risks = torch.stack(positive_risks)
    corrected_positive_risks = torch.stack(corrected_positive_risks)
    total = negative_risks.sum() + corrected_positive_risks.sum()
    diagnostics = {
        "negative": negative_risks,
        "positive_uncorrected": positive_risks,
        "positive_corrected": corrected_positive_risks,
    }
    return total, diagnostics


def combined_adiw_scar_loss(
        source_outputs,
        source_complements,
        source_branch_weights,
        target_outputs,
        target_complements,
        target_class_prior,
        target_complement_prior,
        target_fraction,
        correction="abs"):
    """Combine target-only and ADIW-weighted source SCAR risks."""
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must be between zero and one")
    source_risk, source_diagnostics = scar_branch_risk(
        source_outputs,
        source_complements,
        target_class_prior,
        target_complement_prior,
        branch_weights=source_branch_weights,
        correction=correction,
    )
    target_risk, target_diagnostics = scar_branch_risk(
        target_outputs,
        target_complements,
        target_class_prior,
        target_complement_prior,
        branch_weights=None,
        correction=correction,
    )
    total = (
        target_fraction * target_risk
        + (1.0 - target_fraction) * source_risk
    )
    return total, {
        "source_risk": source_risk,
        "target_risk": target_risk,
        "source": source_diagnostics,
        "target": target_diagnostics,
    }
