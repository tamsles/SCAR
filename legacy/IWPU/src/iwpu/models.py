"""Neural networks used by the IW-PU experiments.

The model exposes the paper's notation directly: ``h`` is the shared feature
extractor, ``u`` is the classifier, and ``v`` is the relative importance-ratio
head from Eqs. (22)-(23).  Convenience methods make the two stop-gradient
operations in Algorithm 1 explicit.
"""

from __future__ import annotations

from numbers import Integral, Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a real scalar, got {type(alpha).__name__}")
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    return alpha


class MLPFeatureExtractor(nn.Module):
    """Three-layer ReLU feature extractor from Appendix C."""

    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = width
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} input features, got {x.shape[1]}"
            )
        return self.network(x)


class CIFAR10FeatureExtractor(nn.Module):
    """LeNet-style CIFAR10 feature extractor specified in Appendix C."""

    output_dim = 84

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim != 3 * 32 * 32:
            raise ValueError(
                "CIFAR10 uses 3x32x32 inputs, so input_dim must equal 3072"
            )
        self.input_dim = input_dim
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 32 -> 28 -> 14 -> 10 -> 5, hence 16*5*5 flattened features.
        self.feed_forward = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"expected flattened CIFAR10 width {self.input_dim}, got {x.shape[1]}"
                )
            x = x.reshape(x.shape[0], 3, 32, 32)
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError(
                "CIFAR10 inputs must have shape [batch, 3, 32, 32] or [batch, 3072]"
            )
        return self.feed_forward(self.convolutions(x).flatten(start_dim=1))


class ClassifierHead(nn.Module):
    """The one-layer classifier ``u: R^K -> R`` from Eq. (22)."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features).squeeze(-1)


def _labels_to_one_hot(
    labels: Tensor | float | int,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    labels = torch.as_tensor(labels, device=device)
    if labels.ndim == 2 and labels.shape[-1] == 2:
        if labels.shape[0] != batch_size:
            raise ValueError(
                f"one-hot labels have batch size {labels.shape[0]}, expected {batch_size}"
            )
        one_hot = labels.to(dtype=dtype)
        if torch.any(one_hot < 0) or torch.any(one_hot > 1) or not torch.allclose(
            one_hot.sum(dim=-1), torch.ones(batch_size, device=device, dtype=dtype)
        ):
            raise ValueError("one-hot label rows must be non-negative and sum to one")
        return one_hot

    if labels.ndim == 2 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    if labels.ndim == 0:
        labels = labels.expand(batch_size)
    elif labels.ndim == 1 and labels.numel() == 1:
        labels = labels.expand(batch_size)
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError(
            f"labels must be scalar, [batch], [batch, 1], or [batch, 2]; got {tuple(labels.shape)}"
        )
    if torch.any((labels != -1) & (labels != 1)):
        raise ValueError("labels must use -1/+1 encoding")
    # Column zero is y=-1 and column one is y=+1.
    return F.one_hot((labels > 0).long(), num_classes=2).to(dtype=dtype)


class RelativeRatioHead(nn.Module):
    """Two-layer model for ``m(x,y)`` with range ``(0, 1/alpha)``.

    Appendix C does not report the hidden width of ``v``.  We use ``K``, the
    shared feature width, which is the smallest architecture-preserving
    interpretation and is recorded here to keep the reproduction explicit.
    """

    def __init__(self, feature_dim: int, alpha: float) -> None:
        super().__init__()
        self.alpha = _validate_alpha(alpha)
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, features: Tensor, labels: Tensor | float | int) -> Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [batch, K], got {tuple(features.shape)}"
            )
        one_hot = _labels_to_one_hot(
            labels,
            features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )
        raw = self.network(torch.cat((features, one_hot), dim=-1)).squeeze(-1)
        return torch.sigmoid(raw) / self.alpha


class IWPUModel(nn.Module):
    """Shared-feature classifier and relative-ratio model (Eqs. 22-23)."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        alpha: float,
    ) -> None:
        super().__init__()
        # Names mirror the notation in the paper and keep optimizer groups
        # unambiguous: (h,u) for classification and v for ratio estimation.
        self.h = feature_extractor
        self.u = ClassifierHead(feature_dim)
        self.v = RelativeRatioHead(feature_dim, alpha)
        self.feature_dim = feature_dim
        self.alpha = self.v.alpha

    @property
    def feature_extractor(self) -> nn.Module:
        return self.h

    @property
    def classifier(self) -> ClassifierHead:
        return self.u

    @property
    def ratio_model(self) -> RelativeRatioHead:
        return self.v

    def encode(self, x: Tensor, *, detach: bool = False) -> Tensor:
        """Compute ``h(x)`` and optionally stop its gradient."""

        features = self.h(x)
        return features.detach() if detach else features

    def classify_features(self, features: Tensor) -> Tensor:
        """Compute classifier logits from already-encoded features."""

        return self.u(features)

    def forward(self, x: Tensor) -> Tensor:
        """Compute classifier logits ``f(x)=u(h(x))``."""

        return self.classify_features(self.encode(x))

    def importance_from_features(
        self,
        features: Tensor,
        y: Tensor | float | int,
        *,
        detach_features: bool = False,
        detach_output: bool = False,
    ) -> Tensor:
        """Evaluate ``m(x,y)`` with Algorithm 1's stop-gradients available.

        Use ``detach_features=True`` in the importance-estimation step, where
        only ``v`` is updated.  Use ``detach_output=True`` in the classifier
        step, where the current importance weights must remain fixed even
        though ``h`` is updated.
        """

        if detach_features:
            features = features.detach()
        weights = self.v(features, y)
        return weights.detach() if detach_output else weights

    def importance(
        self,
        x: Tensor,
        y: Tensor | float | int,
        *,
        detach_features: bool = False,
        detach_output: bool = False,
    ) -> Tensor:
        """Encode inputs and evaluate their relative importance weights."""

        features = self.encode(x, detach=detach_features)
        return self.importance_from_features(
            features, y, detach_output=detach_output
        )


def build_model(dataset: str, input_dim: int, alpha: float) -> IWPUModel:
    """Build the paper's architecture for a supported dataset.

    ``MNIST`` and ``FMNIST`` use width 128, ``DIABETES`` uses width 32, and
    ``CIFAR10`` uses the two-convolution LeNet-style extractor ending at 84
    features (Appendix C).  ``FOODSTAMP`` also uses width 32: the paper only
    reports its 239-dimensional input, so reusing the tabular architecture is
    an explicit reproduction assumption.
    """

    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if not isinstance(input_dim, Integral) or isinstance(input_dim, bool):
        raise TypeError("input_dim must be a positive integer")
    input_dim = int(input_dim)
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    alpha = _validate_alpha(alpha)
    key = dataset.strip().lower().replace("-", "").replace("_", "")

    if key in {"mnist", "fmnist", "fashionmnist"}:
        feature_dim = 128
        h: nn.Module = MLPFeatureExtractor(input_dim, feature_dim)
    elif key in {"diabetes", "foodstamp", "foodstamps"}:
        feature_dim = 32
        h = MLPFeatureExtractor(input_dim, feature_dim)
    elif key == "cifar10":
        feature_dim = CIFAR10FeatureExtractor.output_dim
        h = CIFAR10FeatureExtractor(input_dim)
    else:
        raise ValueError(
            f"unsupported dataset {dataset!r}; expected MNIST, FMNIST, CIFAR10, "
            "DIABETES, or FOODSTAMP"
        )
    return IWPUModel(h, feature_dim, alpha)


__all__ = [
    "CIFAR10FeatureExtractor",
    "ClassifierHead",
    "IWPUModel",
    "MLPFeatureExtractor",
    "RelativeRatioHead",
    "build_model",
]
