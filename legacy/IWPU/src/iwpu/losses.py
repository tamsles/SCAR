"""Losses for importance-weighted positive-unlabeled learning.

The implementation follows Eqs. (4), (13), and (19) of Kumagai et al.,
*Importance-weighted Positive-unlabeled Learning for Distribution Shift
Adaptation*.  Every expectation is represented by an ordinary sample mean;
importance weights are deliberately **not** normalized by their sum.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor


def _validate_probability(value: float, name: str) -> float:
    """Return a scalar probability after validating its range."""

    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def _validate_alpha(alpha: float) -> float:
    """Validate alpha for the bounded relative density ratio."""

    if not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a real scalar, got {type(alpha).__name__}")
    alpha = float(alpha)
    # Although Eq. (14) discusses alpha=0 conceptually, Eq. (19) and the
    # network output (sigmoid / alpha) are undefined there.
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    return alpha


def _mean(values: Tensor, name: str) -> Tensor:
    if values.numel() == 0:
        raise ValueError(f"{name} must contain at least one value")
    return values.mean()


def _weighted_mean(values: Tensor, weights: Tensor | None, name: str) -> Tensor:
    if weights is not None:
        weights = torch.as_tensor(weights, device=values.device, dtype=values.dtype)
        try:
            values = values * weights
        except RuntimeError as exc:
            raise ValueError(
                f"{name} weights with shape {tuple(weights.shape)} are not "
                f"broadcastable to losses with shape {tuple(values.shape)}"
            ) from exc
    return _mean(values, name)


def sigmoid_loss(logits: Tensor, y: Tensor | float | int) -> Tensor:
    """Return the elementwise sigmoid classification loss ``sigmoid(-y*f)``.

    Parameters
    ----------
    logits:
        Classifier scores ``f(x)``.
    y:
        Labels encoded as ``-1`` or ``+1``.  A scalar label is broadcast over
        ``logits``.

    Notes
    -----
    The paper specifies sigmoid loss in Section 5.2.  This is the bounded
    sigmoid loss used in the cited PU-learning work, not binary cross entropy.
    """

    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor")
    targets = torch.as_tensor(y, device=logits.device, dtype=logits.dtype)
    if torch.any((targets != -1) & (targets != 1)):
        raise ValueError("y must contain only labels encoded as -1 or +1")
    return torch.sigmoid(-targets * logits)


def _split_positive_weights(
    pos_weights: Tensor | Sequence[Tensor] | None,
    pos_logits: Tensor,
) -> tuple[Tensor | None, Tensor | None]:
    """Resolve weights for positive examples evaluated at both labels.

    A two-item sequence means ``(m(x,+1), m(x,-1))``.  A tensor whose final
    dimension adds a two-column label axis has the same meaning.  Any other
    tensor is shared between the two label evaluations; this is convenient
    for ordinary (unweighted) PU baselines.
    """

    if pos_weights is None:
        return None, None
    if isinstance(pos_weights, Sequence) and not torch.is_tensor(pos_weights):
        if len(pos_weights) != 2:
            raise ValueError("pos_weights sequence must have exactly two entries")
        return pos_weights[0], pos_weights[1]
    if not torch.is_tensor(pos_weights):
        raise TypeError("pos_weights must be a tensor, a two-tensor sequence, or None")
    if pos_weights.shape == (*pos_logits.shape, 2):
        return pos_weights[..., 0], pos_weights[..., 1]
    return pos_weights, pos_weights


def pu_risk(
    pos_logits: Tensor,
    unl_logits: Tensor,
    pi: float,
    correct: bool = True,
    pos_weights: Tensor | Sequence[Tensor] | None = None,
    unl_weights: Tensor | None = None,
) -> Tensor:
    """Compute the (optionally importance-weighted) empirical PU risk.

    This implements Eq. (4) when no weights are supplied and Eq. (13) when
    relative importance weights are supplied::

        pi * mean[m(x,+) l(f(x),+)]
        + C(mean[m(x,-) l(f(x),-)]_U
            - pi * mean[m(x,-) l(f(x),-)]_P),

    where ``C(z)=abs(z)`` when ``correct=True`` and ``C(z)=z`` otherwise.

    ``pos_weights`` may be ``(positive_label_weights,
    negative_label_weights)`` or an ``[..., 2]`` tensor, because the same
    positive examples are evaluated with both labels in the PU identity.  A
    single tensor is shared by both terms.  ``unl_weights`` contains the
    negative-label weights for unlabeled examples.
    """

    if not torch.is_tensor(pos_logits) or not torch.is_tensor(unl_logits):
        raise TypeError("pos_logits and unl_logits must be torch.Tensor objects")
    pi = _validate_probability(pi, "pi")
    pos_w_pos, pos_w_neg = _split_positive_weights(pos_weights, pos_logits)

    positive = pi * _weighted_mean(
        sigmoid_loss(pos_logits, +1), pos_w_pos, "positive examples"
    )
    negative = _weighted_mean(
        sigmoid_loss(unl_logits, -1), unl_weights, "unlabeled examples"
    ) - pi * _weighted_mean(
        sigmoid_loss(pos_logits, -1), pos_w_neg, "positive examples"
    )
    if correct:
        negative = negative.abs()
    return positive + negative


def _ratio_quadratic(m: Tensor, alpha: float) -> Tensor:
    """The ``M(x,y)=alpha*m(x,y)^2-2*m(x,y)`` term from Eq. (19)."""

    return alpha * m.square() - 2.0 * m


def importance_ratio_loss(
    m_te_pos_y_pos: Tensor,
    m_te_unl_y_neg: Tensor,
    m_te_pos_y_neg: Tensor,
    m_tr_pos_y_pos: Tensor,
    m_tr_unl_y_neg: Tensor,
    m_tr_pos_y_neg: Tensor,
    pi_tr: float,
    pi_te: float,
    alpha: float,
    correct: bool = True,
) -> Tensor:
    """Compute the tensor-level relative importance-ratio objective (Eq. 19).

    The six tensors are evaluations of ``m(x,y)`` on, in order:

    - test positive examples with ``y=+1``;
    - test unlabeled examples with ``y=-1``;
    - test positive examples with ``y=-1``;
    - training positive examples with ``y=+1``;
    - training unlabeled examples with ``y=-1``;
    - training positive examples with ``y=-1``.

    With correction, the test negative-conditional estimate uses its known
    lower bound ``-(1-pi_te)/alpha`` and the training estimate uses the usual
    absolute-value non-negative correction.  As in the displayed Eq. (19),
    the additive constant ``-(1-pi_te)/alpha`` is omitted because it has zero
    gradient.  Set ``correct=False`` to use the unbiased, uncorrected Eq. (16)
    decomposition.
    """

    values = (
        m_te_pos_y_pos,
        m_te_unl_y_neg,
        m_te_pos_y_neg,
        m_tr_pos_y_pos,
        m_tr_unl_y_neg,
        m_tr_pos_y_neg,
    )
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("all m arguments must be torch.Tensor objects")
    pi_tr = _validate_probability(pi_tr, "pi_tr")
    pi_te = _validate_probability(pi_te, "pi_te")
    alpha = _validate_alpha(alpha)

    test_positive = pi_te * _mean(
        _ratio_quadratic(m_te_pos_y_pos, alpha), "test positive weights"
    )
    test_negative = _mean(
        _ratio_quadratic(m_te_unl_y_neg, alpha), "test unlabeled weights"
    ) - pi_te * _mean(
        _ratio_quadratic(m_te_pos_y_neg, alpha), "test positive weights"
    )

    train_positive = pi_tr * (1.0 - alpha) * _mean(
        m_tr_pos_y_pos.square(), "training positive weights"
    )
    train_negative = (1.0 - alpha) * (
        _mean(m_tr_unl_y_neg.square(), "training unlabeled weights")
        - pi_tr * _mean(m_tr_pos_y_neg.square(), "training positive weights")
    )

    if correct:
        test_negative = (test_negative + (1.0 - pi_te) / alpha).abs()
        train_negative = train_negative.abs()

    return test_positive + test_negative + train_positive + train_negative


__all__ = ["importance_ratio_loss", "pu_risk", "sigmoid_loss"]
