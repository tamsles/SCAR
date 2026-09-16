from __future__ import annotations

import pytest
import torch

from iwpu.losses import importance_ratio_loss, pu_risk, sigmoid_loss


def test_sigmoid_loss_is_bounded_sigmoid_not_bce() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0])
    labels = torch.tensor([-1.0, 1.0, 1.0])
    expected = torch.sigmoid(-labels * logits)
    torch.testing.assert_close(sigmoid_loss(logits, labels), expected)


def test_pu_risk_matches_eq4_with_and_without_correction() -> None:
    pos = torch.tensor([-2.0, -2.0])
    unl = torch.tensor([-4.0, -4.0])
    pi = 0.8
    positive = pi * torch.sigmoid(-pos).mean()
    negative = torch.sigmoid(unl).mean() - pi * torch.sigmoid(pos).mean()
    assert negative < 0
    torch.testing.assert_close(
        pu_risk(pos, unl, pi, correct=False), positive + negative
    )
    torch.testing.assert_close(pu_risk(pos, unl, pi), positive + negative.abs())


def test_pu_risk_uses_label_specific_weights_and_plain_sample_means() -> None:
    pos = torch.tensor([0.0, 1.0])
    unl = torch.tensor([-1.0, 2.0, 0.5])
    pos_w_pos = torch.tensor([2.0, 4.0])
    pos_w_neg = torch.tensor([3.0, 5.0])
    unl_w_neg = torch.tensor([7.0, 11.0, 13.0])
    pi = 0.4
    expected = pi * (pos_w_pos * torch.sigmoid(-pos)).mean()
    expected += (
        (unl_w_neg * torch.sigmoid(unl)).mean()
        - pi * (pos_w_neg * torch.sigmoid(pos)).mean()
    ).abs()
    actual = pu_risk(
        pos,
        unl,
        pi,
        pos_weights=(pos_w_pos, pos_w_neg),
        unl_weights=unl_w_neg,
    )
    torch.testing.assert_close(actual, expected)

    stacked = torch.stack((pos_w_pos, pos_w_neg), dim=-1)
    torch.testing.assert_close(
        pu_risk(pos, unl, pi, pos_weights=stacked, unl_weights=unl_w_neg),
        expected,
    )


def test_importance_ratio_loss_matches_corrected_eq19() -> None:
    alpha, pi_tr, pi_te = 0.5, 0.4, 0.3
    te_pp = torch.tensor([0.2, 0.8], requires_grad=True)
    te_un = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    te_pn = torch.tensor([0.3, 0.7], requires_grad=True)
    tr_pp = torch.tensor([0.4, 0.9], requires_grad=True)
    tr_un = torch.tensor([0.2, 0.6, 1.2], requires_grad=True)
    tr_pn = torch.tensor([0.1, 0.8], requires_grad=True)

    m = lambda z: alpha * z.square() - 2.0 * z
    expected = pi_te * m(te_pp).mean()
    expected += (m(te_un).mean() - pi_te * m(te_pn).mean() + (1 - pi_te) / alpha).abs()
    expected += pi_tr * (1 - alpha) * tr_pp.square().mean()
    expected += (1 - alpha) * (
        tr_un.square().mean() - pi_tr * tr_pn.square().mean()
    ).abs()

    actual = importance_ratio_loss(
        te_pp, te_un, te_pn, tr_pp, tr_un, tr_pn, pi_tr, pi_te, alpha
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(tensor.grad is not None for tensor in (te_pp, te_un, te_pn, tr_pp, tr_un, tr_pn))


def test_importance_ratio_loss_uncorrected_matches_eq16_decomposition() -> None:
    tensors = [torch.tensor([value, value + 0.2]) for value in (0.1, 0.4, 0.2, 0.6, 0.3, 0.5)]
    pi_tr, pi_te, alpha = 0.5, 0.25, 0.4
    te_pp, te_un, te_pn, tr_pp, tr_un, tr_pn = tensors
    m = lambda z: alpha * z.square() - 2.0 * z
    expected = pi_te * m(te_pp).mean()
    expected += m(te_un).mean() - pi_te * m(te_pn).mean()
    expected += pi_tr * (1 - alpha) * tr_pp.square().mean()
    expected += (1 - alpha) * (
        tr_un.square().mean() - pi_tr * tr_pn.square().mean()
    )
    actual = importance_ratio_loss(
        *tensors, pi_tr=pi_tr, pi_te=pi_te, alpha=alpha, correct=False
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_importance_ratio_loss_rejects_invalid_alpha(alpha: float) -> None:
    values = [torch.ones(2)] * 6
    with pytest.raises(ValueError, match="alpha"):
        importance_ratio_loss(*values, pi_tr=0.5, pi_te=0.5, alpha=alpha)


def test_losses_reject_empty_batches_and_invalid_labels() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pu_risk(torch.ones(2), torch.empty(0), 0.5)
    with pytest.raises(ValueError, match=r"-1 or \+1"):
        sigmoid_loss(torch.ones(2), torch.tensor([0, 1]))
