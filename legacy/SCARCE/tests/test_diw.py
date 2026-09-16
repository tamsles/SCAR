import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from diw import get_kernel_width, kmm
from utils_algo import SCARCE_loss, complementary_nll


def reference_scarce_loss(outputs, labels):
    mask = torch.zeros_like(outputs)
    mask[torch.arange(len(labels)), labels] = 1.0
    positive_loss = F.softplus(-outputs)
    negative_loss = F.softplus(outputs)
    class_counts = mask.sum(dim=0)
    class_counts[class_counts == 0.0] = 1.0
    negative_risk = (negative_loss * mask).sum(dim=0) / class_counts
    positive_unlabeled_risk = positive_loss.mean(dim=0)
    positive_complementary_risk = (
        positive_loss * mask
    ).sum(dim=0) / class_counts
    selection_probability = outputs.new_full(
        (outputs.shape[1],), 1.0 - 1.0 / outputs.shape[1]
    )
    return (
        (selection_probability * negative_risk).sum()
        + torch.abs(
            positive_unlabeled_risk
            - selection_probability * positive_complementary_risk
        ).sum()
    )


class KMMTest(unittest.TestCase):
    def test_weights_respect_constraints(self):
        train_losses = np.array([0.1, 0.2, 0.7, 1.2]).reshape(-1, 1)
        validation_losses = np.array([0.15, 0.25, 0.35]).reshape(-1, 1)
        kernel_width = get_kernel_width(train_losses)

        weights = kmm(
            train_losses,
            validation_losses,
            kernel_width=kernel_width,
            max_weight=50.0,
            solver='scipy',
        )

        epsilon = (math.sqrt(len(train_losses)) - 1.0) / math.sqrt(
            len(train_losses)
        )
        self.assertEqual(weights.shape, (len(train_losses),))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights >= -1e-7).all())
        self.assertTrue((weights <= 50.0 + 1e-7).all())
        self.assertGreaterEqual(
            weights.sum() + 1e-6,
            len(train_losses) * (1.0 - epsilon),
        )
        self.assertLessEqual(
            weights.sum() - 1e-6,
            len(train_losses) * (1.0 + epsilon),
        )

    def test_constant_losses_have_valid_kernel_width(self):
        width = get_kernel_width(np.ones((4, 1)))
        self.assertEqual(width, 1.0)


class WeightedSCARCETest(unittest.TestCase):
    def test_unit_weights_match_unweighted_loss(self):
        outputs = torch.tensor([
            [1.0, -0.5, 0.2],
            [-0.1, 0.8, 0.4],
            [0.2, 0.1, 1.1],
            [0.7, 0.3, -0.2],
        ])
        labels = torch.tensor([1, 2, 0, 2])

        unweighted = SCARCE_loss(outputs, labels)
        weighted = SCARCE_loss(
            outputs, labels, sample_weight=torch.ones(len(labels))
        )
        reference = reference_scarce_loss(outputs, labels)

        self.assertTrue(torch.allclose(unweighted, weighted, atol=1e-7))
        self.assertTrue(torch.allclose(unweighted, reference, atol=1e-7))

    def test_complementary_nll_is_pointwise_and_finite(self):
        outputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([1, 0])
        losses = complementary_nll(outputs, labels)

        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(torch.isfinite(losses).all().item())
        self.assertTrue((losses >= 0).all().item())


if __name__ == '__main__':
    unittest.main()
