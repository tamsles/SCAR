import math
import unittest
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import LabelConditionedRatioEstimator
from ratio_estimation import RatioEstimatorTrainer


class CountingBackbone(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, feature_dim)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tanh(self.linear(x))


def make_model(estimator_type="conditioned", input_dim=4, q=3):
    backbone = CountingBackbone(input_dim, 8)
    model = LabelConditionedRatioEstimator(
        backbone=backbone,
        num_classes=q,
        feature_dim=8,
        class_embedding_dim=5,
        state_embedding_dim=3,
        hidden_dims=(16,),
        estimator_type=estimator_type,
        log_ratio_clip_min=-6.0,
        log_ratio_clip_max=6.0,
        ratio_clip_min=1.0e-4,
        ratio_clip_max=1.0e4,
    )
    return model, backbone


class RatioEstimatorInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_shapes_conditioned_and_multihead(self):
        x = torch.randn(7, 4)
        bar_y = torch.randint(0, 2, (7, 3))
        for estimator_type in ("conditioned", "multihead"):
            model, _ = make_model(estimator_type)
            self.assertEqual(tuple(model.forward_all(x).shape), (7, 3, 2))
            self.assertEqual(tuple(model.forward_selected(x, bar_y).shape), (7, 3))
            self.assertEqual(tuple(model.estimate_all_ratios(x).shape), (7, 3, 2))
            self.assertEqual(
                tuple(model.estimate_selected_ratios(x, bar_y).shape), (7, 3)
            )

    def test_ratios_are_positive_and_finite(self):
        model, _ = make_model()
        logits = torch.tensor(
            [[[-1.0e20, 1.0e20], [float("nan"), 0.0], [2.0, -2.0]]]
        )
        ratios = model.ratios_from_logits(logits)
        self.assertTrue(bool(torch.isfinite(ratios).all().item()))
        self.assertTrue(bool((ratios > 0).all().item()))

    def test_vectorized_backbone_call(self):
        model, backbone = make_model()
        x = torch.randn(9, 4)
        bar_y = torch.randint(0, 2, (9, 3))
        model.forward_all(x)
        self.assertEqual(backbone.calls, 1)
        backbone.calls = 0
        model.forward_selected(x, bar_y)
        self.assertEqual(backbone.calls, 1)

    def test_zero_embedding_dimensions_remove_conditions(self):
        backbone = CountingBackbone(4, 8)
        model = LabelConditionedRatioEstimator(
            backbone,
            num_classes=3,
            feature_dim=8,
            class_embedding_dim=0,
            state_embedding_dim=0,
            hidden_dims=(8,),
        )
        logits = model.forward_all(torch.randn(5, 4))
        reference = logits[:, :1, :1].expand_as(logits)
        self.assertTrue(torch.allclose(logits, reference))

    def test_gradients_reach_all_conditioned_components(self):
        model, backbone = make_model()
        x = torch.randn(12, 4)
        pattern = torch.tensor([[0, 1, 0], [1, 0, 1]])
        bar_y = pattern.repeat(6, 1)
        loss = model.forward_selected(x, bar_y).square().mean()
        loss.backward()
        self.assertIsNotNone(backbone.linear.weight.grad)
        self.assertGreater(float(backbone.linear.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.class_embedding.weight.grad)
        self.assertGreater(float(model.class_embedding.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.state_embedding.weight.grad)
        self.assertGreater(float(model.state_embedding.weight.grad.abs().sum()), 0.0)
        head_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.ratio_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(head_grad, 0.0)

    def test_prior_correction_balanced_and_unbalanced(self):
        model, _ = make_model(q=2)
        source = torch.tensor([[80.0, 20.0], [50.0, 50.0]])
        target = torch.tensor([[20.0, 80.0], [50.0, 50.0]])
        model.set_domain_counts(source, target, balanced_sampling=False)
        expected = torch.log(source / target)
        self.assertTrue(torch.allclose(model.log_prior_correction, expected))
        ratios = model.ratios_from_logits(torch.zeros(1, 2, 2))
        self.assertTrue(torch.allclose(ratios[0], source / target, atol=1.0e-6))
        model.set_domain_counts(source, target, balanced_sampling=True)
        self.assertTrue(torch.equal(model.log_prior_correction, torch.zeros(2, 2)))

    def test_importance_weights_self_normalize_per_branch(self):
        model, _ = make_model(q=3)
        x = torch.randn(20, 4)
        bar_y = torch.randint(0, 2, (20, 3))
        weights = model.compute_importance_weights(
            x, bar_y, detach=True, self_normalize=True
        )
        for k in range(3):
            for z in range(2):
                selected = weights[bar_y[:, k] == z, k]
                if selected.numel() > 0:
                    self.assertAlmostEqual(float(selected.mean()), 1.0, places=5)


class RatioEstimatorTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def _trainer(self, model, lr=0.02, min_branch_count=1):
        return RatioEstimatorTrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=lr),
            device=torch.device("cpu"),
            branch_balancing="balanced_bce",
            min_branch_count=min_branch_count,
            balanced_domain_sampling=False,
        )

    def test_identical_distribution_learns_ratio_near_one(self):
        model, _ = make_model(input_dim=2, q=2)
        trainer = self._trainer(model)
        x = torch.randn(192, 2)
        bar_y = torch.randint(0, 2, (192, 2))
        batch = (x, bar_y)
        model.set_domain_counts(torch.full((2, 2), 96), torch.full((2, 2), 96))
        for _ in range(80):
            metrics = trainer.train_step(batch, batch)
        ratios = model.estimate_selected_ratios(x, bar_y)
        self.assertFalse(metrics["has_nan_or_inf"])
        self.assertLess(abs(float(ratios.mean()) - 1.0), 0.20)

    def test_known_shift_produces_branch_specific_ratios(self):
        model, _ = make_model(input_dim=1, q=2)
        trainer = self._trainer(model, lr=0.015)
        repeats = 48
        patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).repeat(
            repeats, 1
        )
        state_zero = (patterns[:, 0] == 0).float()
        source_mean = torch.where(state_zero > 0, -2.0, 2.0)
        target_mean = -source_mean
        source_x = source_mean[:, None] + 0.25 * torch.randn(len(patterns), 1)
        target_x = target_mean[:, None] + 0.25 * torch.randn(len(patterns), 1)
        source_batch = (source_x, patterns)
        target_batch = (target_x, patterns)
        model.set_domain_counts(
            torch.full((2, 2), len(patterns) // 2),
            torch.full((2, 2), len(patterns) // 2),
        )
        for _ in range(140):
            trainer.train_step(source_batch, target_batch)
        evaluation = torch.tensor([[2.0]])
        ratios = model.estimate_all_ratios(evaluation)[0]
        self.assertGreater(float(ratios[0, 0]), 2.0)
        self.assertLess(float(ratios[0, 1]), 0.5)
        self.assertGreater(float(ratios[0, 0] / ratios[0, 1]), 8.0)

    def test_missing_branch_is_safe_and_reported(self):
        model, _ = make_model(input_dim=2, q=2)
        trainer = self._trainer(model, min_branch_count=2)
        x_source = torch.randn(32, 2)
        x_target = torch.randn(32, 2)
        all_zero = torch.zeros(32, 2, dtype=torch.long)
        source_loader = DataLoader(TensorDataset(x_source, all_zero), batch_size=16)
        target_loader = DataLoader(TensorDataset(x_target, all_zero), batch_size=16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            statistics = trainer.compute_branch_statistics(
                source_loader, target_loader
            )
        self.assertTrue(caught)
        self.assertEqual(statistics["source_count"], [[32, 0], [32, 0]])
        metrics = trainer.train_step((x_source, all_zero), (x_target, all_zero))
        self.assertFalse(metrics["has_nan_or_inf"])
        self.assertEqual(metrics["active_branches"], [[True, False], [True, False]])

    def test_balanced_bce_uses_effective_unit_prior(self):
        model, _ = make_model(input_dim=2, q=2)
        trainer = self._trainer(model)
        source_y = torch.tensor([[0, 0]] * 8 + [[1, 1]] * 2)
        target_y = torch.tensor([[0, 0]] * 2 + [[1, 1]] * 8)
        source_loader = DataLoader(
            TensorDataset(torch.randn(10, 2), source_y), batch_size=5
        )
        target_loader = DataLoader(
            TensorDataset(torch.randn(10, 2), target_y), batch_size=5
        )
        trainer.compute_branch_statistics(source_loader, target_loader)
        self.assertTrue(
            torch.equal(model.log_prior_correction, torch.zeros(2, 2))
        )


if __name__ == "__main__":
    unittest.main()
