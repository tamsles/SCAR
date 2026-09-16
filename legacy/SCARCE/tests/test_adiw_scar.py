import unittest

import torch
import torch.nn.functional as F

from adiw_scar import (
    BranchWeightMemory,
    adiw_kmm_pgd,
    branch_prior_ratios,
    combined_adiw_scar_loss,
    joint_to_conditional_weights,
    loss_value_representation,
    mmd_weight_objective,
    project_kmm_weights,
    rbf_gamma_from_quantile,
    scar_branch_risk,
)
from domain_data import generate_complementary_vectors
from models import linear_model
from utils_algo import SCARCE_loss


class ADIWSCARMathTest(unittest.TestCase):
    def test_single_complement_vectors_never_include_true_class(self):
        labels = torch.tensor([0, 1, 2, 0, 2, 1])
        vectors = generate_complementary_vectors(
            labels, num_classes=3, generation='single', seed=7
        )

        self.assertEqual(tuple(vectors.shape), (6, 3))
        self.assertTrue(torch.equal(vectors.sum(dim=1), torch.ones(6)))
        self.assertTrue(
            torch.equal(
                vectors[torch.arange(len(labels)), labels],
                torch.zeros(len(labels)),
            )
        )

    def test_branch_risk_matches_original_scarce_without_shift(self):
        outputs = torch.tensor([
            [0.2, -0.4, 0.9],
            [-0.7, 0.1, 0.4],
            [0.5, 0.6, -0.2],
            [-0.1, 0.8, -0.6],
            [0.9, -0.3, 0.2],
            [-0.5, 0.7, 0.3],
        ])
        complementary_labels = torch.tensor([0, 1, 2, 0, 1, 2])
        complementary_vectors = F.one_hot(
            complementary_labels, num_classes=3
        ).float()
        complement_prior = complementary_vectors.mean(dim=0)
        class_prior = torch.full((3,), 1.0 / 3.0)

        branch_loss, _ = scar_branch_risk(
            outputs,
            complementary_vectors,
            class_prior,
            complement_prior,
            branch_weights=torch.ones_like(outputs),
            correction='abs',
        )
        original_loss = SCARCE_loss(outputs, complementary_labels)

        self.assertTrue(
            torch.allclose(branch_loss, original_loss, atol=1e-6)
        )

    def test_projection_satisfies_box_and_sum_constraints(self):
        projected = project_kmm_weights(
            torch.tensor([-4.0, 0.1, 8.0, 20.0]),
            max_weight=3.0,
            sum_tolerance=0.1,
        )

        self.assertTrue((projected >= 0.0).all().item())
        self.assertTrue((projected <= 3.0).all().item())
        self.assertGreaterEqual(projected.sum().item(), 3.6 - 1e-5)
        self.assertLessEqual(projected.sum().item(), 4.4 + 1e-5)

    def test_pgd_does_not_increase_mmd_objective(self):
        source = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        target = torch.tensor([[2.0], [2.5], [3.0], [4.0]])
        initial = torch.ones(4)
        gamma = rbf_gamma_from_quantile(source, target, quantile=0.5)
        before = mmd_weight_objective(source, target, initial, gamma)
        updated = adiw_kmm_pgd(
            source,
            target,
            initial,
            step_size=1.0,
            num_steps=20,
            max_weight=5.0,
            kernel_quantile=0.5,
            sum_tolerance=0.5,
        )
        after = mmd_weight_objective(source, target, updated, gamma)

        self.assertLessEqual(after.item(), before.item() + 1e-7)
        self.assertTrue((updated >= 0.0).all().item())
        self.assertTrue((updated <= 5.0).all().item())

    def test_joint_ratio_decomposition_recovers_conditionals(self):
        source_prior = torch.tensor([0.2, 0.4, 0.6])
        target_prior = torch.tensor([0.3, 0.2, 0.5])
        positive_rho, negative_rho = branch_prior_ratios(
            source_prior, target_prior
        )
        conditional = torch.tensor([0.5, 1.0, 1.5])

        recovered_positive = joint_to_conditional_weights(
            conditional * positive_rho, positive_rho
        )
        recovered_negative = joint_to_conditional_weights(
            conditional * negative_rho, negative_rho
        )
        self.assertTrue(torch.allclose(recovered_positive, conditional))
        self.assertTrue(torch.allclose(recovered_negative, conditional))

    def test_combined_objective_endpoints(self):
        source_outputs = torch.tensor([
            [0.2, -0.4], [-0.3, 0.7], [0.1, -0.5], [0.8, -0.2]
        ])
        target_outputs = torch.tensor([
            [-0.1, 0.6], [0.5, -0.7], [-0.4, 0.2], [0.3, -0.2]
        ])
        source_complements = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]
        ])
        target_complements = torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]
        ])
        class_prior = torch.tensor([0.5, 0.5])
        complement_prior = torch.tensor([0.5, 0.5])
        weights = torch.ones_like(source_outputs)

        source_only, source_diagnostics = combined_adiw_scar_loss(
            source_outputs,
            source_complements,
            weights,
            target_outputs,
            target_complements,
            class_prior,
            complement_prior,
            target_fraction=0.0,
        )
        target_only, target_diagnostics = combined_adiw_scar_loss(
            source_outputs,
            source_complements,
            weights,
            target_outputs,
            target_complements,
            class_prior,
            complement_prior,
            target_fraction=1.0,
        )

        self.assertTrue(torch.allclose(
            source_only, source_diagnostics['source_risk']
        ))
        self.assertTrue(torch.allclose(
            target_only, target_diagnostics['target_risk']
        ))


class ADIWSCARIntegrationTest(unittest.TestCase):
    def test_one_training_step_uses_only_complement_vectors(self):
        torch.manual_seed(11)
        num_classes = 3
        source_images = torch.randn(12, 1, 2, 2)
        target_images = torch.randn(9, 1, 2, 2) + 0.5
        source_true_labels = torch.tensor([0, 1, 2] * 4)
        target_true_labels = torch.tensor([2, 1, 0] * 3)
        source_complements = generate_complementary_vectors(
            source_true_labels, num_classes, seed=3
        )
        target_complements = generate_complementary_vectors(
            target_true_labels, num_classes, seed=5
        )
        source_prior = source_complements.mean(dim=0)
        target_prior = target_complements.mean(dim=0)
        class_prior = torch.full((num_classes,), 1.0 / num_classes)

        model = linear_model(input_dim=4, output_dim=num_classes)
        source_outputs = model(source_images)
        target_outputs = model(target_images)
        memory = BranchWeightMemory(12, num_classes, torch.device('cpu'))
        weights = memory.estimate(
            torch.arange(12),
            loss_value_representation(source_outputs.detach()),
            source_complements,
            loss_value_representation(target_outputs.detach()),
            target_complements,
            source_prior,
            target_prior,
            num_steps=2,
        )
        loss, _ = combined_adiw_scar_loss(
            source_outputs,
            source_complements,
            weights,
            target_outputs,
            target_complements,
            class_prior,
            target_prior,
            target_fraction=0.25,
        )
        loss.backward()

        self.assertEqual(tuple(weights.shape), (12, num_classes))
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )


if __name__ == '__main__':
    unittest.main()
