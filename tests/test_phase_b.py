import unittest

import torch

from src.ratio_diagnostics import ratio_components_from_logits, weighted_mmd
from src.risk_recovery_metrics import paired_interval, risk_recovery_summary
from src.weight_interventions import apply_weight_intervention
from src.synthetic_gaussian import (
    generate_gaussian_shift,
    oracle_branch_ratios,
    oracle_global_ratio,
)


class PhaseBUtilityTests(unittest.TestCase):
    def test_weight_interventions_preserve_the_declared_invariants(self):
        weights = torch.tensor(
            [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]], dtype=torch.float32
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        shuffled = apply_weight_intervention(
            weights, "shuffle", generator=generator
        )
        self.assertTrue(
            torch.allclose(
                torch.sort(shuffled, dim=0).values,
                torch.sort(weights, dim=0).values,
            )
        )
        mean = apply_weight_intervention(weights, "mean")
        self.assertTrue(torch.allclose(mean, torch.full_like(mean, 1.75)))
        self.assertTrue(
            torch.equal(
                apply_weight_intervention(weights, "ones"),
                torch.ones_like(weights),
            )
        )

    def test_domain_ratio_direction_and_label_reversal(self):
        # Equal-variance Gaussian log density ratio:
        # log N(x;mu_t,1)/N(x;mu_s,1).
        x = torch.tensor([-2.0, 0.0, 2.0])
        source_mean = 0.0
        target_mean = 1.0
        gaussian_log_ratio = (
            -0.5 * (x - target_mean).square()
            + 0.5 * (x - source_mean).square()
        )
        equal = ratio_components_from_logits(torch.zeros(16))
        self.assertTrue(
            torch.allclose(
                equal["unclipped_ratio"], torch.ones(16), atol=1.0e-6
            )
        )
        target_dense = ratio_components_from_logits(gaussian_log_ratio[-1:])
        reversed_labels = ratio_components_from_logits(
            -gaussian_log_ratio[-1:]
        )
        self.assertGreater(
            float(target_dense["unclipped_ratio"].item()), 1.0
        )
        self.assertTrue(
            torch.allclose(
                target_dense["unclipped_ratio"]
                * reversed_labels["unclipped_ratio"],
                torch.ones(1),
                atol=1.0e-5,
            )
        )

    def test_paired_and_risk_recovery_statistics(self):
        interval = paired_interval([1.0, 2.0, 3.0], bootstrap_samples=100)
        self.assertEqual(interval["count"], 3)
        self.assertAlmostEqual(interval["mean_difference"], 2.0)
        recovery = risk_recovery_summary(
            [0.5, 0.2, 0.4], [0.6, 0.3, 0.5]
        )
        self.assertAlmostEqual(recovery["bias"], 0.1)
        self.assertEqual(recovery["selection_regret"], 0.0)

    def test_gaussian_no_shift_oracle_ratios_are_one(self):
        data = generate_gaussian_shift(
            "no_shift",
            seed=3,
            source_size=128,
            target_weak_size=64,
            target_test_size=64,
        )
        global_ratio = oracle_global_ratio(
            data["source"]["x"], data["parameters"]
        )
        branch_ratio = oracle_branch_ratios(
            data["source"]["x"],
            data["source"]["bar_y"],
            data["parameters"],
        )
        self.assertTrue(torch.allclose(global_ratio, torch.ones_like(global_ratio)))
        self.assertTrue(
            torch.allclose(branch_ratio, torch.ones_like(branch_ratio))
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_weighted_mmd_aligns_gpu_weights_with_cpu_features(self):
        source = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        target = source + 0.25
        cpu_weights = torch.tensor([0.5, 1.0, 1.5, 1.0])
        expected = weighted_mmd(source, target, cpu_weights, seed=7)
        actual = weighted_mmd(
            source, target, cpu_weights.cuda(), seed=7
        )
        self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
