import unittest
from collections.abc import Mapping

import torch

from models import build_ratio_estimator
from ratio_estimation import (
    RatioEstimatorTrainer,
    build_conditional_masks,
    process_conditional_weights,
    weighted_classwise_risk,
)


def configuration(architecture):
    return {
        "input_dim": 4,
        "num_classes": 3,
        "feature_dim": 8,
        "backbone_hidden_dims": [10],
        "hidden_dims": [12, 6],
        "ratio_arch": architecture,
        "ratio": {
            "self_normalize": True,
            "normalization_scope": "conditional_branch",
            "min_weight": 0.01,
            "max_weight": 20.0,
            "min_condition_samples": 2,
            "eps": 1.0e-8,
            "detach_classifier_update": True,
        },
        "fusion": {
            "gate_regularization": 0.01,
            "gate_init": 0.7,
            "detach_global_for_gate": False,
        },
    }


def balanced_labels(repeats=4):
    pattern = torch.tensor(
        [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    )
    return pattern.repeat(repeats, 1)


class LabelLeakTrap(Mapping):
    def __init__(self, x, bar_y):
        self.values = {"x": x, "bar_y": bar_y, "domain": "source"}

    def __getitem__(self, key):
        if key in ("y", "label", "true_label"):
            raise AssertionError("ordinary label was read")
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class ConditionalArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        self.x = torch.randn(16, 4)
        self.y_bar = balanced_labels()

    def _trainer(self, model):
        return RatioEstimatorTrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=0.005),
            device=torch.device("cpu"),
            min_branch_count=2,
            branch_balancing="balanced_bce",
        )

    def test_shape_selection_and_positivity_for_every_architecture(self):
        for architecture in ("unified", "separate", "multihead", "fusion"):
            with self.subTest(architecture=architecture):
                model = build_ratio_estimator(configuration(architecture))
                all_ratios = model.predict_all_ratios(self.x)
                selected = model.select_observed_ratios(self.x, self.y_bar)
                expected = all_ratios.gather(
                    2, self.y_bar.unsqueeze(-1)
                ).squeeze(-1)
                self.assertEqual(tuple(all_ratios.shape), (16, 3, 2))
                self.assertEqual(tuple(selected.shape), (16, 3))
                self.assertTrue(torch.allclose(selected, expected))
                self.assertTrue(bool(torch.isfinite(all_ratios).all().item()))
                self.assertTrue(bool((all_ratios > 0).all().item()))

    def test_conditional_masks_use_only_matching_complement_states(self):
        source = torch.tensor([[0, 1], [1, 1], [0, 0]])
        target = torch.tensor([[1, 0], [1, 1], [0, 1]])
        masks = build_conditional_masks(source, target, 2)
        self.assertTrue(
            torch.equal(masks[(0, 0)][0], torch.tensor([True, False, True]))
        )
        self.assertTrue(
            torch.equal(masks[(1, 1)][1], torch.tensor([False, True, True]))
        )

    def test_parameter_sharing_matches_architecture_contract(self):
        separate = build_ratio_estimator(configuration("separate"))
        left = next(separate.branch_estimators[0][0].parameters())
        right = next(separate.branch_estimators[0][1].parameters())
        self.assertNotEqual(left.data_ptr(), right.data_ptr())
        trainer = self._trainer(separate)
        self.assertEqual(len(trainer.branch_optimizers), 6)
        self.assertEqual(len({id(item) for item in trainer.branch_optimizers}), 6)

        multihead = build_ratio_estimator(configuration("multihead"))
        self.assertIsNot(
            multihead.ratio_heads[0][0], multihead.ratio_heads[0][1]
        )
        self.assertTrue(hasattr(multihead, "ratio_encoder"))

        fusion = build_ratio_estimator(configuration("fusion"))
        outputs = fusion.ratio_training_outputs(self.x)
        self.assertEqual(tuple(outputs["gates"].shape), (16, 3, 2))
        self.assertAlmostEqual(
            float(outputs["gates"].mean().item()), 0.7, places=5
        )

    def test_empty_branch_skips_update_and_uses_unit_fallback(self):
        model = build_ratio_estimator(configuration("separate"))
        trainer = self._trainer(model)
        all_zero = torch.zeros(16, 3, dtype=torch.long)
        before = [
            parameter.detach().clone()
            for parameter in model.branch_estimators[0][1].parameters()
        ]
        metrics = trainer.train_step(
            (self.x, all_zero), (self.x + 0.4, all_zero)
        )
        after = list(model.branch_estimators[0][1].parameters())
        self.assertFalse(metrics["has_nan_or_inf"])
        self.assertEqual(int(model.skipped_update_count[0, 1]), 1)
        self.assertFalse(bool(model.branch_initialized[0, 1]))
        for old, new in zip(before, after):
            self.assertTrue(torch.equal(old, new))
        query_y = torch.ones(16, 3, dtype=torch.long)
        fallback = model.get_conditional_weights(
            self.x, query_y, self_normalize=False
        )
        self.assertTrue(torch.equal(fallback, torch.ones_like(fallback)))

    def test_branch_normalization_and_statistics(self):
        raw = torch.exp(torch.randn(16, 3))
        processed, statistics = process_conditional_weights(
            raw,
            self.y_bar,
            min_weight=0.01,
            max_weight=20.0,
            self_normalize=True,
        )
        for k in range(3):
            for b in range(2):
                branch = processed[self.y_bar[:, k] == b, k]
                self.assertAlmostEqual(float(branch.mean()), 1.0, places=5)
                entry = statistics["{}_{}".format(k, b)]
                self.assertGreater(entry["ess"], 0.0)
                self.assertIsNotNone(entry["q50"])

    def test_classifier_backward_is_detached_from_ratio_model(self):
        model = build_ratio_estimator(configuration("multihead"))
        model.branch_initialized.fill_(True)
        classifier = torch.nn.Linear(4, 3)
        logits = classifier(self.x)
        classwise_loss = (logits - self.y_bar.float()).square()
        risk = weighted_classwise_risk(
            model,
            self.x,
            self.y_bar,
            classwise_loss,
            ratio_config=configuration("multihead")["ratio"],
        )
        risk.backward()
        self.assertIsNotNone(classifier.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_training_batch_never_reads_an_ordinary_label(self):
        model = build_ratio_estimator(configuration("multihead"))
        trainer = self._trainer(model)
        source = LabelLeakTrap(self.x, self.y_bar)
        target = LabelLeakTrap(self.x + 0.2, self.y_bar)
        metrics = trainer.train_step(source, target)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])))

    def test_one_epoch_smoke_for_every_architecture(self):
        for architecture in ("unified", "separate", "multihead", "fusion"):
            with self.subTest(architecture=architecture):
                model = build_ratio_estimator(configuration(architecture))
                trainer = self._trainer(model)
                history = trainer.fit(
                    [(self.x, self.y_bar)],
                    [(self.x + 0.5, self.y_bar)],
                    epochs=1,
                    verbose=False,
                )
                ratios = model.predict_all_ratios(self.x)
                self.assertEqual(len(history), 1)
                self.assertFalse(history[0]["has_nan_or_inf"])
                self.assertTrue(bool(torch.isfinite(ratios).all().item()))

    def test_fusion_reports_gate_and_component_statistics(self):
        model = build_ratio_estimator(configuration("fusion"))
        statistics = model.extra_ratio_statistics(self.x)
        self.assertEqual(len(statistics["gate_mean"]), 3)
        self.assertEqual(len(statistics["gate_histogram"]), 10)
        self.assertIn("global_ratio", statistics)
        self.assertIn("conditional_ratio", statistics)
        self.assertIn("fused_ratio", statistics)


if __name__ == "__main__":
    unittest.main()
