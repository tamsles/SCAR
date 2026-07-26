import unittest

import torch

from models import LabelConditionedRatioEstimator
from ratio_estimation.data import unpack_batch
from ratio_estimation.risk import weighted_classwise_risk


class DataAndRiskTests(unittest.TestCase):
    def test_mapping_adapter_and_classwise_risk(self):
        x = torch.randn(5, 3)
        bar_y = torch.randint(0, 2, (5, 2))
        extracted = unpack_batch({"x": x, "bar_y": bar_y})
        self.assertIs(extracted[0], x)
        self.assertIs(extracted[1], bar_y)
        backbone = torch.nn.Linear(3, 4)
        model = LabelConditionedRatioEstimator(backbone, 2, 4, hidden_dims=(8,))
        classwise_loss = torch.ones(5, 2)
        risk = weighted_classwise_risk(model, x, bar_y, classwise_loss)
        self.assertEqual(risk.ndim, 0)
        self.assertTrue(bool(torch.isfinite(risk).item()))

    def test_risk_rejects_sample_level_loss(self):
        x = torch.randn(5, 3)
        bar_y = torch.randint(0, 2, (5, 2))
        model = LabelConditionedRatioEstimator(torch.nn.Linear(3, 4), 2, 4)
        with self.assertRaises(ValueError):
            weighted_classwise_risk(model, x, bar_y, torch.ones(5))


if __name__ == "__main__":
    unittest.main()

