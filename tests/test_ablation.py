import os
import tempfile
import unittest

import torch

from datasets import (
    SyntheticComplementaryDataset,
    empirical_branch_priors,
    synthetic_oracle_log_ratios,
)
from experiments.ablation import aggregate, write_reports


class AblationUtilityTests(unittest.TestCase):
    def test_oracle_log_ratio_is_finite_and_branch_specific(self):
        config = {
            "input_dim": 3,
            "num_classes": 2,
            "target_mean_shift": [0.8, -0.2, 0.0],
            "target_label_shift": [0.8, -0.6],
        }
        source = SyntheticComplementaryDataset(512, 3, 2, 1, [0.0] * 3, 0.0)
        target = SyntheticComplementaryDataset(
            512, 3, 2, 2, config["target_mean_shift"], config["target_label_shift"]
        )
        log_ratio = synthetic_oracle_log_ratios(
            source.x,
            source.bar_y,
            config,
            empirical_branch_priors(source.bar_y),
            empirical_branch_priors(target.bar_y),
        )
        self.assertEqual(tuple(log_ratio.shape), (512, 2))
        self.assertTrue(bool(torch.isfinite(log_ratio).all()))
        self.assertGreater(float((log_ratio[:, 0] - log_ratio[:, 1]).abs().mean()), 0.01)

    def test_aggregate_and_report_files(self):
        base = {
            "name": "proposed",
            "category": "proposed",
            "description": "test",
            "log_ratio_rmse": 0.5,
            "log_ratio_mae": 0.4,
            "log_ratio_correlation": 0.8,
            "moment_rmse": 0.3,
            "source_ratio_mean_abs_error": 0.1,
            "ess_fraction": 0.7,
            "validation_loss": 0.6,
            "validation_accuracy": 0.7,
            "parameter_count": 100,
            "train_seconds": 1.0,
        }
        records = []
        for name, category, offset in (
            ("proposed", "proposed", 0.0),
            ("unit_ratio", "baseline", 0.5),
            ("marginal_ratio", "baseline", 0.3),
        ):
            for seed in (1, 2):
                record = dict(base)
                record.update(
                    {
                        "name": name,
                        "category": category,
                        "description": name,
                        "seed": seed,
                        "log_ratio_rmse": base["log_ratio_rmse"] + offset,
                    }
                )
                records.append(record)
        summaries = aggregate(records)
        self.assertEqual(len(summaries), 3)
        with tempfile.TemporaryDirectory() as directory:
            write_reports(directory, {}, records, summaries)
            self.assertTrue(os.path.isfile(os.path.join(directory, "raw_results.json")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "REPORT.md")))


if __name__ == "__main__":
    unittest.main()
