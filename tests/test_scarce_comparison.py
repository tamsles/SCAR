import csv
import json
import os
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.scarce_ratio_comparison import (
    ScarceMLPFeatureBackbone,
    WeakPairView,
)
from experiments.summarize_scarce_methods import summarize


class FakeWeakDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index, torch.full((1, 2, 2), float(index)), torch.tensor([0, 1])


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu1 = nn.ReLU()


class ScarceComparisonTests(unittest.TestCase):
    def test_zero_copy_weak_pair_view(self):
        view = WeakPairView(FakeWeakDataset())
        image, bar_y = view[2]
        self.assertEqual(tuple(image.shape), (1, 2, 2))
        self.assertTrue(torch.equal(bar_y, torch.tensor([0, 1])))

    def test_original_mlp_feature_layer_is_reused(self):
        original = FakeMLP()
        backbone = ScarceMLPFeatureBackbone(original)
        self.assertIs(backbone.fc1, original.fc1)
        output = backbone(torch.randn(5, 1, 2, 2))
        self.assertEqual(tuple(output.shape), (5, 3))

    def test_summarizer_combines_original_and_neural_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = os.path.join(directory, "baseline.csv")
            fields = [
                "method",
                "num_seeds",
                "mean_accuracy",
                "std_accuracy",
                "delta_vs_adiw",
                "paired_delta_std",
                "seed_accuracies",
                "paired_seed_deltas",
            ]
            baseline_rows = [
                ("ADIW-SCAR (train + test)", "60;61"),
                ("SCAR (pooled train + test, no ADIW)", "59;60"),
                ("SCAR (train only)", "54;55"),
                ("SCAR (test only)", "38;39"),
            ]
            with open(baseline_path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for method, values in baseline_rows:
                    writer.writerow(
                        {
                            "method": method,
                            "num_seeds": 2,
                            "mean_accuracy": 0,
                            "std_accuracy": 0,
                            "delta_vs_adiw": 0,
                            "paired_delta_std": 0,
                            "seed_accuracies": values,
                            "paired_seed_deltas": "0;0",
                        }
                    )
            for seed, accuracy in ((0, 62.0), (1, 63.0)):
                result_path = os.path.join(
                    directory, "conditioned_ratio_scar_seed{}.json".format(seed)
                )
                with open(result_path, "w") as stream:
                    json.dump(
                        {
                            "method": "conditioned_ratio_scar",
                            "seed": seed,
                            "target_accuracy": accuracy,
                        },
                        stream,
                    )
            rows = summarize(baseline_path, directory)
            conditioned = next(
                row for row in rows if row["method"] == "Conditioned-Ratio-SCAR"
            )
            self.assertAlmostEqual(conditioned["mean_accuracy"], 62.5)
            self.assertAlmostEqual(conditioned["delta_vs_pooled"], 3.0)

    def test_summarizer_rejects_a_wrong_neural_seed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = os.path.join(directory, "baseline.csv")
            fields = ["method", "seed_accuracies"]
            with open(baseline_path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for method in (
                    "ADIW-SCAR (train + test)",
                    "SCAR (pooled train + test, no ADIW)",
                ):
                    writer.writerow(
                        {"method": method, "seed_accuracies": "60;61"}
                    )
            for seed in (1, 2):
                result_path = os.path.join(
                    directory, "conditioned_ratio_scar_seed{}.json".format(seed)
                )
                with open(result_path, "w") as stream:
                    json.dump(
                        {
                            "method": "conditioned_ratio_scar",
                            "seed": seed,
                            "target_accuracy": 60.0 + seed,
                        },
                        stream,
                    )
            with self.assertRaisesRegex(ValueError, "has seeds"):
                summarize(baseline_path, directory)


if __name__ == "__main__":
    unittest.main()
