import csv
import os
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.next_round import (
    CSV_FILES,
    build_next_round_ratio_estimator,
    gate_metric_rows,
    initialize_result_files,
    paired_split_hash,
)
from ratio_estimation import process_weight_stages, weight_stage_rows


class FakeWeakDataset(Dataset):
    def __init__(self):
        self.base_indices = torch.tensor([5, 1, 9, 2], dtype=torch.long)
        self.complementary_vectors = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float32
        )

    def __len__(self):
        return len(self.base_indices)

    def __getitem__(self, index):
        return (
            index,
            torch.zeros(1, 2, 2),
            self.complementary_vectors[index],
        )


class FakeScarceModels(object):
    @staticmethod
    def mlp_model(input_dim, hidden_dim, output_dim):
        del output_dim

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu1 = nn.ReLU()

        return Model()


class NextRoundTests(unittest.TestCase):
    def test_ratio_stage_order_is_raw_then_clipped_then_normalized(self):
        raw = torch.tensor(
            [[0.001, 100.0], [2.0, 4.0], [8.0, 0.02], [4.0, 2.0]]
        )
        complements = torch.tensor(
            [[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.float32
        )
        stages = process_weight_stages(
            raw,
            complements,
            min_weight=0.01,
            max_weight=20.0,
            normalization_scope="conditional_branch",
        )
        self.assertTrue(torch.equal(stages["raw"], raw))
        self.assertAlmostEqual(stages["clipped"][0, 0].item(), 0.01, places=6)
        self.assertAlmostEqual(stages["clipped"][0, 1].item(), 20.0, places=6)
        for class_index in range(2):
            for state in range(2):
                mask = complements[:, class_index].long() == state
                self.assertAlmostEqual(
                    stages["normalized"][mask, class_index].mean().item(),
                    1.0,
                    places=6,
                )
        self.assertTrue(
            torch.equal(stages["normalized"], stages["classifier_used"])
        )

    def test_all_normalization_scopes_are_supported(self):
        raw = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
        complements = torch.tensor([[0, 0], [1, 1]])
        for scope in (
            "conditional_branch",
            "class_level",
            "global",
            "none",
        ):
            stages = process_weight_stages(
                raw,
                complements,
                min_weight=0.01,
                max_weight=20.0,
                normalization_scope=scope,
            )
            self.assertTrue(torch.isfinite(stages["classifier_used"]).all())
        global_stages = process_weight_stages(
            raw, complements, normalization_scope="global"
        )
        self.assertAlmostEqual(
            global_stages["normalized"].mean().item(), 1.0, places=6
        )

    def test_weight_rows_contain_required_saturation_diagnostics(self):
        raw = torch.tensor([[0.001, 30.0], [2.0, 4.0]])
        complements = torch.tensor([[0, 1], [0, 1]])
        stages = process_weight_stages(
            raw,
            complements,
            min_weight=0.01,
            max_weight=20.0,
            normalization_scope="class_level",
        )
        rows = weight_stage_rows(
            stages, complements, 1, 0.01, 20.0
        )["raw"]
        fields = set(rows[0])
        for field in (
            "q01",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "q99",
            "lower_clipping_rate",
            "upper_clipping_rate",
            "finite_ratio_rate",
            "log_ratio_mean",
            "log_ratio_std",
            "ess",
        ):
            self.assertIn(field, fields)

    def test_paired_split_hash_is_method_independent(self):
        data = {
            "source_weak": FakeWeakDataset(),
            "target_weak": FakeWeakDataset(),
        }
        shift = {"type": "rotation", "rotation_degrees": 20.0}
        first = paired_split_hash(data, shift)
        second = paired_split_hash(data, shift)
        self.assertEqual(first, second)
        changed = paired_split_hash(
            data, {"type": "rotation", "rotation_degrees": 30.0}
        )
        self.assertNotEqual(first, changed)

    def test_logging_completeness_creates_every_required_file(self):
        with tempfile.TemporaryDirectory() as directory:
            initialize_result_files(directory)
            for filename, required_fields in CSV_FILES.items():
                path = os.path.join(directory, filename)
                self.assertTrue(os.path.isfile(path), filename)
                with open(path, "r", newline="", encoding="utf-8") as stream:
                    fields = next(csv.reader(stream))
                for field in required_fields:
                    self.assertIn(field, fields)

    def test_fusion_gate_rows_are_persistable_and_domain_specific(self):
        complements = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        collected = {}
        for domain in ("source", "target"):
            collected[domain + "_complements"] = complements
            collected[domain + "_gates"] = torch.tensor(
                [[0.05, 0.95], [0.5, 0.2]]
            )
            collected[domain + "_global"] = torch.zeros(2, 2)
            collected[domain + "_conditional"] = torch.ones(2, 2)
            collected[domain + "_fused"] = torch.full((2, 2), 0.5)
        rows = gate_metric_rows(collected, epoch=3, split="validation")
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(row["domain"] for row in rows), {"source", "target"})
        self.assertIn("fraction_gate_lt_0_1", rows[0])
        self.assertIn("abs_fused_minus_conditional", rows[0])

    def test_separate_matched_capacity_is_within_ten_percent(self):
        multihead, multihead_metadata = build_next_round_ratio_estimator(
            FakeScarceModels,
            input_dim=784,
            num_classes=10,
            method="multihead",
            device=torch.device("cpu"),
            ratio_clip_min=0.01,
            ratio_clip_max=20.0,
            normalization="conditional_branch",
            gate_init=0.5,
        )
        matched, matched_metadata = build_next_round_ratio_estimator(
            FakeScarceModels,
            input_dim=784,
            num_classes=10,
            method="separate_matched_capacity",
            device=torch.device("cpu"),
            ratio_clip_min=0.01,
            ratio_clip_max=20.0,
            normalization="conditional_branch",
            gate_init=0.5,
        )
        del multihead, matched
        relative_difference = abs(
            matched_metadata["parameter_count"]
            - multihead_metadata["parameter_count"]
        ) / float(multihead_metadata["parameter_count"])
        self.assertLessEqual(relative_difference, 0.10)


if __name__ == "__main__":
    unittest.main()
