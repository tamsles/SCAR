import unittest

import torch

from domain_data import (
    ShiftedEvaluationDataset,
    WeakDomainDataset,
    generate_complementary_vectors,
)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = torch.arange(24, dtype=torch.float32).view(6, 1, 2, 2)
        self.targets = torch.tensor([0, 1, 2, 0, 1, 2])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


class DomainDataTest(unittest.TestCase):
    def test_weak_dataset_does_not_return_ordinary_label(self):
        base = DummyDataset()
        indices = torch.tensor([0, 2, 4])
        vectors = generate_complementary_vectors(
            base.targets[indices], num_classes=3, seed=4
        )
        weak = WeakDomainDataset(base, indices, vectors)

        item = weak[1]
        self.assertEqual(len(item), 3)
        self.assertEqual(item[0], 1)
        self.assertEqual(tuple(item[1].shape), (1, 2, 2))
        self.assertEqual(tuple(item[2].shape), (3,))

    def test_target_evaluation_applies_label_permutation(self):
        shifted = ShiftedEvaluationDataset(
            DummyDataset(), num_classes=3, label_shift=1
        )
        _, label = shifted[2]
        self.assertEqual(label, 0)

    def test_target_evaluation_composes_rotation_and_label_shift(self):
        shifted = ShiftedEvaluationDataset(
            DummyDataset(),
            num_classes=3,
            rotation_degrees=30.0,
            label_shift=1,
        )
        image, label = shifted[2]

        self.assertEqual(tuple(image.shape), (1, 2, 2))
        self.assertEqual(label, 0)

    def test_independent_scar_allows_zero_or_multiple_labels(self):
        labels = torch.tensor([0, 1, 2] * 20)
        vectors = generate_complementary_vectors(
            labels,
            num_classes=3,
            generation='independent',
            selection_probability=0.5,
            seed=9,
        )
        row_counts = vectors.sum(dim=1)

        self.assertTrue((row_counts == 0).any().item())
        self.assertTrue((row_counts > 1).any().item())
        self.assertTrue(torch.equal(
            vectors[torch.arange(len(labels)), labels],
            torch.zeros(len(labels)),
        ))


if __name__ == '__main__':
    unittest.main()
