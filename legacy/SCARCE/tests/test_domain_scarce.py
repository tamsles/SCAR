import unittest

import torch

from domain_data import generate_complementary_vectors
from main_domain_scarce import select_training_domain


class DomainSCARCETest(unittest.TestCase):
    def test_train_and_test_select_disjoint_inputs_and_statistics(self):
        train_dataset = object()
        test_dataset = object()
        train_complement_prior = torch.tensor([0.2, 0.8])
        test_complement_prior = torch.tensor([0.7, 0.3])
        train_class_prior = torch.tensor([0.4, 0.6])
        test_class_prior = torch.tensor([0.6, 0.4])
        data = {
            'source_weak': train_dataset,
            'target_weak': test_dataset,
            'source_complement_prior': train_complement_prior,
            'target_complement_prior': test_complement_prior,
            'source_class_prior_oracle': train_class_prior,
            'target_class_prior_oracle': test_class_prior,
        }

        selected_train = select_training_domain(data, 'train')
        selected_test = select_training_domain(data, 'test')

        self.assertIs(selected_train[0], train_dataset)
        self.assertIs(selected_test[0], test_dataset)
        self.assertIs(selected_train[1], train_complement_prior)
        self.assertIs(selected_test[1], test_complement_prior)
        self.assertIs(selected_train[2], train_class_prior)
        self.assertIs(selected_test[2], test_class_prior)

    def test_weak_vectors_do_not_reveal_ordinary_labels(self):
        ordinary_labels = torch.tensor([0, 1, 2, 0, 1, 2])
        vectors = generate_complementary_vectors(
            ordinary_labels, num_classes=3, seed=17
        )

        self.assertTrue(torch.equal(vectors.sum(dim=1), torch.ones(6)))
        self.assertTrue(torch.equal(
            vectors[torch.arange(6), ordinary_labels], torch.zeros(6)
        ))

    def test_pooled_domain_concatenates_without_adiw(self):
        train_dataset = torch.utils.data.TensorDataset(torch.arange(6))
        test_dataset = torch.utils.data.TensorDataset(torch.arange(2))
        data = {
            'source_weak': train_dataset,
            'target_weak': test_dataset,
            'source_complement_prior': torch.tensor([0.2, 0.8]),
            'target_complement_prior': torch.tensor([0.6, 0.4]),
            'source_class_prior_oracle': torch.tensor([0.5, 0.5]),
            'target_class_prior_oracle': torch.tensor([0.7, 0.3]),
        }

        dataset, complement_prior, class_prior = select_training_domain(
            data, 'pooled'
        )

        self.assertIsInstance(dataset, torch.utils.data.ConcatDataset)
        self.assertEqual(len(dataset), 8)
        self.assertTrue(torch.allclose(
            complement_prior, torch.tensor([0.3, 0.7])
        ))
        self.assertTrue(torch.allclose(
            class_prior, torch.tensor([0.55, 0.45])
        ))


if __name__ == '__main__':
    unittest.main()
