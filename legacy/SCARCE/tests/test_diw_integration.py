import types
import unittest

import numpy as np
import torch

from main import estimate_batch_weights
from models import linear_model
from utils_algo import chosen_loss_c
from utils_data import prepare_train_loaders


class DIWIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.data = torch.randn(15, 1, 2, 2)
        self.labels = torch.tensor([0, 1, 2] * 5)
        self.ordinary_dataset = torch.utils.data.TensorDataset(
            self.data, self.labels
        )

    def full_loader(self):
        return torch.utils.data.DataLoader(
            self.ordinary_dataset,
            batch_size=len(self.ordinary_dataset),
            shuffle=False,
        )

    def test_scarce_path_keeps_all_training_samples(self):
        (
            _,
            complementary_loader,
            validation_loader,
            complementary_class_prior,
            input_dim,
        ) = prepare_train_loaders(
            full_train_loader=self.full_loader(),
            batch_size=4,
            ordinary_train_dataset=self.ordinary_dataset,
            complementary_type='random',
            seed=7,
            num_val=0,
        )

        self.assertEqual(len(complementary_loader.dataset), 15)
        self.assertIsNone(validation_loader)
        self.assertEqual(input_dim, 4)
        self.assertEqual(complementary_class_prior.shape, (3,))

    def test_synthetic_diw_training_step(self):
        (
            _,
            complementary_loader,
            validation_loader,
            complementary_class_prior,
            input_dim,
        ) = prepare_train_loaders(
            full_train_loader=self.full_loader(),
            batch_size=4,
            ordinary_train_dataset=self.ordinary_dataset,
            complementary_type='random',
            seed=7,
            num_val=3,
            validation_batch_size=3,
        )

        self.assertEqual(len(complementary_loader.dataset), 12)
        self.assertEqual(len(validation_loader.dataset), 3)
        self.assertEqual(input_dim, 4)
        self.assertEqual(complementary_class_prior.shape, (3,))
        self.assertAlmostEqual(complementary_class_prior.sum(), 1.0)

        model = linear_model(input_dim=input_dim, output_dim=3)
        train_images, complementary_labels = next(iter(complementary_loader))
        validation_batch = next(iter(validation_loader))
        args = types.SimpleNamespace(
            diw_kernel_quantile=0.01,
            diw_max_weight=50.0,
            kmm_solver='auto',
        )
        weights = estimate_batch_weights(
            model,
            train_images,
            complementary_labels,
            validation_batch,
            args,
            torch.device('cpu'),
        )

        self.assertEqual(tuple(weights.shape), (len(train_images),))
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertTrue((weights >= 0).all().item())

        model.train()
        outputs = model(train_images)
        loss, loss_vector = chosen_loss_c(
            f=outputs,
            K=3,
            labels=complementary_labels,
            ccp=np.asarray(complementary_class_prior),
            meta_method='DIW',
            device=torch.device('cpu'),
            sample_weight=weights,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNone(loss_vector)
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )


if __name__ == '__main__':
    unittest.main()
