"""Dual-domain weak-label datasets for ADIW-SCAR experiments."""

import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.transforms.functional as transform_functional


def parse_probability_vector(value, num_classes):
    """Parse a scalar or comma-separated class-wise SCAR selection rate."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        probabilities = np.asarray([float(part) for part in parts])
    else:
        probabilities = np.asarray(value, dtype=np.float64).reshape(-1)
    if probabilities.size == 1:
        probabilities = np.repeat(probabilities, num_classes)
    if probabilities.size != num_classes:
        raise ValueError(
            "selection probability must be scalar or contain K values"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("selection probability contains a non-finite value")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(
            "selection probabilities must be between zero and one"
        )
    return probabilities


def generate_complementary_vectors(
        ordinary_labels,
        num_classes,
        generation="single",
        selection_probability=None,
        seed=0):
    """Generate single or generalized SCAR complementary-label vectors."""
    labels = torch.as_tensor(ordinary_labels, dtype=torch.long).view(-1)
    if labels.numel() == 0:
        raise ValueError("ordinary_labels must be non-empty")
    if labels.min().item() < 0 or labels.max().item() >= num_classes:
        raise ValueError("ordinary label outside the configured label space")
    if generation not in ("single", "independent"):
        raise ValueError("generation must be single or independent")

    random_state = np.random.RandomState(seed)
    labels_numpy = labels.cpu().numpy()
    vectors = np.zeros((len(labels_numpy), num_classes), dtype=np.float32)
    if generation == "single":
        sampled = random_state.randint(0, num_classes - 1, len(labels_numpy))
        sampled += (sampled >= labels_numpy).astype(np.int64)
        vectors[np.arange(len(labels_numpy)), sampled] = 1.0
    else:
        if selection_probability is None:
            selection_probability = 1.0 / float(num_classes - 1)
        probabilities = parse_probability_vector(
            selection_probability, num_classes
        )
        vectors = (
            random_state.rand(len(labels_numpy), num_classes)
            < probabilities.reshape(1, -1)
        ).astype(np.float32)
        vectors[np.arange(len(labels_numpy)), labels_numpy] = 0.0

    if vectors[np.arange(len(labels_numpy)), labels_numpy].any():
        raise RuntimeError(
            "a true class was selected as a complementary label"
        )
    return torch.from_numpy(vectors)


def complementary_class_prior(complementary_vectors):
    vectors = torch.as_tensor(complementary_vectors, dtype=torch.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("complementary_vectors must be a non-empty matrix")
    return vectors.mean(dim=0)


def ordinary_class_prior(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts / counts.sum()


def _dataset_targets(dataset):
    if not hasattr(dataset, 'targets'):
        raise ValueError("dataset must expose ordinary labels through targets")
    return torch.as_tensor(dataset.targets, dtype=torch.long)


class WeakDomainDataset(torch.utils.data.Dataset):
    """Return only index, input and observed complementary-label vector."""

    def __init__(
            self,
            base_dataset,
            base_indices,
            complementary_vectors,
            rotation_degrees=0.0):
        self.base_dataset = base_dataset
        self.base_indices = torch.as_tensor(
            base_indices, dtype=torch.long
        ).view(-1)
        self.complementary_vectors = torch.as_tensor(
            complementary_vectors, dtype=torch.float32
        )
        self.rotation_degrees = float(rotation_degrees)
        if len(self.base_indices) != len(self.complementary_vectors):
            raise ValueError("indices and complementary vectors must align")

    def __len__(self):
        return len(self.base_indices)

    def __getitem__(self, local_index):
        base_index = int(self.base_indices[local_index].item())
        image, _ = self.base_dataset[base_index]
        if self.rotation_degrees:
            image = transform_functional.rotate(
                image, self.rotation_degrees
            )
        return (
            int(local_index),
            image,
            self.complementary_vectors[local_index],
        )


class ShiftedEvaluationDataset(torch.utils.data.Dataset):
    """Target-domain inputs with ordinary labels used only for evaluation."""

    def __init__(
            self,
            base_dataset,
            num_classes,
            rotation_degrees=0.0,
            label_shift=0):
        self.base_dataset = base_dataset
        self.num_classes = int(num_classes)
        self.rotation_degrees = float(rotation_degrees)
        self.label_shift = int(label_shift)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        if self.rotation_degrees:
            image = transform_functional.rotate(
                image, self.rotation_degrees
            )
        label = (int(label) + self.label_shift) % self.num_classes
        return image, label


def load_base_datasets(dataname):
    """Load the same benchmark datasets and normalization as SCARCE."""
    if dataname == 'mnist':
        train_dataset = dsets.MNIST(
            root='./dataset/mnist', train=True,
            transform=transforms.ToTensor(), download=True,
        )
        test_dataset = dsets.MNIST(
            root='./dataset/mnist', train=False,
            transform=transforms.ToTensor(), download=True,
        )
    elif dataname == 'kmnist':
        train_dataset = dsets.KMNIST(
            root='./dataset/KMNIST', train=True,
            transform=transforms.ToTensor(), download=True,
        )
        test_dataset = dsets.KMNIST(
            root='./dataset/KMNIST', train=False,
            transform=transforms.ToTensor(), download=True,
        )
    elif dataname == 'fashion':
        train_dataset = dsets.FashionMNIST(
            root='./dataset/FashionMnist', train=True,
            transform=transforms.ToTensor(), download=True,
        )
        test_dataset = dsets.FashionMNIST(
            root='./dataset/FashionMnist', train=False,
            transform=transforms.ToTensor(), download=True,
        )
    elif dataname == 'cifar10':
        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.247, 0.243, 0.261),
            ),
        ])
        train_dataset = dsets.CIFAR10(
            root='./dataset', train=True,
            transform=tensor_transform, download=True,
        )
        test_dataset = dsets.CIFAR10(
            root='./dataset', train=False,
            transform=tensor_transform, download=True,
        )
    else:
        raise ValueError("unknown dataset: {}".format(dataname))
    return train_dataset, test_dataset, 10


def prepare_dual_domain_datasets(
        dataname,
        target_weak_size,
        source_size=0,
        seed=0,
        shift="none",
        target_rotation=30.0,
        label_shift=1,
        source_generation="single",
        target_generation="single",
        source_selection_probability=None,
        target_selection_probability=None):
    """Build disjoint source/target weak sets and target evaluation data."""
    if shift not in ("none", "rotation", "label_permutation", "joint"):
        raise ValueError(
            "shift must be none, rotation, label_permutation, or joint"
        )
    train_dataset, test_dataset, num_classes = load_base_datasets(dataname)
    num_train = len(train_dataset)
    if target_weak_size <= 0 or target_weak_size >= num_train:
        raise ValueError("target_weak_size must be between 1 and N - 1")

    random_state = np.random.RandomState(seed)
    permutation = random_state.permutation(num_train)
    target_indices = permutation[:target_weak_size]
    remaining_indices = permutation[target_weak_size:]
    if source_size == 0:
        source_size = len(remaining_indices)
    if source_size <= 0 or source_size > len(remaining_indices):
        raise ValueError("source_size exceeds the available source samples")
    source_indices = remaining_indices[:source_size]

    all_labels = _dataset_targets(train_dataset)
    source_labels = all_labels[source_indices]
    target_labels = all_labels[target_indices]
    effective_label_shift = 0
    if shift in ("label_permutation", "joint"):
        effective_label_shift = int(label_shift) % num_classes
        if effective_label_shift == 0:
            raise ValueError(
                "label_shift must be nonzero for label_permutation"
            )
        target_labels = (
            target_labels + effective_label_shift
        ) % num_classes

    source_vectors = generate_complementary_vectors(
        source_labels,
        num_classes,
        generation=source_generation,
        selection_probability=source_selection_probability,
        seed=seed + 101,
    )
    target_vectors = generate_complementary_vectors(
        target_labels,
        num_classes,
        generation=target_generation,
        selection_probability=target_selection_probability,
        seed=seed + 202,
    )
    effective_rotation = (
        target_rotation if shift in ("rotation", "joint") else 0.0
    )

    source_weak_dataset = WeakDomainDataset(
        train_dataset,
        source_indices,
        source_vectors,
        rotation_degrees=0.0,
    )
    target_weak_dataset = WeakDomainDataset(
        train_dataset,
        target_indices,
        target_vectors,
        rotation_degrees=effective_rotation,
    )
    target_evaluation_dataset = ShiftedEvaluationDataset(
        test_dataset,
        num_classes,
        rotation_degrees=effective_rotation,
        label_shift=effective_label_shift,
    )
    _, sample_image, _ = source_weak_dataset[0]

    return {
        "source_weak": source_weak_dataset,
        "target_weak": target_weak_dataset,
        "target_evaluation": target_evaluation_dataset,
        "num_classes": num_classes,
        "input_dim": int(sample_image.numel()),
        "source_complement_prior": complementary_class_prior(source_vectors),
        "target_complement_prior": complementary_class_prior(target_vectors),
        "source_class_prior_oracle": ordinary_class_prior(
            source_labels, num_classes
        ),
        "target_class_prior_oracle": ordinary_class_prior(
            target_labels, num_classes
        ),
    }
