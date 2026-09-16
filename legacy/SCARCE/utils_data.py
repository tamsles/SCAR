import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from numpy.testing import assert_array_almost_equal

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
    y = y.numpy()
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        #flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def generate_compl_labels_random(
        labels, random_state=None, num_classes=None):
    # args, labels: ordinary labels
    K = (
        int(num_classes)
        if num_classes is not None
        else int(torch.max(labels).item()) + 1
    )
    random_generator = np.random.RandomState(random_state)
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = random_generator.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    print('finish generating complementary labels!')
    return complementary_labels

def class_prior(complementary_labels, num_classes=None):
    minimum_length = 0 if num_classes is None else int(num_classes)
    return np.bincount(
        complementary_labels, minlength=minimum_length
    ) / len(complementary_labels)

def prepare_cv_datasets(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='./dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./dataset/mnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='./dataset/KMNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.KMNIST(root='./dataset/KMNIST', train=False, transform=transforms.ToTensor())
    elif dataname == 'fashion':
        ordinary_train_dataset = dsets.FashionMNIST(root='./dataset/FashionMnist', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.FashionMNIST(root='./dataset/FashionMnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.ToTensor(), # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ordinary_train_dataset = dsets.CIFAR10(root='./dataset', train=True, transform=train_transform, download=True)
        test_dataset = dsets.CIFAR10(root='./dataset', train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=False, num_workers=0)
    num_classes = 10
    return full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes

def prepare_train_loaders(
        full_train_loader,
        batch_size,
        ordinary_train_dataset,
        complementary_type,
        seed,
        num_val=0,
        validation_batch_size=None):
    for data, labels in full_train_loader:
        # K is the number of classes; full_train_loader is a full batch.
        K = int(torch.max(labels).item()) + 1

    clean_validation_loader = None
    if num_val:
        if num_val < 1 or num_val >= len(labels):
            raise ValueError("num_val must be between 1 and the training size - 1")
        random_generator = np.random.RandomState(seed)
        validation_indices = random_generator.choice(
            len(labels), size=num_val, replace=False
        )
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[validation_indices] = False
        train_mask = torch.from_numpy(train_mask)
        validation_indices = torch.from_numpy(validation_indices).long()

        validation_data = data[validation_indices]
        validation_labels = labels[validation_indices].long()
        data = data[train_mask]
        labels = labels[train_mask]

        validation_dataset = torch.utils.data.TensorDataset(
            validation_data, validation_labels
        )
        clean_validation_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=validation_batch_size or batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    if complementary_type=="random":
        complementary_labels = generate_compl_labels_random(
            labels, random_state=seed, num_classes=K
        )
    elif complementary_type=="set1":
        transition_mat = np.array([[0, 0.75/3, 0.13/3, 0.12/3, 0.13/3, 0.12/3, 0.75/3, 0.12/3, 0.75/3, 0.13/3], [0.13/3, 0, 0.75/3, 0.13/3, 0.12/3, 0.13/3, 0.12/3, 0.75/3, 0.12/3, 0.75/3], [0.75/3, 0.13/3, 0, 0.75/3, 0.13/3, 0.12/3, 0.13/3, 0.12/3, 0.75/3, 0.12/3], [0.12/3, 0.75/3, 0.13/3, 0, 0.75/3, 0.13/3, 0.12/3, 0.13/3, 0.12/3, 0.75/3], [0.75/3, 0.12/3, 0.75/3, 0.13/3, 0, 0.75/3, 0.13/3, 0.12/3, 0.13/3, 0.12/3], [0.12/3, 0.75/3, 0.12/3, 0.75/3, 0.13/3, 0, 0.75/3, 0.13/3, 0.12/3, 0.13/3], [0.13/3, 0.12/3, 0.75/3, 0.12/3, 0.75/3, 0.13/3, 0, 0.75/3, 0.13/3, 0.12/3], [0.12/3, 0.13/3, 0.12/3, 0.75/3, 0.12/3, 0.75/3, 0.13/3, 0, 0.75/3, 0.13/3], [0.13/3, 0.12/3, 0.13/3, 0.12/3, 0.75/3, 0.12/3, 0.75/3, 0.13/3, 0, 0.75/3], [0.75/3, 0.13/3, 0.12/3, 0.13/3, 0.12/3, 0.75/3, 0.12/3, 0.75/3, 0.13/3, 0]])
        complementary_labels = multiclass_noisify(y=labels, P=transition_mat, random_state=seed)
    elif complementary_type=="set2":
        transition_mat = np.array([[0, 0.66/3, 0.24/3, 0.1/3, 0.24/3, 0.1/3, 0.66/3, 0.1/3, 0.66/3, 0.24/3], [0.24/3, 0, 0.66/3, 0.24/3, 0.1/3, 0.24/3, 0.1/3, 0.66/3, 0.1/3, 0.66/3], [0.66/3, 0.24/3, 0, 0.66/3, 0.24/3, 0.1/3, 0.24/3, 0.1/3, 0.66/3, 0.1/3], [0.1/3, 0.66/3, 0.24/3, 0, 0.66/3, 0.24/3, 0.1/3, 0.24/3, 0.1/3, 0.66/3], [0.66/3, 0.1/3, 0.66/3, 0.24/3, 0, 0.66/3, 0.24/3, 0.1/3, 0.24/3, 0.1/3], [0.1/3, 0.66/3, 0.1/3, 0.66/3, 0.24/3, 0, 0.66/3, 0.24/3, 0.1/3, 0.24/3], [0.24/3, 0.1/3, 0.66/3, 0.1/3, 0.66/3, 0.24/3, 0, 0.66/3, 0.24/3, 0.1/3], [0.1/3, 0.24/3, 0.1/3, 0.66/3, 0.1/3, 0.66/3, 0.24/3, 0, 0.66/3, 0.24/3], [0.24/3, 0.1/3, 0.24/3, 0.1/3, 0.66/3, 0.1/3, 0.66/3, 0.24/3, 0, 0.66/3], [0.66/3, 0.24/3, 0.1/3, 0.24/3, 0.1/3, 0.66/3, 0.1/3, 0.66/3, 0.24/3, 0]])
        complementary_labels = multiclass_noisify(y=labels, P=transition_mat, random_state=seed)
    ccp = class_prior(complementary_labels, num_classes=K)
    complementary_dataset = torch.utils.data.TensorDataset(
        data, torch.from_numpy(complementary_labels).long()
    )
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    complementary_train_loader = torch.utils.data.DataLoader(
        dataset=complementary_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    dim = int(data.reshape(-1).shape[0]/data.shape[0])
    return (
        ordinary_train_loader,
        complementary_train_loader,
        clean_validation_loader,
        ccp,
        dim,
    )
