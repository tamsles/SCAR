import argparse
import os
import random

import numpy as np
import torch

from cifar_models import convnet, densenet, resnet
from diw import estimate_importance_weights
from models import LeNet, linear_model, mlp_model
from utils_algo import (
    accuracy_check,
    chosen_loss_c,
    complementary_nll,
    supervised_loss_vector,
)
from utils_data import prepare_cv_datasets, prepare_train_loaders


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', help="optimizer's learning rate", default=1e-3, type=float)
    parser.add_argument('-bs', help='training batch size', default=256, type=int)
    parser.add_argument(
        '-ds',
        help='dataset',
        default='mnist',
        choices=['mnist', 'kmnist', 'fashion', 'cifar10'],
        type=str,
    )
    parser.add_argument(
        '-me',
        help='method type',
        choices=['SCARCE', 'DIW'],
        type=str,
        required=True,
    )
    parser.add_argument(
        '-mo',
        help='model name',
        default='mlp',
        choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet', 'convnet'],
        type=str,
    )
    parser.add_argument('-ep', help='number of epochs', type=int, default=200)
    parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
    parser.add_argument('-seed', help='random seed', default=0, type=int)
    parser.add_argument('-gpu', help='GPU index', default='0', type=str)
    parser.add_argument(
        '-op',
        help='optimizer',
        default='adam',
        choices=['adam', 'sgd'],
        type=str,
    )
    parser.add_argument(
        '-gen',
        help='generation process of complementary labels',
        default='random',
        choices=['random', 'set1', 'set2'],
        type=str,
    )
    parser.add_argument('-run_times', help='random run times', default=5, type=int)

    parser.add_argument(
        '-num_val',
        help='number of clean validation samples used by DIW',
        default=1000,
        type=int,
    )
    parser.add_argument(
        '-val_bs',
        help='DIW validation batch size; defaults to -bs',
        default=None,
        type=int,
    )
    parser.add_argument(
        '-diw_warmup',
        help='number of unweighted warm-up epochs',
        default=1,
        type=int,
    )
    parser.add_argument(
        '-diw_kernel_quantile',
        help='pairwise-loss distance quantile used as the KMM RBF gamma',
        default=0.01,
        type=float,
    )
    parser.add_argument(
        '-diw_max_weight',
        help='upper bound for each KMM sample weight',
        default=50.0,
        type=float,
    )
    parser.add_argument(
        '-kmm_solver',
        help='quadratic-program solver used by DIW',
        default='auto',
        choices=['auto', 'cvxopt', 'scipy'],
        type=str,
    )
    return parser


def validate_args(parser, args):
    if args.bs <= 0 or args.ep <= 0 or args.run_times <= 0:
        parser.error('-bs, -ep, and -run_times must be positive')
    if args.me == 'DIW':
        if args.num_val <= 0:
            parser.error('-num_val must be positive when -me DIW')
        if args.val_bs is not None and args.val_bs <= 0:
            parser.error('-val_bs must be positive')
        if args.diw_warmup < 0:
            parser.error('-diw_warmup must be non-negative')
        if not 0.0 <= args.diw_kernel_quantile <= 1.0:
            parser.error('-diw_kernel_quantile must be between 0 and 1')
        if args.diw_max_weight <= 0.0:
            parser.error('-diw_max_weight must be positive')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name, input_dim, num_classes):
    if model_name == 'mlp':
        return mlp_model(input_dim=input_dim, hidden_dim=500, output_dim=num_classes)
    if model_name == 'linear':
        return linear_model(input_dim=input_dim, output_dim=num_classes)
    if model_name == 'lenet':
        return LeNet(output_dim=num_classes)
    if model_name == 'densenet':
        return densenet(num_classes=num_classes)
    if model_name == 'resnet':
        return resnet(depth=32, num_classes=num_classes)
    if model_name == 'convnet':
        return convnet.Cnn(
            input_channels=3, n_outputs=num_classes, dropout_rate=0.25
        )
    raise ValueError('Unknown model: {}'.format(model_name))


def build_optimizer(args, model):
    if args.op == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=0.9,
        )
    return torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )


def next_validation_batch(validation_loader, validation_iterator):
    try:
        batch = next(validation_iterator)
    except StopIteration:
        validation_iterator = iter(validation_loader)
        batch = next(validation_iterator)
    return batch, validation_iterator


def estimate_batch_weights(
        model,
        train_images,
        complementary_labels,
        validation_batch,
        args,
        device):
    validation_images, validation_labels = validation_batch[:2]
    validation_images = validation_images.to(device)
    validation_labels = validation_labels.to(device)

    model.eval()
    with torch.no_grad():
        train_outputs = model(train_images)
        train_losses = complementary_nll(
            train_outputs, complementary_labels
        )
        validation_outputs = model(validation_images)
        validation_losses = supervised_loss_vector(
            validation_outputs, validation_labels
        )

    coefficients = estimate_importance_weights(
        train_losses.detach().cpu().numpy().reshape(-1, 1),
        validation_losses.detach().cpu().numpy().reshape(-1, 1),
        kernel_quantile=args.diw_kernel_quantile,
        max_weight=args.diw_max_weight,
        solver=args.kmm_solver,
    )
    return torch.from_numpy(coefficients).to(
        device=device, dtype=train_outputs.dtype
    )


def result_paths(args):
    total_dir = './result/total'
    detail_dir = './result/detail'
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    diw_suffix = ''
    if args.me == 'DIW':
        effective_val_bs = args.val_bs or args.bs
        diw_suffix = '_nv{}_vb{}_wu{}_q{}_mw{}_ks{}'.format(
            args.num_val,
            effective_val_bs,
            args.diw_warmup,
            args.diw_kernel_quantile,
            args.diw_max_weight,
            args.kmm_solver,
        )
    name = '{}_{}_{}_{}_{}_lr{}_wd{}_b{}_e{}_s{}_r{}{}'.format(
        args.ds,
        args.gen,
        args.me,
        args.mo,
        args.op,
        args.lr,
        args.wd,
        args.bs,
        args.ep,
        args.seed,
        args.run_times,
        diw_suffix,
    )
    return (
        os.path.join(total_dir, 'Res_total_{}.csv'.format(name)),
        os.path.join(detail_dir, 'Res_detail_{}.csv'.format(name)),
    )


def initialize_result_files(total_path, detail_path):
    with open(total_path, 'w') as file_handle:
        file_handle.write('run_idx,acc,std\n')
    with open(detail_path, 'w') as file_handle:
        file_handle.write('run_idx,epoch,train_loss,train_accuracy,test_accuracy\n')


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    device = torch.device(
        'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    )

    total_path, detail_path = result_paths(args)
    initialize_result_files(total_path, detail_path)
    run_accuracies = np.zeros(args.run_times, dtype=np.float64)

    for run_idx in range(args.run_times):
        run_seed = args.seed + run_idx
        set_random_seed(run_seed)
        print('the {}-th random round'.format(run_idx))

        (
            full_train_loader,
            train_loader,
            test_loader,
            ordinary_train_dataset,
            _,
            num_classes,
        ) = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
        (
            _,
            complementary_train_loader,
            clean_validation_loader,
            complementary_class_prior,
            input_dim,
        ) = prepare_train_loaders(
            full_train_loader=full_train_loader,
            batch_size=args.bs,
            ordinary_train_dataset=ordinary_train_dataset,
            complementary_type=args.gen,
            seed=run_seed,
            num_val=args.num_val if args.me == 'DIW' else 0,
            validation_batch_size=args.val_bs or args.bs,
        )

        model = build_model(args.mo, input_dim, num_classes).to(device)
        optimizer = build_optimizer(args, model)

        train_accuracy = accuracy_check(train_loader, model, device)
        test_accuracy = accuracy_check(test_loader, model, device)
        print(
            'Epoch: 0. Tr Acc: {}. Te Acc: {}'.format(
                train_accuracy, test_accuracy
            )
        )

        test_accuracies = []
        train_accuracies = []
        for epoch in range(args.ep):
            validation_iterator = (
                iter(clean_validation_loader)
                if clean_validation_loader is not None
                else None
            )
            epoch_loss_total = 0.0
            epoch_sample_count = 0
            epoch_weight_means = []

            for images, labels in complementary_train_loader:
                images = images.to(device)
                labels = labels.to(device)
                sample_weight = None

                if args.me == 'DIW':
                    if epoch < args.diw_warmup:
                        sample_weight = images.new_ones(images.shape[0])
                    else:
                        validation_batch, validation_iterator = next_validation_batch(
                            clean_validation_loader, validation_iterator
                        )
                        sample_weight = estimate_batch_weights(
                            model,
                            images,
                            labels,
                            validation_batch,
                            args,
                            device,
                        )
                    epoch_weight_means.append(sample_weight.mean().item())

                model.train()
                optimizer.zero_grad()
                outputs = model(images)
                loss, _ = chosen_loss_c(
                    f=outputs,
                    K=num_classes,
                    labels=labels,
                    ccp=complementary_class_prior,
                    meta_method=args.me,
                    device=device,
                    sample_weight=sample_weight,
                )
                loss.backward()
                optimizer.step()

                epoch_loss_total += loss.item() * images.shape[0]
                epoch_sample_count += images.shape[0]

            epoch_loss = epoch_loss_total / epoch_sample_count
            train_accuracy = accuracy_check(train_loader, model, device)
            test_accuracy = accuracy_check(test_loader, model, device)
            with open(detail_path, 'a') as file_handle:
                file_handle.write(
                    '{},{},{:.6f},{:.6f},{:.6f}\n'.format(
                        run_idx + 1,
                        epoch + 1,
                        epoch_loss,
                        train_accuracy,
                        test_accuracy,
                    )
                )

            if epoch >= args.ep - 10:
                test_accuracies.append(test_accuracy)
                train_accuracies.append(train_accuracy)

            weight_message = ''
            if epoch_weight_means:
                weight_message = ' Mean IW: {:.4f}.'.format(
                    np.mean(epoch_weight_means)
                )
            print(
                'Epoch: {}. Tr Acc: {}. Te Acc: {}.{}'.format(
                    epoch + 1,
                    train_accuracy,
                    test_accuracy,
                    weight_message,
                )
            )

        average_test_accuracy = np.mean(test_accuracies)
        average_train_accuracy = np.mean(train_accuracies)
        run_accuracies[run_idx] = average_test_accuracy
        with open(total_path, 'a') as file_handle:
            file_handle.write(
                '{},{:.6f},None\n'.format(
                    run_idx + 1, average_test_accuracy
                )
            )
        print('Average Test Accuracy over Last 10 Epochs:', average_test_accuracy)
        print('Average Training Accuracy over Last 10 Epochs:', average_train_accuracy)

    print(
        'Avg_acc:{}    std_acc:{}'.format(
            run_accuracies.mean(), run_accuracies.std()
        )
    )
    with open(total_path, 'a') as file_handle:
        file_handle.write(
            'in total,{:.6f},{:.6f}\n'.format(
                run_accuracies.mean(), run_accuracies.std()
            )
        )
    print(
        'NOW is dataset: {} with method {} with model {} weight_decay {} '
        'learning rate {} batch_size {} op {}'.format(
            args.ds,
            args.me,
            args.mo,
            args.wd,
            args.lr,
            args.bs,
            args.op,
        )
    )


if __name__ == '__main__':
    main()
