"""Train the dual-domain ADIW-SCAR method described in the project slides."""

import argparse
import os

import numpy as np
import torch

from adiw_scar import (
    BranchWeightMemory,
    combined_adiw_scar_loss,
    loss_value_representation,
)
from domain_data import prepare_dual_domain_datasets
from main import build_model, build_optimizer, set_random_seed
from utils_algo import accuracy_check


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-bs', default=256, type=int)
    parser.add_argument('-target_bs', default=None, type=int)
    parser.add_argument(
        '-ds', default='mnist',
        choices=['mnist', 'kmnist', 'fashion', 'cifar10'],
    )
    parser.add_argument(
        '-mo', default='mlp',
        choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet', 'convnet'],
    )
    parser.add_argument('-op', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('-ep', default=200, type=int)
    parser.add_argument('-wd', default=1e-5, type=float)
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-gpu', default='0', type=str)
    parser.add_argument('-run_times', default=5, type=int)
    parser.add_argument('-workers', default=0, type=int)

    parser.add_argument(
        '-shift', default='rotation',
        choices=['none', 'rotation', 'label_permutation', 'joint'],
        help=(
            'none, rotation, label permutation, or their joint input/label '
            'shift'
        ),
    )
    parser.add_argument('-target_weak_size', default=1000, type=int)
    parser.add_argument(
        '-source_size', default=0, type=int,
        help='zero uses every training sample outside the target weak split',
    )
    parser.add_argument('-target_rotation', default=30.0, type=float)
    parser.add_argument('-label_shift', default=1, type=int)
    parser.add_argument(
        '-source_gen', default='single',
        choices=['single', 'independent'],
    )
    parser.add_argument(
        '-target_gen', default='single',
        choices=['single', 'independent'],
    )
    parser.add_argument(
        '-source_scar_rate', default=None,
        help='scalar or comma-separated c_tr,k values for independent SCAR',
    )
    parser.add_argument(
        '-target_scar_rate', default=None,
        help='scalar or comma-separated c_te,k values for independent SCAR',
    )
    parser.add_argument(
        '-target_prior', default='uniform',
        choices=['uniform', 'oracle'],
        help='oracle is for synthetic diagnostics only',
    )
    parser.add_argument(
        '-eta', default=-1.0, type=float,
        help='target-risk fraction; negative selects n_te/(n_tr+n_te)',
    )
    parser.add_argument(
        '-risk_correction', default='abs',
        choices=['abs', 'relu', 'none'],
    )

    parser.add_argument('-adiw_steps', default=1, type=int)
    parser.add_argument('-adiw_lr', default=1.0, type=float)
    parser.add_argument('-adiw_max_weight', default=50.0, type=float)
    parser.add_argument('-adiw_kernel_quantile', default=0.5, type=float)
    parser.add_argument(
        '-adiw_sum_tolerance', default=None, type=float,
        help=(
            'KMM mean-weight tolerance; defaults to the '
            'KMM n-dependent value'
        ),
    )
    parser.add_argument('-adiw_warmup', default=0, type=int)
    return parser


def validate_args(parser, args):
    positive_values = (
        args.lr, args.bs, args.ep, args.run_times,
        args.target_weak_size, args.adiw_lr, args.adiw_max_weight,
    )
    if any(value <= 0 for value in positive_values):
        parser.error(
            'learning, batch, epoch, run, target, and ADIW values '
            'must be positive'
        )
    if args.target_bs is not None and args.target_bs <= 0:
        parser.error('-target_bs must be positive')
    if args.source_size < 0 or args.workers < 0:
        parser.error('-source_size and -workers must be non-negative')
    if args.adiw_steps < 0 or args.adiw_warmup < 0:
        parser.error('-adiw_steps and -adiw_warmup must be non-negative')
    if not 0.0 <= args.adiw_kernel_quantile <= 1.0:
        parser.error('-adiw_kernel_quantile must be between zero and one')
    if args.adiw_sum_tolerance is not None:
        if not 0.0 <= args.adiw_sum_tolerance <= 1.0:
            parser.error('-adiw_sum_tolerance must be between zero and one')
    if args.eta >= 0.0 and not 0.0 <= args.eta <= 1.0:
        parser.error('-eta must be negative for auto or between zero and one')


def next_target_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def target_fraction(args, num_source, num_target):
    if args.eta >= 0.0:
        return args.eta
    return float(num_target) / float(num_source + num_target)


def result_paths(args):
    total_dir = './result/adiw_scar/total'
    detail_dir = './result/adiw_scar/detail'
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)
    name = (
        '{}_{}_{}_tr{}_te{}_{}_{}_p{}_eta{}_t{}_wl{}_s{}_r{}'
    ).format(
        args.ds,
        args.shift,
        args.mo,
        args.source_size,
        args.target_weak_size,
        args.source_gen,
        args.target_gen,
        args.target_prior,
        args.eta,
        args.adiw_steps,
        args.adiw_lr,
        args.seed,
        args.run_times,
    )
    return (
        os.path.join(total_dir, 'Res_total_{}.csv'.format(name)),
        os.path.join(detail_dir, 'Res_detail_{}.csv'.format(name)),
    )


def initialize_results(total_path, detail_path):
    with open(total_path, 'w') as file_handle:
        file_handle.write('run_idx,acc,std\n')
    with open(detail_path, 'w') as file_handle:
        file_handle.write(
            'run_idx,epoch,loss,source_risk,target_risk,'
            'target_accuracy,weight_mean,weight_std\n'
        )


def choose_target_prior(data, mode, device):
    num_classes = data['num_classes']
    if mode == 'oracle':
        return data['target_class_prior_oracle'].to(device)
    return torch.full(
        (num_classes,), 1.0 / num_classes, device=device
    )


def train_one_run(args, run_seed, device, detail_path, run_index):
    set_random_seed(run_seed)
    data = prepare_dual_domain_datasets(
        dataname=args.ds,
        target_weak_size=args.target_weak_size,
        source_size=args.source_size,
        seed=run_seed,
        shift=args.shift,
        target_rotation=args.target_rotation,
        label_shift=args.label_shift,
        source_generation=args.source_gen,
        target_generation=args.target_gen,
        source_selection_probability=args.source_scar_rate,
        target_selection_probability=args.target_scar_rate,
    )
    source_dataset = data['source_weak']
    target_dataset = data['target_weak']
    source_loader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=args.target_bs or args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    evaluation_loader = torch.utils.data.DataLoader(
        data['target_evaluation'],
        batch_size=args.target_bs or args.bs,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    num_classes = data['num_classes']
    model = build_model(
        args.mo, data['input_dim'], num_classes
    ).to(device)
    optimizer = build_optimizer(args, model)
    weight_memory = BranchWeightMemory(
        len(source_dataset), num_classes, device
    )
    source_complement_prior = data['source_complement_prior'].to(device)
    target_complement_prior = data['target_complement_prior'].to(device)
    target_class_prior = choose_target_prior(data, args.target_prior, device)
    eta = target_fraction(args, len(source_dataset), len(target_dataset))
    target_iterator = iter(target_loader)

    print(
        'Run {}: source={}, target weak={}, eta={:.6f}, prior={}'.format(
            run_index + 1,
            len(source_dataset),
            len(target_dataset),
            eta,
            args.target_prior,
        )
    )
    print('source bar_pi:', source_complement_prior.detach().cpu().numpy())
    print('target bar_pi:', target_complement_prior.detach().cpu().numpy())
    print('target pi:', target_class_prior.detach().cpu().numpy())

    last_accuracies = []
    for epoch in range(args.ep):
        model.train()
        total_loss = 0.0
        total_source_risk = 0.0
        total_target_risk = 0.0
        total_samples = 0
        weight_sum = 0.0
        weight_square_sum = 0.0
        weight_count = 0

        for source_batch in source_loader:
            target_batch, target_iterator = next_target_batch(
                target_loader, target_iterator
            )
            source_indices, source_images, source_complements = source_batch
            _, target_images, target_complements = target_batch
            source_indices = source_indices.to(device)
            source_images = source_images.to(device)
            source_complements = source_complements.to(device)
            target_images = target_images.to(device)
            target_complements = target_complements.to(device)

            optimizer.zero_grad()
            source_outputs = model(source_images)
            target_outputs = model(target_images)
            if epoch < args.adiw_warmup:
                source_weights = source_outputs.new_ones(
                    source_outputs.shape
                )
            else:
                source_weights = weight_memory.estimate(
                    source_indices,
                    loss_value_representation(source_outputs.detach()),
                    source_complements,
                    loss_value_representation(target_outputs.detach()),
                    target_complements,
                    source_complement_prior,
                    target_complement_prior,
                    step_size=args.adiw_lr,
                    num_steps=args.adiw_steps,
                    max_weight=args.adiw_max_weight,
                    kernel_quantile=args.adiw_kernel_quantile,
                    sum_tolerance=args.adiw_sum_tolerance,
                )

            loss, diagnostics = combined_adiw_scar_loss(
                source_outputs,
                source_complements,
                source_weights,
                target_outputs,
                target_complements,
                target_class_prior,
                target_complement_prior,
                eta,
                correction=args.risk_correction,
            )
            loss.backward()
            optimizer.step()

            batch_size = source_images.shape[0]
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_source_risk += (
                diagnostics['source_risk'].item() * batch_size
            )
            total_target_risk += (
                diagnostics['target_risk'].item() * batch_size
            )
            detached_weights = source_weights.detach()
            weight_sum += detached_weights.sum().item()
            weight_square_sum += (
                detached_weights * detached_weights
            ).sum().item()
            weight_count += detached_weights.numel()

        epoch_loss = total_loss / total_samples
        source_risk = total_source_risk / total_samples
        target_risk = total_target_risk / total_samples
        weight_mean = weight_sum / weight_count
        weight_variance = max(
            weight_square_sum / weight_count - weight_mean * weight_mean,
            0.0,
        )
        weight_std = np.sqrt(weight_variance)
        target_accuracy = accuracy_check(
            evaluation_loader, model, device
        )
        if epoch >= args.ep - 10:
            last_accuracies.append(target_accuracy)
        with open(detail_path, 'a') as file_handle:
            file_handle.write(
                '{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                    run_index + 1,
                    epoch + 1,
                    epoch_loss,
                    source_risk,
                    target_risk,
                    target_accuracy,
                    weight_mean,
                    weight_std,
                )
            )
        print(
            'Epoch {}: loss {:.4f}, source {:.4f}, target {:.4f}, '
            'target acc {:.2f}, weight {:.4f}+/-{:.4f}'.format(
                epoch + 1,
                epoch_loss,
                source_risk,
                target_risk,
                target_accuracy,
                weight_mean,
                weight_std,
            )
        )

    return float(np.mean(last_accuracies))


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    device = torch.device(
        'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    )
    total_path, detail_path = result_paths(args)
    initialize_results(total_path, detail_path)

    run_accuracies = np.zeros(args.run_times, dtype=np.float64)
    for run_index in range(args.run_times):
        run_accuracy = train_one_run(
            args,
            args.seed + run_index,
            device,
            detail_path,
            run_index,
        )
        run_accuracies[run_index] = run_accuracy
        with open(total_path, 'a') as file_handle:
            file_handle.write(
                '{},{:.6f},None\n'.format(run_index + 1, run_accuracy)
            )

    with open(total_path, 'a') as file_handle:
        file_handle.write(
            'in total,{:.6f},{:.6f}\n'.format(
                run_accuracies.mean(), run_accuracies.std()
            )
        )
    print(
        'ADIW-SCAR target accuracy: {:.6f} +/- {:.6f}'.format(
            run_accuracies.mean(), run_accuracies.std()
        )
    )


if __name__ == '__main__':
    main()
