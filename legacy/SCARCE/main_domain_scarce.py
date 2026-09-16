"""Train SCARCE on exactly one of the train/test weak-label domains."""

import argparse
import os

import numpy as np
import torch

from adiw_scar import scar_branch_risk
from domain_data import prepare_dual_domain_datasets
from main import build_model, build_optimizer, set_random_seed
from utils_algo import accuracy_check


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-domain', required=True, choices=['train', 'test', 'pooled'],
        help='train, test, or their unweighted pooled weak-label data',
    )
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-bs', default=256, type=int)
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
    )
    parser.add_argument('-target_weak_size', default=1000, type=int)
    parser.add_argument(
        '-source_size', default=0, type=int,
        help='zero uses every training sample outside the test-domain split',
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
    parser.add_argument('-source_scar_rate', default=None)
    parser.add_argument('-target_scar_rate', default=None)
    parser.add_argument(
        '-class_prior', default='uniform', choices=['uniform', 'oracle'],
        help='oracle is for synthetic diagnostics only',
    )
    parser.add_argument(
        '-risk_correction', default='abs',
        choices=['abs', 'relu', 'none'],
    )
    return parser


def validate_args(parser, args):
    positive_values = (
        args.lr, args.bs, args.ep, args.run_times, args.target_weak_size,
    )
    if any(value <= 0 for value in positive_values):
        parser.error('learning, batch, epoch, run, and target values must be positive')
    if args.source_size < 0 or args.workers < 0:
        parser.error('-source_size and -workers must be non-negative')


def select_training_domain(data, domain):
    """Return one domain or their unweighted pooled SCAR baseline."""
    if domain == 'train':
        return (
            data['source_weak'],
            data['source_complement_prior'],
            data['source_class_prior_oracle'],
        )
    if domain == 'test':
        return (
            data['target_weak'],
            data['target_complement_prior'],
            data['target_class_prior_oracle'],
        )
    if domain == 'pooled':
        source_size = len(data['source_weak'])
        target_size = len(data['target_weak'])
        total_size = float(source_size + target_size)
        source_fraction = source_size / total_size
        target_fraction = target_size / total_size
        return (
            torch.utils.data.ConcatDataset([
                data['source_weak'], data['target_weak'],
            ]),
            source_fraction * data['source_complement_prior']
            + target_fraction * data['target_complement_prior'],
            source_fraction * data['source_class_prior_oracle']
            + target_fraction * data['target_class_prior_oracle'],
        )
    raise ValueError('domain must be train, test, or pooled')


def choose_class_prior(num_classes, oracle_prior, mode, device):
    if mode == 'oracle':
        return oracle_prior.to(device)
    return torch.full((num_classes,), 1.0 / num_classes, device=device)


def result_paths(args):
    total_dir = './result/domain_scarce/total'
    detail_dir = './result/domain_scarce/detail'
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)
    name = '{}_{}_{}_{}_tr{}_te{}_{}_{}_p{}_s{}_r{}'.format(
        args.ds,
        args.shift,
        args.domain,
        args.mo,
        args.source_size,
        args.target_weak_size,
        args.source_gen,
        args.target_gen,
        args.class_prior,
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
        file_handle.write('run_idx,epoch,loss,target_accuracy\n')


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
    training_dataset, complement_prior, oracle_prior = select_training_domain(
        data, args.domain
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    evaluation_loader = torch.utils.data.DataLoader(
        data['target_evaluation'],
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    model = build_model(
        args.mo, data['input_dim'], data['num_classes']
    ).to(device)
    optimizer = build_optimizer(args, model)
    complement_prior = complement_prior.to(device)
    class_prior = choose_class_prior(
        data['num_classes'], oracle_prior, args.class_prior, device
    )

    print(
        'Run {}: SCARCE uses {} data ({} weak samples)'.format(
            run_index + 1, args.domain, len(training_dataset)
        )
    )
    print('{} bar_pi:'.format(args.domain), complement_prior.cpu().numpy())
    print('{} pi:'.format(args.domain), class_prior.cpu().numpy())

    last_accuracies = []
    for epoch in range(args.ep):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for _, images, complementary_vectors in training_loader:
            images = images.to(device)
            complementary_vectors = complementary_vectors.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = scar_branch_risk(
                outputs,
                complementary_vectors,
                class_prior,
                complement_prior,
                branch_weights=None,
                correction=args.risk_correction,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]

        epoch_loss = total_loss / total_samples
        target_accuracy = accuracy_check(evaluation_loader, model, device)
        if epoch >= args.ep - 10:
            last_accuracies.append(target_accuracy)
        with open(detail_path, 'a') as file_handle:
            file_handle.write(
                '{},{},{:.6f},{:.6f}\n'.format(
                    run_index + 1, epoch + 1, epoch_loss, target_accuracy
                )
            )
        print(
            'Epoch {}: loss {:.4f}, target acc {:.2f}'.format(
                epoch + 1, epoch_loss, target_accuracy
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
        'SCARCE {}-domain target accuracy: {:.6f} +/- {:.6f}'.format(
            args.domain, run_accuracies.mean(), run_accuracies.std()
        )
    )


if __name__ == '__main__':
    main()
