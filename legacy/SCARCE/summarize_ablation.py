"""Summarize matched-seed ADIW-SCAR and single-domain SCAR ablations."""

import argparse
import csv
import glob
import os
import re

import numpy as np


EXPERIMENTS = (
    (
        'ADIW-SCAR (train + test)',
        'adiw_scar/total/Res_total_mnist_rotation_mlp_tr0_te1000_'
        'single_single_puniform_eta-1.0_t1_wl1.0_s*_r1.csv',
    ),
    (
        'SCAR (pooled train + test, no ADIW)',
        'domain_scarce/total/Res_total_mnist_rotation_pooled_mlp_'
        'tr0_te1000_single_single_puniform_s*_r1.csv',
    ),
    (
        'SCAR (train only)',
        'domain_scarce/total/Res_total_mnist_rotation_train_mlp_'
        'tr0_te1000_single_single_puniform_s*_r1.csv',
    ),
    (
        'SCAR (test only)',
        'domain_scarce/total/Res_total_mnist_rotation_test_mlp_'
        'tr0_te1000_single_single_puniform_s*_r1.csv',
    ),
)


def read_total_accuracy(path):
    with open(path, newline='') as file_handle:
        for row in csv.reader(file_handle):
            if row and row[0] == 'in total':
                return float(row[1])
    raise ValueError('missing in-total row: {}'.format(path))


def seed_from_path(path):
    match = re.search(r'_s(\d+)_r1\.csv$', path)
    if match is None:
        raise ValueError('cannot parse seed from path: {}'.format(path))
    return int(match.group(1))


def collect_experiment(result_dir, relative_pattern):
    paths = glob.glob(os.path.join(result_dir, relative_pattern))
    values = {}
    for path in paths:
        seed = seed_from_path(path)
        if seed in values:
            raise ValueError('duplicate seed {} for {}'.format(seed, path))
        values[seed] = read_total_accuracy(path)
    return values


def build_summary(result_dir, required_seeds=(0, 1, 2, 3, 4)):
    required_seeds = tuple(required_seeds)
    collected = []
    for method, pattern in EXPERIMENTS:
        seed_values = collect_experiment(result_dir, pattern)
        missing = sorted(set(required_seeds) - set(seed_values))
        if missing:
            raise ValueError(
                '{} is missing seeds {}'.format(method, missing)
            )
        ordered = np.asarray(
            [seed_values[seed] for seed in required_seeds],
            dtype=np.float64,
        )
        collected.append((method, ordered))

    adiw_values = collected[0][1]
    adiw_mean = adiw_values.mean()
    rows = []
    for method, values in collected:
        paired_deltas = values - adiw_values
        rows.append({
            'method': method,
            'num_seeds': len(values),
            'mean_accuracy': float(values.mean()),
            'std_accuracy': float(values.std()),
            'delta_vs_adiw': float(values.mean() - adiw_mean),
            'paired_delta_std': float(paired_deltas.std()),
            'seed_accuracies': ';'.join(
                '{:.6f}'.format(value) for value in values
            ),
            'paired_seed_deltas': ';'.join(
                '{:.6f}'.format(value) for value in paired_deltas
            ),
        })
    return rows


def write_summary(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fieldnames = (
        'method', 'num_seeds', 'mean_accuracy', 'std_accuracy',
        'delta_vs_adiw', 'paired_delta_std', 'seed_accuracies',
        'paired_seed_deltas',
    )
    with open(path, 'w', newline='') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in (
                    'mean_accuracy', 'std_accuracy', 'delta_vs_adiw',
                    'paired_delta_std'):
                formatted[key] = '{:.6f}'.format(row[key])
            writer.writerow(formatted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-result_dir', default='./result')
    parser.add_argument(
        '-output', default='./result/ablation_summary.csv'
    )
    args = parser.parse_args()
    rows = build_summary(args.result_dir)
    write_summary(args.output, rows)
    print('method\tmean +/- std\tpaired delta vs ADIW-SCAR')
    for row in rows:
        print(
            '{}\t{:.3f} +/- {:.3f}\t{:+.3f} +/- {:.3f}'.format(
                row['method'],
                row['mean_accuracy'],
                row['std_accuracy'],
                row['delta_vs_adiw'],
                row['paired_delta_std'],
            )
        )
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
