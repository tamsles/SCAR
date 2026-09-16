"""Summarize four matched shift conditions for ADIW versus pooled SCAR."""

import argparse
import csv
import os

import numpy as np

try:
    from scipy import stats
except ImportError:
    # Wisteria's system SciPy is older than its NumPy. Training does not need
    # scipy.stats; inference is computed after results are downloaded locally.
    stats = None

from summarize_ablation import collect_experiment


CONDITIONS = (
    ('P(X): rotation', 'rotation'),
    ('P(Y|X): label permutation', 'label_permutation'),
    ('P(barY|Y): SCAR mechanism', 'none'),
    ('Joint: all three shifts', 'joint'),
)


def patterns(shift):
    adiw = (
        'adiw_scar/total/Res_total_mnist_{}_mlp_tr0_te1000_'
        'independent_independent_puniform_eta-1.0_t1_wl1.0_s*_r1.csv'
    ).format(shift)
    pooled = (
        'domain_scarce/total/Res_total_mnist_{}_pooled_mlp_tr0_te1000_'
        'independent_independent_puniform_s*_r1.csv'
    ).format(shift)
    return adiw, pooled


def build_shift_summary(result_dir, required_seeds=(0, 1, 2, 3, 4)):
    required_seeds = tuple(required_seeds)
    rows = []
    for condition, shift in CONDITIONS:
        adiw_pattern, pooled_pattern = patterns(shift)
        adiw_map = collect_experiment(result_dir, adiw_pattern)
        pooled_map = collect_experiment(result_dir, pooled_pattern)
        for method, values in (
                ('ADIW-SCAR', adiw_map), ('pooled SCAR', pooled_map)):
            missing = sorted(set(required_seeds) - set(values))
            if missing:
                raise ValueError(
                    '{} / {} is missing seeds {}'.format(
                        condition, method, missing
                    )
                )
        adiw = np.asarray([adiw_map[s] for s in required_seeds])
        pooled = np.asarray([pooled_map[s] for s in required_seeds])
        differences = adiw - pooled
        if stats is None:
            confidence = (float('nan'), float('nan'))
            p_value = float('nan')
        else:
            standard_error = stats.sem(differences)
            confidence = stats.t.interval(
                0.95,
                len(differences) - 1,
                loc=differences.mean(),
                scale=standard_error,
            )
            p_value = float(stats.ttest_rel(adiw, pooled).pvalue)
        rows.append({
            'condition': condition,
            'adiw_mean': float(adiw.mean()),
            'adiw_std': float(adiw.std()),
            'pooled_mean': float(pooled.mean()),
            'pooled_std': float(pooled.std()),
            'adiw_minus_pooled': float(differences.mean()),
            'paired_difference_std': float(differences.std()),
            'paired_p_value': p_value,
            'ci95_low': float(confidence[0]),
            'ci95_high': float(confidence[1]),
            'adiw_seed_accuracies': ';'.join(
                '{:.6f}'.format(value) for value in adiw
            ),
            'pooled_seed_accuracies': ';'.join(
                '{:.6f}'.format(value) for value in pooled
            ),
        })
    return rows


def write_summary(path, rows):
    fieldnames = (
        'condition', 'adiw_mean', 'adiw_std', 'pooled_mean', 'pooled_std',
        'adiw_minus_pooled', 'paired_difference_std', 'paired_p_value',
        'ci95_low', 'ci95_high', 'adiw_seed_accuracies',
        'pooled_seed_accuracies',
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in fieldnames[1:10]:
                formatted[key] = '{:.6f}'.format(row[key])
            writer.writerow(formatted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-result_dir', default='./result')
    parser.add_argument(
        '-output', default='./result/shift_ablation_summary.csv'
    )
    args = parser.parse_args()
    rows = build_shift_summary(args.result_dir)
    write_summary(args.output, rows)
    for row in rows:
        print(
            '{}: ADIW {:.3f}+/-{:.3f}, pooled {:.3f}+/-{:.3f}, '
            'delta {:+.3f}, p={:.3f}'.format(
                row['condition'], row['adiw_mean'], row['adiw_std'],
                row['pooled_mean'], row['pooled_std'],
                row['adiw_minus_pooled'], row['paired_p_value'],
            )
        )
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
