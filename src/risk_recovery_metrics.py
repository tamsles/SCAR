"""Cross-checkpoint risk-recovery and paired-comparison statistics."""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
}


def _finite_pair(
    left: Sequence[float], right: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _correlation(
    left: Sequence[float], right: Sequence[float], rank: bool = False
) -> Optional[float]:
    x, y = _finite_pair(left, right)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return None
    if rank:
        x = np.argsort(np.argsort(x)).astype(np.float64)
        y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def risk_recovery_summary(
    true_risk: Sequence[float], estimated_risk: Sequence[float]
) -> Dict[str, Any]:
    """Evaluate an estimator across checkpoints and methods."""
    truth, estimate = _finite_pair(true_risk, estimated_risk)
    if truth.size == 0:
        return {
            "count": 0,
            "bias": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "selection_regret": None,
            "selected_index": None,
            "oracle_index": None,
        }
    error = estimate - truth
    selected = int(np.argmin(estimate))
    oracle = int(np.argmin(truth))
    return {
        "count": int(truth.size),
        "bias": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson": _correlation(truth, estimate),
        "spearman": _correlation(truth, estimate, rank=True),
        "selection_regret": float(truth[selected] - truth[oracle]),
        "selected_index": selected,
        "oracle_index": oracle,
    }


def paired_interval(
    differences: Sequence[float],
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 23000,
) -> Dict[str, Any]:
    """Return paired differences, t CI, and paired bootstrap CI."""
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("paired comparison has no finite differences")
    mean = float(values.mean())
    sample_sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size > 1:
        critical = T_CRITICAL_95.get(int(values.size - 1), 1.96)
        half_width = critical * sample_sd / math.sqrt(values.size)
        random_state = np.random.RandomState(bootstrap_seed)
        indices = random_state.randint(
            0, values.size, size=(bootstrap_samples, values.size)
        )
        means = values[indices].mean(axis=1)
        boot_low, boot_high = np.percentile(means, [2.5, 97.5])
    else:
        half_width = float("nan")
        boot_low = boot_high = mean
    return {
        "count": int(values.size),
        "mean_difference": mean,
        "sample_sd": sample_sd,
        "t_ci95_lower": mean - half_width,
        "t_ci95_upper": mean + half_width,
        "bootstrap_ci95_lower": float(boot_low),
        "bootstrap_ci95_upper": float(boot_high),
        "differences": [float(value) for value in values],
    }


def flatten_checkpoint_rows(
    runs: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        for checkpoint in run.get("checkpoints", []):
            row = {
                "experiment": run.get("experiment"),
                "method": run.get("method"),
                "seed": run.get("seed"),
            }
            row.update(checkpoint)
            rows.append(row)
    return rows

