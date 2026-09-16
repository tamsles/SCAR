"""Dynamic Importance Weighting utilities.

This module adapts the KMM implementation from TongtongFANG/DIW to the
SCARCE training loop. The quadratic program is unchanged, while a SciPy
backend is provided so the existing SCARCE dependency set remains usable.
"""

import math

import numpy as np
from scipy.optimize import minimize

try:
    from cvxopt import matrix, solvers
except ImportError:  # cvxopt is optional; SciPy is the fallback backend.
    matrix = None
    solvers = None


def _as_2d_float_array(data, name):
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("{} must be a non-empty 1D or 2D array".format(name))
    if not np.isfinite(array).all():
        raise ValueError("{} contains NaN or infinite values".format(name))
    return array


def get_kernel_width(data, quantile=0.01):
    """Return the DIW distance quantile used as the RBF gamma value."""
    data = _as_2d_float_array(data, "data")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if len(data) < 2:
        return 1.0

    differences = data[:, None, :] - data[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    distances = distances[np.triu_indices(len(data), k=1)]
    width = float(np.quantile(distances, quantile))

    if not np.isfinite(width) or width <= np.finfo(np.float64).eps:
        positive_distances = distances[distances > np.finfo(np.float64).eps]
        if len(positive_distances) == 0:
            return 1.0
        width = float(np.median(positive_distances))
    return width


def _rbf_kernel(x, y, gamma):
    differences = x[:, None, :] - y[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    return np.exp(-gamma * squared_distances)


def _solve_with_cvxopt(kernel, kappa, lower_sum, upper_sum, max_weight):
    n_train = len(kappa)
    constraint_matrix = np.vstack([
        np.ones((1, n_train)),
        -np.ones((1, n_train)),
        -np.eye(n_train),
        np.eye(n_train),
    ])
    constraint_bounds = np.vstack([
        [[upper_sum], [-lower_sum]],
        np.zeros((n_train, 1)),
        np.full((n_train, 1), max_weight),
    ])

    solvers.options["show_progress"] = False
    solution = solvers.qp(
        matrix(kernel, tc="d"),
        matrix(kappa, tc="d"),
        matrix(constraint_matrix, tc="d"),
        matrix(constraint_bounds, tc="d"),
    )
    if solution["status"] != "optimal":
        raise RuntimeError("CVXOPT KMM solver returned {}".format(solution["status"]))
    return np.asarray(solution["x"], dtype=np.float64).reshape(-1)


def _solve_with_scipy(kernel, kappa, lower_sum, upper_sum, max_weight):
    n_train = len(kappa)
    ones = np.ones(n_train, dtype=np.float64)

    def objective(beta):
        return 0.5 * np.dot(beta, np.dot(kernel, beta)) + np.dot(kappa, beta)

    def gradient(beta):
        return np.dot(kernel, beta) + kappa

    constraints = (
        {
            "type": "ineq",
            "fun": lambda beta: np.sum(beta) - lower_sum,
            "jac": lambda beta: ones,
        },
        {
            "type": "ineq",
            "fun": lambda beta: upper_sum - np.sum(beta),
            "jac": lambda beta: -ones,
        },
    )
    result = minimize(
        objective,
        np.ones(n_train, dtype=np.float64),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_train,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    if not result.success:
        raise RuntimeError("SciPy KMM solver failed: {}".format(result.message))
    return np.asarray(result.x, dtype=np.float64)


def kmm(x_train, x_reference, kernel_width, max_weight=50.0, solver="auto"):
    """Estimate train-sample weights by Kernel Mean Matching."""
    x_train = _as_2d_float_array(x_train, "x_train")
    x_reference = _as_2d_float_array(x_reference, "x_reference")
    if x_train.shape[1] != x_reference.shape[1]:
        raise ValueError("x_train and x_reference must have the same feature size")
    if kernel_width <= 0.0 or not np.isfinite(kernel_width):
        raise ValueError("kernel_width must be a positive finite value")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    if solver not in ("auto", "cvxopt", "scipy"):
        raise ValueError("solver must be one of: auto, cvxopt, scipy")

    n_train = len(x_train)
    n_reference = len(x_reference)
    kernel = _rbf_kernel(x_train, x_train, kernel_width)
    kernel += 1e-5 * np.eye(n_train)
    cross_kernel = _rbf_kernel(x_train, x_reference, kernel_width)
    kappa = -(float(n_train) / float(n_reference)) * np.sum(cross_kernel, axis=1)

    epsilon = (math.sqrt(n_train) - 1.0) / math.sqrt(n_train)
    lower_sum = n_train * (1.0 - epsilon)
    upper_sum = n_train * (1.0 + epsilon)

    selected_solver = solver
    if selected_solver == "auto":
        selected_solver = "cvxopt" if solvers is not None else "scipy"
    if selected_solver == "cvxopt":
        if solvers is None:
            raise ImportError("cvxopt is not installed; use solver='scipy' or install cvxopt")
        weights = _solve_with_cvxopt(
            kernel, kappa, lower_sum, upper_sum, max_weight
        )
    else:
        weights = _solve_with_scipy(
            kernel, kappa, lower_sum, upper_sum, max_weight
        )

    if not np.isfinite(weights).all():
        raise RuntimeError("KMM returned NaN or infinite weights")
    return np.clip(weights, 0.0, max_weight)


def estimate_importance_weights(
        train_losses,
        validation_losses,
        kernel_quantile=0.01,
        max_weight=50.0,
        solver="auto"):
    """Estimate DIW coefficients from train and clean-validation losses."""
    train_losses = _as_2d_float_array(train_losses, "train_losses")
    validation_losses = _as_2d_float_array(
        validation_losses, "validation_losses"
    )
    kernel_width = get_kernel_width(train_losses, quantile=kernel_quantile)
    return kmm(
        train_losses,
        validation_losses,
        kernel_width=kernel_width,
        max_weight=max_weight,
        solver=solver,
    )
