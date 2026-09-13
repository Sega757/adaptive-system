"""
Huber loss M-estimation via Iteratively Reweighted Least Squares (IRLS).

This is real, standard robust statistics (Huber, 1964) — not the
invented "chronos_logos.py" machinery from the source document. It's
useful whenever you're fitting a model to telemetry/sensor data that
may contain outliers or spikes you don't want dominating the fit.

Huber loss:
    L_delta(a) = 0.5 * a^2                  for |a| <= delta
    L_delta(a) = delta * (|a| - 0.5*delta)  for |a| > delta

Minimizing this via IRLS: at each iteration, points with large
residuals get down-weighted (weight = delta / |residual|) instead of
being squared, which caps their influence on the fit.
"""

from __future__ import annotations

import numpy as np


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain ordinary least squares, for comparison. X should already
    include an intercept column if desired."""
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def huber_irls_fit(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 1.35,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """Fit y ~ X @ theta by minimizing Huber loss via IRLS.

    Returns (theta, n_iterations_used).
    """
    n, p = X.shape
    theta = ols_fit(X, y)  # warm start from OLS

    for it in range(1, max_iter + 1):
        residuals = y - X @ theta
        abs_r = np.abs(residuals)
        # Huber weights: 1 for |r|<=delta, delta/|r| for |r|>delta.
        # Avoid divide-by-zero for near-perfect residuals.
        weights = np.ones_like(abs_r)
        mask = abs_r > delta
        weights[mask] = delta / np.maximum(abs_r[mask], 1e-12)

        W = weights
        # Weighted least squares: theta = (X^T W X)^-1 X^T W y
        XtW = X.T * W
        A = XtW @ X
        b = XtW @ y
        theta_new = np.linalg.solve(A, b)

        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    return theta, it


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])
