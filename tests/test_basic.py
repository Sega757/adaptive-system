import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy_router import shannon_entropy
from robust_filter import add_intercept, huber_irls_fit, ols_fit


def test_entropy_uniform_is_max():
    # Uniform distribution over k classes has entropy log2(k).
    k = 4
    uniform = np.full(k, 1.0 / k)
    h = shannon_entropy(uniform)
    assert abs(h - np.log2(k)) < 1e-9


def test_entropy_certain_is_zero():
    dist = np.array([1.0, 0.0, 0.0])
    h = shannon_entropy(dist)
    assert h < 1e-9


def test_entropy_batch_shapes():
    dists = np.array([[0.5, 0.5], [1.0, 0.0]])
    h = shannon_entropy(dists)
    assert h.shape == (2,)
    assert h[0] > h[1]


def test_huber_recovers_clean_linear_fit():
    rng = np.random.default_rng(1)
    theta_true = np.array([1.0, 2.0])
    X = add_intercept(rng.normal(size=(200, 1)))
    y = X @ theta_true + rng.normal(scale=0.1, size=200)
    theta_ols = ols_fit(X, y)
    theta_huber, _ = huber_irls_fit(X, y)
    assert np.allclose(theta_ols, theta_true, atol=0.2)
    assert np.allclose(theta_huber, theta_true, atol=0.2)


def test_huber_beats_ols_under_contamination():
    rng = np.random.default_rng(2)
    theta_true = np.array([1.0, 2.0])
    X = add_intercept(rng.normal(size=(300, 1)))
    y = X @ theta_true + rng.normal(scale=0.1, size=300)
    y_dirty = y.copy()
    out_idx = rng.choice(300, size=45, replace=False)
    y_dirty[out_idx] += rng.normal(scale=50, size=45)

    theta_ols = ols_fit(X, y_dirty)
    theta_huber, _ = huber_irls_fit(X, y_dirty)

    err_ols = np.linalg.norm(theta_ols - theta_true)
    err_huber = np.linalg.norm(theta_huber - theta_true)
    assert err_huber < err_ols


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod, inspect.isfunction)
             if name.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
