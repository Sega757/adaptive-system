"""
Demo: OLS vs Huber IRLS under outlier contamination.

We generate a known linear relationship y = X @ theta_true + noise,
then contaminate a fraction of the observations with large outliers
(simulating corrupted/adversarial telemetry), and measure how far each
fitted theta ends up from theta_true.
"""

import numpy as np

from robust_filter import add_intercept, huber_irls_fit, ols_fit


def run(n=500, contamination=0.15, outlier_scale=40.0, seed=0):
    rng = np.random.default_rng(seed)

    theta_true = np.array([2.0, -3.5, 1.2])  # intercept, coef1, coef2
    X_raw = rng.normal(size=(n, 2))
    X = add_intercept(X_raw)
    noise = rng.normal(scale=1.0, size=n)
    y = X @ theta_true + noise

    # Contaminate a fraction of the targets with large outliers.
    n_out = int(n * contamination)
    out_idx = rng.choice(n, size=n_out, replace=False)
    y_contaminated = y.copy()
    y_contaminated[out_idx] += rng.normal(scale=outlier_scale, size=n_out)

    theta_ols = ols_fit(X, y_contaminated)
    theta_huber, n_iter = huber_irls_fit(X, y_contaminated, delta=1.35)

    err_ols = np.linalg.norm(theta_ols - theta_true)
    err_huber = np.linalg.norm(theta_huber - theta_true)

    print("=" * 60)
    print("ROBUST FILTERING DEMO: OLS vs Huber IRLS")
    print("=" * 60)
    print(f"n={n}, contamination={contamination:.0%}, "
          f"outlier_scale={outlier_scale}")
    print(f"True theta:      {np.round(theta_true, 3)}")
    print(f"OLS theta:       {np.round(theta_ols, 3)}  "
          f"(L2 error: {err_ols:.4f})")
    print(f"Huber theta:     {np.round(theta_huber, 3)}  "
          f"(L2 error: {err_huber:.4f}, converged in {n_iter} iters)")
    print()
    improvement = (err_ols - err_huber) / err_ols * 100
    print(f"Huber reduces parameter error by {improvement:.1f}% "
          f"relative to OLS under this contamination level.")
    return {"err_ols": err_ols, "err_huber": err_huber, "n_iter": n_iter}


if __name__ == "__main__":
    run()
    print()
    # Sanity check: with zero contamination, Huber and OLS should
    # basically agree (Huber shouldn't hurt when there are no outliers).
    print("Sanity check with 0% contamination (Huber should ~match OLS):")
    run(contamination=0.0, outlier_scale=0.0)
