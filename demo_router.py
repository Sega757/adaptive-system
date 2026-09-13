"""
Demo: entropy-gated routing on the sklearn `digits` dataset.

Fast path  = a single small Decision Tree (cheap, less accurate).
Slow path  = a Random Forest with 300 trees (expensive, more accurate).

We compare three strategies:
  1. Always use fast model only.
  2. Always use slow model only.
  3. Entropy-gated router: use fast model, escalate to slow model only
     when the fast model's prediction entropy is high.

All numbers below are measured, not asserted.
"""

import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from entropy_router import EntropyGatedRouter, shannon_entropy


def run(tau_high: float = 1.0, seed: int = 0):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )

    fast_model = DecisionTreeClassifier(max_depth=4, random_state=seed)
    slow_model = RandomForestClassifier(n_estimators=300, random_state=seed)

    # Baseline: fast only
    t0 = time.perf_counter()
    fast_model.fit(X_train, y_train)
    fast_preds = fast_model.predict(X_test)
    t_fast_only = time.perf_counter() - t0
    fast_acc = (fast_preds == y_test).mean()

    # Baseline: slow only
    t0 = time.perf_counter()
    slow_model2 = RandomForestClassifier(n_estimators=300, random_state=seed)
    slow_model2.fit(X_train, y_train)
    slow_preds = slow_model2.predict(X_test)
    t_slow_only = time.perf_counter() - t0
    slow_acc = (slow_preds == y_test).mean()

    # Entropy-gated router
    router = EntropyGatedRouter(
        fast_model=DecisionTreeClassifier(max_depth=4, random_state=seed),
        slow_model=RandomForestClassifier(n_estimators=300, random_state=seed),
        tau_high=tau_high,
    )
    t0 = time.perf_counter()
    router.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    router_preds, escalated = router.predict(X_test)
    predict_time = time.perf_counter() - t0
    router_acc = (router_preds == y_test).mean()

    print("=" * 60)
    print("ENTROPY-GATED ROUTING DEMO (sklearn digits, real numbers)")
    print("=" * 60)
    print(f"Test set size: {len(y_test)}")
    print(f"tau_high (entropy threshold, bits): {tau_high}")
    print()
    print(f"{'Strategy':<25}{'Accuracy':<12}{'Inference time (s)'}")
    print(f"{'Fast only (tree)':<25}{fast_acc:<12.4f}{t_fast_only:.4f}")
    print(f"{'Slow only (forest)':<25}{slow_acc:<12.4f}{t_slow_only:.4f}")
    print(f"{'Entropy-gated router':<25}{router_acc:<12.4f}{predict_time:.4f}")
    print()
    print(f"Escalation rate: {router.stats.escalation_rate:.1%} "
          f"({router.stats.n_escalated}/{router.stats.n_total} samples "
          f"sent to slow model)")
    print(f"Entropy stats: mean={np.mean(router.stats.entropies):.3f} bits, "
          f"max={np.max(router.stats.entropies):.3f} bits")
    return {
        "fast_acc": fast_acc, "slow_acc": slow_acc, "router_acc": router_acc,
        "t_fast_only": t_fast_only, "t_slow_only": t_slow_only,
        "predict_time": predict_time,
        "escalation_rate": router.stats.escalation_rate,
    }


if __name__ == "__main__":
    run(tau_high=1.0)
