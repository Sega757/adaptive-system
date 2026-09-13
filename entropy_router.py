"""
Entropy-gated routing between a cheap ("fast path") and an expensive
("slow path") model.

The idea (this part is genuinely standard and well-established in ML,
unlike the pseudo-architecture in the source document): when a cheap
model's output distribution over classes is low-entropy (confident),
trust it and skip the expensive model. When entropy is high
(ambiguous input), spend the extra compute on the expensive model.

This module makes no claims about hallucination elimination, legal-grade
verification, or anything cryptographic — it's just uncertainty-gated
compute allocation, measured honestly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def shannon_entropy(prob_dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy H(X) = -sum(p_i * log2(p_i)) for one or more
    probability distributions (rows of `prob_dist`).

    Returns entropy in bits. Accepts a single 1D distribution or a 2D
    array of shape (n_samples, n_classes).
    """
    p = np.clip(prob_dist, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=-1)


@dataclass
class RoutingStats:
    n_total: int = 0
    n_escalated: int = 0
    fast_time_s: float = 0.0
    slow_time_s: float = 0.0
    entropies: list = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / self.n_total if self.n_total else 0.0

    @property
    def total_time_s(self) -> float:
        return self.fast_time_s + self.slow_time_s


class EntropyGatedRouter:
    """Routes each sample to a cheap model, and escalates to an expensive
    model only when the cheap model's predicted-class entropy exceeds
    `tau_high`.

    Both models must implement `predict_proba(X)` (sklearn-style API).
    """

    def __init__(self, fast_model, slow_model, tau_high: float):
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.tau_high = tau_high
        self.stats = RoutingStats()

    def fit(self, X_train, y_train):
        self.fast_model.fit(X_train, y_train)
        self.slow_model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict labels for X, escalating per-sample based on entropy.
        Returns (predictions, per_sample_escalated_mask).
        """
        n = len(X)
        preds = np.empty(n, dtype=object)
        escalated = np.zeros(n, dtype=bool)

        t0 = time.perf_counter()
        fast_proba = self.fast_model.predict_proba(X)
        t1 = time.perf_counter()
        self.stats.fast_time_s += (t1 - t0)

        fast_labels = self.fast_model.classes_[np.argmax(fast_proba, axis=1)]
        entropies = shannon_entropy(fast_proba)
        self.stats.entropies.extend(entropies.tolist())

        escalate_mask = entropies > self.tau_high
        preds[:] = fast_labels
        escalated[:] = escalate_mask

        n_escalate = int(escalate_mask.sum())
        if n_escalate > 0:
            X_escalate = X[escalate_mask] if hasattr(X, "__getitem__") else [
                x for x, m in zip(X, escalate_mask) if m
            ]
            t0 = time.perf_counter()
            slow_proba = self.slow_model.predict_proba(X_escalate)
            t1 = time.perf_counter()
            self.stats.slow_time_s += (t1 - t0)
            slow_labels = self.slow_model.classes_[np.argmax(slow_proba, axis=1)]
            preds[escalate_mask] = slow_labels

        self.stats.n_total += n
        self.stats.n_escalated += n_escalate
        return preds, escalated
