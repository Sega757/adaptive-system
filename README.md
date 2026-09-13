[![Tests](https://github.com/Sega757/adaptive-system/actions/workflows/tests.yml/badge.svg)](https://github.com/Sega757/adaptive-system/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Adaptive System: Inference Optimization & Robust Estimation

A lightweight Python library for compute cost reduction and robust signal filtering without heavy external dependencies. It tackles two practical problems: inference resource optimization and outlier mitigation in numerical pipelines.

---

## Core Modules

### 1. Entropy-Based Model Router (`entropy_router.py`)
A two-tier adaptive inference router driven by Shannon information entropy.

* **Architecture:** Routine inputs are handled by a fast, low-cost baseline model (Decision Tree). For every prediction, the router measures the entropy across the class probability distribution. If entropy exceeds the configured threshold (indicating ambiguous or borderline cases), execution escalates to an ensemble model (Random Forest).
* **Benchmark Results (Digits dataset):**
  * Fast Tier (Single Tree): **55%** accuracy, **~0.01s** latency
  * Slow Tier (Random Forest): **97%** accuracy, **~0.82s** latency
  * Adaptive Router: **95%** accuracy, **~0.04s** average latency (escalating only **62%** of ambiguous cases)

### 2. Robust Parameter Filter (`robust_filter.py`)
M-estimation using Huber Loss optimized via Iteratively Reweighted Least Squares (IRLS).

* **Architecture:** Unlike Ordinary Least Squares (OLS)—which penalizes residuals quadratically and shifts severely toward extreme anomalies—the Huber estimator penalizes large residuals linearly, stabilizing regression coefficients against noise.
* **Benchmark Results (15% synthetic outliers):**
  * Parameter error reduction vs. OLS: **91.8%**
  * Performance on clean, normally distributed data: asymptotically equivalent to OLS with negligible efficiency loss.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/Sega757/adaptive-system.git](https://github.com/Sega757/adaptive-system.git)
cd adaptive-system
pip install -r requirements.txt
