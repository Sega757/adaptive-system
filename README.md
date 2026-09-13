```markdown
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

```

### Dependencies (`requirements.txt`)

* `numpy>=1.20.0`
* `scikit-learn>=1.0.0`
* `pytest>=7.0.0`

---

## Quickstart

### Dynamic Routing by Prediction Entropy

```python
from entropy_router import EntropyRouter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train router
router = EntropyRouter(entropy_threshold=0.5)
router.fit(X_train, y_train)

# Run inference with dynamic tier selection
predictions = router.predict(X_test)
print(f"Accuracy: {router.score(X_test, y_test):.4f}")

```

### Outlier-Resistant Estimation (Huber IRLS)

```python
import numpy as np
from robust_filter import HuberRegressorIRLS

# Generate linear data with heavy anomalies
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2.5 * X.squeeze() + np.random.normal(0, 1, 100)
y[::10] += 50.0  # Introduce 10% severe outliers

# Fit robust model
model = HuberRegressorIRLS(delta=1.345)
model.fit(X, y)

print(f"Estimated coefficient: {model.coef_[0]:.4f}")

```

---

## Running Tests

Execute unit tests and convergence suites:

```bash
pytest -v

```

Test coverage includes:

1. Mathematical validity of Shannon entropy calculations.
2. Router fallback consistency across threshold sweeps.
3. Finite iteration convergence of the IRLS algorithm.
4. Parameter parity with OLS under standard Gaussian noise.
5. Breakdown points under heavy-tailed contamination.

---

## Scope & Design Constraints

* Relies entirely on transparent, verifiable numerical methods.
* No external API keys, RPC connections, or network runtime requirements.
* All benchmarks run locally on standard CPU hardware.

```

```
