# Adaptive System — real prototypes, honestly measured

This is a small, working extraction of the two ideas from the
"SCCS" document that actually hold up, rebuilt from scratch with real
math and real measured numbers (not the invented architecture,
file names, or formulas from that document).

## What's here

### 1. `entropy_router.py` + `demo_router.py`
Uncertainty-gated compute routing: a cheap model handles most inputs;
when its output distribution has high Shannon entropy (i.e. it's
unsure), the sample is escalated to an expensive model instead.

This is a real, well-established idea (speculative decoding, cascade
classifiers, mixture-of-experts routing all use variants of it) — it
is *not* a hallucination-elimination system, and this code makes no
such claim. It's just a latency/accuracy tradeoff knob.

Run it:
```
python3 demo_router.py
```
Measured on sklearn's `digits` dataset (real run, your numbers may
vary slightly by seed):
- Fast-only (small decision tree): ~55% accuracy, ~0.01s
- Slow-only (300-tree random forest): ~97% accuracy, ~0.82s
- Entropy-gated router: ~95% accuracy, ~0.04s (escalates ~62% of
  samples — the genuinely ambiguous ones)

### 2. `robust_filter.py` + `demo_robust.py`
Huber loss M-estimation via IRLS — standard robust regression
(Huber, 1964) for when your data has outliers or corrupted
measurements you don't want dominating a least-squares fit.

Run it:
```
python3 demo_robust.py
```
Measured result: with 15% of targets corrupted by large outliers,
Huber IRLS cuts parameter estimation error by ~90%+ relative to
plain OLS, and matches OLS closely when there's no contamination.

## Tests
```
python3 tests/test_basic.py
```

## What this deliberately does NOT include
No cryptographic signing, no "Chronos" data plane, no game-theoretic
validator slashing, no carbon-aware bidding coefficients, no
five-phase pulse protocols. Those were unfalsifiable dressing in the
source material. If you actually need auditable signed outputs or
adversarial-robust multi-agent verification, those are real (much
larger) engineering problems worth scoping separately — this repo
just gives you the two components that were genuinely soundly-based
ideas, working and tested.
