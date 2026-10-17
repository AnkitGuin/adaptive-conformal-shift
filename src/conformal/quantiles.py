"""Finite-sample conformal quantiles (unweighted and weighted).

This module contains the central mathematical operation of the project.

Conventions (used everywhere in the repository)
-----------------------------------------------
Let s_1, ..., s_n be calibration nonconformity scores and alpha in (0, 1) the
target miscoverage level.

**Unweighted split-conformal quantile.**  We use the order statistic

    q = s_(k),   k = ceil((n + 1) * (1 - alpha)),

i.e. the k-th smallest score.  If k > n the quantile is +inf (the interval is
the whole real line).  Under exchangeability of the n calibration scores and
the test score this gives P(s_{n+1} <= q) >= 1 - alpha, and, if scores are
almost surely distinct, also P(s_{n+1} <= q) <= 1 - alpha + 1/(n + 1)
(Vovk et al., 2005; Lei et al., 2018).  We never use NumPy's default linear
interpolation, which has no finite-sample guarantee.

**Weighted conformal quantile.**  Given non-negative weights w_1..w_n for the
calibration scores and a weight w_test for the (unknown) test score, define
normalised masses

    p_i = w_i / (sum_j w_j + w_test),     p_test = w_test / (sum_j w_j + w_test)

and put the test mass on +inf.  The weighted quantile is

    q = inf { s : sum_{i : s_i <= s} p_i >= 1 - alpha }

(+inf if the finite masses never reach 1 - alpha).  With all weights equal
this reduces *exactly* to the unweighted rule above.  This is the form used
by weighted / non-exchangeable conformal prediction (Tibshirani et al., 2019;
Barber et al., 2023).  Note: with fixed, data-independent weights the
coverage guarantee of Barber et al. (2023) degrades gracefully by a term that
depends on how far the data are from exchangeable; it is NOT an unconditional
guarantee under arbitrary distribution shift.

A tiny tolerance (``_TOL``) is used in the ">= 1 - alpha" comparison so that
floating-point round-off in cumulative sums (e.g. 0.8999999999 vs 0.9) never
changes the selected order statistic.
"""
from __future__ import annotations

import math

import numpy as np

_TOL = 1e-10


def conformal_rank(n: int, alpha: float) -> int:
    """Return k = ceil((n + 1)(1 - alpha)), the 1-based rank of the conformal quantile.

    Computed with a small tolerance so that exact integers are not pushed up
    by floating-point error (e.g. (9 + 1) * 0.9 = 9.000000000000002).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n < 0:
        raise ValueError("n must be non-negative")
    # The exact rank is always at least one for alpha < 1.  The tolerance is
    # only allowed to repair an almost-integer round-up; without this clamp it
    # could incorrectly produce rank zero when alpha is extremely close to 1.
    return max(1, int(math.ceil((n + 1) * (1.0 - alpha) - 1e-9)))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Unweighted finite-sample split-conformal quantile.

    Returns the ceil((n+1)(1-alpha))-th smallest score, or +inf if that rank
    exceeds n.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if np.any(~np.isfinite(s)):
        raise ValueError("scores must be finite")
    n = s.size
    k = conformal_rank(n, alpha)
    if k > n:
        return math.inf
    # np.partition gives the k-th smallest in O(n)
    return float(np.partition(s, k - 1)[k - 1])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Inverse-CDF (type-1, no interpolation) quantile of a weighted empirical distribution.

    Returns inf { v : sum_{i: v_i <= v} w_i / sum_j w_j >= level }.

    This is the standard "lower" weighted quantile.  Values are sorted
    internally, so the input order is irrelevant.  Zero weights are allowed;
    negative weights are not.  ``level`` must lie in [0, 1].
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if v.size == 0:
        raise ValueError("weighted_quantile of an empty array")
    if np.any(~np.isfinite(v)):
        raise ValueError("values must be finite")
    if np.any(w < 0) or np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if not 0.0 <= level <= 1.0:
        raise ValueError("level must be in [0, 1]")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights sum to zero")
    order = np.argsort(v, kind="mergesort")
    cdf = np.cumsum(w[order]) / total
    idx = int(np.searchsorted(cdf, level - _TOL, side="left"))
    idx = min(idx, v.size - 1)
    return float(v[order][idx])


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    test_weight: float = 1.0,
) -> float:
    """Weighted conformal quantile with a point mass ``test_weight`` at +inf.

    See the module docstring for the definition.  Returns +inf when the
    finite calibration mass cannot reach 1 - alpha (for example when very
    few effective samples remain after strong down-weighting).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(scores, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if s.shape != w.shape:
        raise ValueError("scores and weights must have the same shape")
    if np.any(~np.isfinite(s)):
        raise ValueError("scores must be finite")
    if np.any(w < 0) or np.any(~np.isfinite(w)) or not np.isfinite(test_weight) or test_weight < 0:
        raise ValueError("weights must be finite and non-negative")
    if s.size == 0:
        return math.inf
    total = w.sum() + test_weight
    if total <= 0:
        raise ValueError("all weights are zero")
    order = np.argsort(s, kind="mergesort")
    cum = np.cumsum(w[order]) / total
    target = 1.0 - alpha
    idx = int(np.searchsorted(cum, target - _TOL, side="left"))
    if idx >= s.size:
        return math.inf
    return float(s[order][idx])


def exponential_weights(n: int, decay: float) -> np.ndarray:
    """Weights w = exp(-decay * age) for a buffer ordered oldest -> newest.

    The newest stored score has age 1, the oldest has age n.  The (future)
    test point has age 0 and therefore weight exp(0) = 1, which is why the
    conformal routines use ``test_weight = 1``.
    """
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("decay must be finite and non-negative")
    ages = np.arange(n, 0, -1, dtype=float)
    return np.exp(-decay * ages)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0
