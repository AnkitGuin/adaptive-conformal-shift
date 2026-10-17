import math

import numpy as np
import pytest

from src.conformal.quantiles import (conformal_quantile, conformal_rank, effective_sample_size, exponential_weights,
                                     weighted_conformal_quantile, weighted_quantile)


# ---------------------------------------------------------------- conformal rank / unweighted quantile
@pytest.mark.parametrize("n,alpha,k", [(9, 0.1, 9), (19, 0.1, 18), (99, 0.1, 90), (10, 0.1, 10), (4, 0.2, 4), (100, 0.05, 96)])
def test_conformal_rank(n, alpha, k):
    assert conformal_rank(n, alpha) == k


def test_conformal_quantile_is_order_statistic():
    s = np.arange(1, 20, dtype=float)  # 1..19, n=19 -> k=18
    assert conformal_quantile(s, 0.1) == 18.0
    rng = np.random.default_rng(0)
    assert conformal_quantile(rng.permutation(s), 0.1) == 18.0  # order-invariant


def test_conformal_quantile_infinite_when_rank_exceeds_n():
    assert math.isinf(conformal_quantile(np.arange(5.0), 0.1))  # ceil(6*0.9)=6 > 5


def test_conformal_quantile_differs_from_numpy_default():
    s = np.arange(1, 11, dtype=float)
    # numpy's linear interpolation gives 9.1 at 0.9; the conformal rule needs k=ceil(11*0.9)=10
    assert np.quantile(s, 0.9) == pytest.approx(9.1)
    assert conformal_quantile(s, 0.1) == 10.0


def test_finite_sample_coverage_monte_carlo():
    """Exchangeable data: P(s_{n+1} <= q) is in [1-alpha, 1-alpha+1/(n+1)]."""
    rng = np.random.default_rng(1)
    n, alpha, reps = 49, 0.1, 20000
    s = rng.exponential(size=(reps, n + 1))
    q = np.array([conformal_quantile(row[:n], alpha) for row in s])
    cov = np.mean(s[:, n] <= q)
    se = np.sqrt(0.9 * 0.1 / reps)
    assert 1 - alpha - 3 * se <= cov <= 1 - alpha + 1 / (n + 1) + 3 * se


def test_invalid_alpha():
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(3), 0.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(3), 1.0)


def test_conformal_rank_never_zero_near_alpha_one():
    assert conformal_rank(0, 1.0 - 1e-12) == 1
    assert math.isinf(conformal_quantile(np.array([]), 1.0 - 1e-12))


# ---------------------------------------------------------------- weighted quantile
def test_weighted_quantile_uniform_matches_inverse_cdf():
    v = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    w = np.ones(5)
    assert weighted_quantile(v, w, 0.2) == 1.0
    assert weighted_quantile(v, w, 0.21) == 2.0
    assert weighted_quantile(v, w, 0.6) == 3.0
    assert weighted_quantile(v, w, 1.0) == 5.0
    assert weighted_quantile(v, w, 0.0) == 1.0


def test_weighted_quantile_concentrated_weight():
    v = np.array([10.0, 20.0, 30.0, 40.0])
    w = np.array([0.0, 0.0, 1.0, 0.0])
    for level in (0.01, 0.5, 0.99):
        assert weighted_quantile(v, w, level) == 30.0


def test_weighted_quantile_sorted_vs_unsorted():
    rng = np.random.default_rng(3)
    v, w = rng.normal(size=50), rng.uniform(size=50)
    perm = rng.permutation(50)
    for level in (0.1, 0.5, 0.9):
        assert weighted_quantile(v, w, level) == weighted_quantile(v[perm], w[perm], level)


def test_weighted_quantile_scale_invariant_in_weights():
    rng = np.random.default_rng(4)
    v, w = rng.normal(size=30), rng.uniform(size=30)
    assert weighted_quantile(v, w, 0.7) == weighted_quantile(v, 123.4 * w, 0.7)


def test_weighted_quantile_ties_and_single_value():
    assert weighted_quantile(np.array([2.0]), np.array([1.0]), 0.9) == 2.0
    assert weighted_quantile(np.array([1.0, 1.0, 1.0, 2.0]), np.ones(4), 0.75) == 1.0


def test_weighted_quantile_errors():
    with pytest.raises(ValueError):
        weighted_quantile(np.array([]), np.array([]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, -1.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([np.nan, 2.0]), np.array([1.0, 1.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0]), np.array([1.0]), 0.5, test_weight=np.inf)


def test_weighting_is_not_multiplying_residuals():
    """Weights move probability mass; they must not rescale the scores."""
    v = np.array([1.0, 2.0, 100.0])
    w = np.array([0.01, 0.01, 10.0])
    q = weighted_quantile(v, w, 0.5)
    assert q == 100.0  # an actual score value, heavily weighted
    assert q in v


# ---------------------------------------------------------------- weighted conformal quantile
def test_weighted_conformal_equals_unweighted_with_equal_weights():
    rng = np.random.default_rng(5)
    for n in (5, 9, 19, 50, 237):
        s = rng.exponential(size=n)
        for alpha in (0.05, 0.1, 0.2, 0.5):
            assert weighted_conformal_quantile(s, np.ones(n), alpha, test_weight=1.0) == conformal_quantile(s, alpha)


def test_weighted_conformal_infinite_when_mass_insufficient():
    # 3 scores + test weight 1: max finite mass 0.75 < 0.9
    assert math.isinf(weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.ones(3), 0.1))


def test_weighted_conformal_recent_weights_track_recent_scores():
    old, new = np.full(1000, 1.0), np.full(200, 10.0)
    s = np.concatenate([old, new])
    q_flat = weighted_conformal_quantile(s, np.ones(len(s)), 0.1)
    q_recent = weighted_conformal_quantile(s, exponential_weights(len(s), 0.02), 0.1)
    assert q_flat == 10.0 or q_flat == 1.0
    assert q_recent == 10.0
    assert weighted_conformal_quantile(s, exponential_weights(len(s), 0.0), 0.3) == 1.0


def test_exponential_weights_ordering_and_ess():
    w = exponential_weights(5, 0.5)
    assert np.all(np.diff(w) > 0)  # newest (last) has the largest weight
    assert w[-1] == pytest.approx(np.exp(-0.5))  # newest stored score has age 1
    assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    lam = 0.01
    assert effective_sample_size(exponential_weights(20000, lam)) == pytest.approx(2 / lam, rel=0.01)
