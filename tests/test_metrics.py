import numpy as np
import pytest

from src.evaluation import coverage as cov
from src.evaluation import metrics as M
from src.evaluation.statistics import mean_std_ci, paired_bootstrap_ci, wilcoxon_p


def test_coverage_toy():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 2.0, 3.5, 0.0])
    hi = np.array([2.0, 2.0, 4.0, 3.0])
    assert M.coverage_indicator(y, lo, hi).tolist() == [True, True, False, False]
    assert M.empirical_coverage(y, lo, hi) == 0.5
    assert M.mean_width(lo, hi) == pytest.approx((2 + 0 + 0.5 + 3) / 4)
    assert M.coverage_gap(0.5, 0.9) == pytest.approx(0.4)


def test_interval_score():
    alpha = 0.1
    s = M.interval_score(np.array([5.0, 0.0, 12.0]), np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0]), alpha)
    assert s.tolist() == pytest.approx([9.0, 9.0 + 20 * 1.0, 9.0 + 20 * 2.0])


def test_point_metrics():
    assert M.mae([1, 2], [2, 4]) == 1.5
    assert M.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_rolling_and_worst_window():
    c = np.array([1] * 10 + [0] * 5 + [1] * 10, float)
    rc = cov.rolling_coverage(c, 5)
    assert np.isnan(rc[:4]).all() and rc[4] == 1.0 and np.nanmin(rc) == 0.0
    assert cov.worst_window_coverage(c, 5) == 0.0


def test_recovery_time_and_delay():
    c = np.array([1] * 100 + [0] * 30 + [1] * 200, float)
    rt = cov.recovery_time(c, 100, nominal=0.9, window=20, tol=0.1, sustain=10)
    # post-change rolling coverage first reaches >= 0.8 once 16 of the last 20 are covered: index 45
    assert rt == 45.0
    assert np.isnan(cov.recovery_time(np.zeros(300), 100, 0.9, 20, 0.05, 10))
    delay, false, after = cov.detection_delay(np.array([0, 1, 0, 0, 0, 1, 0, 1], bool), 3)
    assert (delay, false, after) == (2.0, 1, 2)


def test_statistics():
    d = mean_std_ci(np.array([1.0, 2.0, 3.0]))
    assert d["mean"] == 2.0 and d["ci_lo"] < 2 < d["ci_hi"]
    m, lo, hi = paired_bootstrap_ci(np.ones(10) * 2, np.ones(10), rng=np.random.default_rng(0))
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    assert wilcoxon_p(np.arange(10.0) + 1, np.zeros(10)) < 0.01
