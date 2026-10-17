import math

import numpy as np
import pytest

from src.conformal.adaptive import ACICalibrator, DAACPCalibrator
from src.conformal.base import absolute_residual_scores, build_intervals, cqr_scores, run_online
from src.conformal.rolling import RollingCalibrator
from src.conformal.split import StaticSplitCalibrator
from src.conformal.weighted import ExpWeightedCalibrator

ALL = [
    lambda a: StaticSplitCalibrator(a),
    lambda a: RollingCalibrator(a, window=100),
    lambda a: ExpWeightedCalibrator(a, decay=0.01, max_buffer=1000),
    lambda a: ACICalibrator(a, gamma=0.01, window=300),
    lambda a: DAACPCalibrator(a, decay=0.005, decay_fast=0.05, ph_threshold=10.0, max_buffer=1000),
    lambda a: DAACPCalibrator(a, kernel="box", decay=0.005, decay_fast=0.05, ph_threshold=10.0, max_buffer=1000),
    lambda a: DAACPCalibrator(a, rolling_updates=False, decay=0.005, decay_fast=0.05, ph_threshold=10.0, max_buffer=500),
]


def _stream(rng, n_cal=500, n_test=2000, shift_at=None, factor=3.0):
    y_hat = rng.normal(size=n_cal + n_test)
    noise = rng.normal(size=n_cal + n_test)
    if shift_at is not None:
        noise[n_cal + shift_at:] *= factor
    y = y_hat + noise
    return y_hat[:n_cal], y[:n_cal], y_hat[n_cal:], y[n_cal:]


def test_split_conformal_hand_example():
    cal = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])  # n=9, alpha=0.1 -> k=9 -> q=4.5
    c = StaticSplitCalibrator(0.1).fit(cal)
    assert c.threshold() == 4.5
    lo, hi = build_intervals(np.array([10.0]), np.array([10.0]), 4.5)
    assert (lo[0], hi[0]) == (5.5, 14.5)


def test_absolute_and_cqr_scores():
    assert np.allclose(absolute_residual_scores([1, 5], [3, 5]), [2, 0])
    s = cqr_scores(np.array([0.0, 5.0, 12.0]), np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0]))
    assert np.allclose(s, [1.0, -4.0, 2.0])


def test_online_coverage_uses_score_set_for_negative_cqr_threshold():
    """A negative CQR correction can make the exact prediction set empty.

    The display helper keeps bounds ordered for plotting, but that safeguard
    must not turn an empty score-defined set into a covered singleton.
    """
    res = run_online(
        StaticSplitCalibrator(0.1).fit(np.full(9, -2.0)),
        np.array([-1.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.5]),
    )
    assert res.q[0] == -2.0
    assert not res.covered[0]
    assert res.lower[0] <= res.upper[0]


@pytest.mark.parametrize("make", ALL)
def test_lower_le_upper(make):
    rng = np.random.default_rng(0)
    yh_c, y_c, yh_t, y_t = _stream(rng, shift_at=800)
    c = make(0.1).fit(absolute_residual_scores(y_c, yh_c))
    res = run_online(c, absolute_residual_scores(y_t, yh_t), yh_t, yh_t, y_t, signed_residuals=y_t - yh_t)
    assert np.all(res.lower <= res.upper)
    assert np.all(np.isfinite(res.lower)) and np.all(np.isfinite(res.upper))


def test_cqr_intervals_never_inverted():
    lo, hi = build_intervals(np.array([0.0, 0.0]), np.array([1.0, 0.2]), -0.3)
    assert np.all(lo <= hi)


@pytest.mark.parametrize("make", ALL)
def test_near_nominal_coverage_without_shift(make):
    """IID stream: all methods should be close to 90% (loose tolerance, finite sample)."""
    rng = np.random.default_rng(1)
    yh_c, y_c, yh_t, y_t = _stream(rng, n_cal=1000, n_test=4000)
    c = make(0.1).fit(absolute_residual_scores(y_c, yh_c))
    res = run_online(c, absolute_residual_scores(y_t, yh_t), yh_t, yh_t, y_t, signed_residuals=y_t - yh_t)
    assert abs(res.covered.mean() - 0.9) < 0.03


def test_static_split_marginal_coverage_monte_carlo():
    rng = np.random.default_rng(2)
    covs = []
    for _ in range(300):
        cal = np.abs(rng.normal(size=99))
        test = np.abs(rng.normal(size=200))
        q = StaticSplitCalibrator(0.1).fit(cal).threshold()
        covs.append(np.mean(test <= q))
    assert 0.895 <= np.mean(covs) <= 0.92  # theory: [0.90, 0.91]


@pytest.mark.parametrize("make", ALL[1:])
def test_online_protocol_is_causal(make):
    """Changing FUTURE scores must not change any PAST threshold."""
    rng = np.random.default_rng(3)
    yh_c, y_c, yh_t, y_t = _stream(rng, shift_at=500)
    cal = absolute_residual_scores(y_c, yh_c)
    s = absolute_residual_scores(y_t, yh_t)
    t_cut = 900
    s2 = s.copy(); s2[t_cut:] *= 50.0
    y2 = y_t.copy(); y2[t_cut:] = yh_t[t_cut:] + (y_t[t_cut:] - yh_t[t_cut:]) * 50.0
    r1 = run_online(make(0.1).fit(cal), s, yh_t, yh_t, y_t, signed_residuals=y_t - yh_t)
    r2 = run_online(make(0.1).fit(cal), s2, yh_t, yh_t, y2, signed_residuals=y2 - yh_t)
    # thresholds up to and including t_cut depend only on scores < t_cut
    assert np.array_equal(r1.q[: t_cut + 1], r2.q[: t_cut + 1])
    assert not np.array_equal(r1.q, r2.q)


def test_rolling_adapts_static_does_not():
    rng = np.random.default_rng(4)
    yh_c, y_c, yh_t, y_t = _stream(rng, n_test=3000, shift_at=1000, factor=3.0)
    cal, s = absolute_residual_scores(y_c, yh_c), absolute_residual_scores(y_t, yh_t)
    st = run_online(StaticSplitCalibrator(0.1).fit(cal), s, yh_t, yh_t, y_t)
    ro = run_online(RollingCalibrator(0.1, window=200).fit(cal), s, yh_t, yh_t, y_t)
    late = slice(2000, 3000)
    assert st.covered[late].mean() < 0.6
    assert ro.covered[late].mean() > 0.85


def test_daacp_detects_and_widens():
    rng = np.random.default_rng(5)
    yh_c, y_c, yh_t, y_t = _stream(rng, n_test=3000, shift_at=1000, factor=3.0)
    cal, s = absolute_residual_scores(y_c, yh_c), absolute_residual_scores(y_t, yh_t)
    c = DAACPCalibrator(0.1, decay=0.001, decay_fast=0.02, ph_threshold=20.0, max_buffer=3000).fit(cal)
    res = run_online(c, s, yh_t, yh_t, y_t)
    alarms = [e.t for e in c.events]
    assert any(1000 <= t < 1300 for t in alarms)
    assert res.q[1400] > 2 * res.q[900]


def test_daacp_without_drift_equals_weighted():
    rng = np.random.default_rng(6)
    yh_c, y_c, yh_t, y_t = _stream(rng, shift_at=700)
    cal, s = absolute_residual_scores(y_c, yh_c), absolute_residual_scores(y_t, yh_t)
    a = run_online(DAACPCalibrator(0.1, decay=0.01, drift_enabled=False, max_buffer=800).fit(cal), s, yh_t, yh_t, y_t)
    b = run_online(ExpWeightedCalibrator(0.1, decay=0.01, max_buffer=800).fit(cal), s, yh_t, yh_t, y_t)
    assert np.allclose(a.q, b.q)


def test_box_kernel_matches_rolling_when_no_drift():
    rng = np.random.default_rng(7)
    yh_c, y_c, yh_t, y_t = _stream(rng)
    cal, s = absolute_residual_scores(y_c, yh_c), absolute_residual_scores(y_t, yh_t)
    lam = 2 / 200  # box horizon 200
    a = run_online(DAACPCalibrator(0.1, kernel="box", decay=lam, drift_enabled=False, max_buffer=1000).fit(cal), s, yh_t, yh_t, y_t)
    b = run_online(RollingCalibrator(0.1, window=200).fit(cal), s, yh_t, yh_t, y_t)
    assert np.allclose(a.q, b.q)


def test_aci_long_run_coverage_under_shift():
    rng = np.random.default_rng(8)
    yh_c, y_c, yh_t, y_t = _stream(rng, n_test=6000, shift_at=2000, factor=3.0)
    cal, s = absolute_residual_scores(y_c, yh_c), absolute_residual_scores(y_t, yh_t)
    res = run_online(ACICalibrator(0.1, gamma=0.01, window=500).fit(cal), s, yh_t, yh_t, y_t)
    assert abs(res.covered.mean() - 0.9) < 0.02


def test_infinite_threshold_fallback_is_counted():
    c = RollingCalibrator(0.1, window=5).fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    q = c.threshold()
    assert q == 5.0 and c.n_infinite == 1
    c2 = RollingCalibrator(0.1, window=5, inf_fallback="none").fit(np.arange(5.0))
    assert math.isinf(c2.threshold())


def test_signed_stream_requires_signed_residuals():
    c = DAACPCalibrator(0.1, drift_stream="signed").fit(np.ones(100))
    with pytest.raises(ValueError):
        c.update(1.0, True)
    c.update(1.0, True, signed=-1.0)
