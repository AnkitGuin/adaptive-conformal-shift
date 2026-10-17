import numpy as np
import pytest

from src.drift.page_hinkley import PageHinkley


def test_detects_increase_after_shift():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.exponential(1.0, 1000), rng.exponential(3.0, 1000)])
    ph = PageHinkley(delta=0.1, threshold=30.0, min_instances=30)
    alarms = [ev.t for x in z if (ev := ph.update(x)) is not None]
    after = [t for t in alarms if t >= 1000]
    assert after and after[0] - 1000 < 100


def test_estimated_onset_close_to_true_change():
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.normal(1, 0.3, 800), rng.normal(2, 0.3, 400)])
    ph = PageHinkley(delta=0.05, threshold=20.0)
    ev = next(ev for x in z if (ev := ph.update(x)) is not None)
    assert abs(ev.onset - 800) <= 10 and ev.t >= 800


def test_few_false_alarms_on_stationary_stream():
    rng = np.random.default_rng(2)
    ph = PageHinkley(delta=0.1, threshold=50.0)
    n_alarms = sum(ph.update(x) is not None for x in rng.exponential(1.0, 20000))
    assert n_alarms <= 2


def test_decrease_direction():
    z = np.concatenate([np.full(500, 2.0), np.full(500, 0.5)])
    up = PageHinkley(delta=0.05, threshold=10.0, direction="increase")
    down = PageHinkley(delta=0.05, threshold=10.0, direction="decrease")
    assert not any(up.update(x) for x in z)
    assert any(down.update(x) for x in z)


def test_reset_after_alarm_and_min_instances():
    ph = PageHinkley(delta=0.0, threshold=1.0, min_instances=10)
    for _ in range(9):
        assert ph.update(100.0 if _ == 8 else 0.0) is None  # no alarm before min_instances
    z = [0.0] * 50 + [5.0] * 5
    events = [e for x in z if (e := ph.update(x))]
    assert events and ph.n < 10  # detector reset after the alarm


def test_invalid_parameters():
    with pytest.raises(ValueError):
        PageHinkley(threshold=0)
    with pytest.raises(ValueError):
        PageHinkley(direction="sideways")
    with pytest.raises(ValueError):
        PageHinkley(min_instances=0)
    with pytest.raises(ValueError):
        PageHinkley().update(np.inf)
