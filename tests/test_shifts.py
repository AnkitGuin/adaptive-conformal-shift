import numpy as np
import pandas as pd
import pytest

from src.data import shifts as S


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({"temp": rng.uniform(0, 1, n), "hum": rng.uniform(0, 1, n), "cnt": rng.poisson(100, n).astype(float),
                         "workingday": rng.integers(0, 2, n), "hr": rng.integers(0, 24, n)})


SCALES = {"temp": 0.2, "hum": 0.1}


def _all(df, seed):
    rng = lambda: np.random.default_rng(seed)  # noqa: E731
    return [
        S.apply_gradual_covariate_shift(df, rng(), ["temp", "hum"], SCALES, 2.0, 0.1, 0.9),
        S.apply_abrupt_covariate_shift(df, rng(), ["temp"], SCALES, 1.5, 0.3, 0.05),
        S.apply_covariate_resample_shift(df, rng(), "hr", [7, 8, 17], 0.2, 0.3, 0.05),
        S.apply_noise_shift(df, rng(), "cnt", 0.5, 50.0, 0.3, 0.05),
        S.apply_conditional_shift(df, rng(), "cnt", 1.5, 0.3, 0.05, {"workingday": [1]}),
    ]


def test_no_mutation(df):
    before = df.copy(deep=True)
    _all(df, 0)
    pd.testing.assert_frame_equal(df, before)


def test_deterministic_with_seed(df):
    for a, b in zip(_all(df, 7), _all(df, 7)):
        pd.testing.assert_frame_equal(a.df, b.df)
        assert a.metadata["change_point"] == b.metadata["change_point"]


def test_different_seeds_differ(df):
    a = S.apply_noise_shift(df, np.random.default_rng(1), "cnt", 0.5, 50.0, 0.3, 0.05)
    b = S.apply_noise_shift(df, np.random.default_rng(2), "cnt", 0.5, 50.0, 0.3, 0.05)
    assert not a.df["cnt"].equals(b.df["cnt"])


def test_abrupt_magnitude(df):
    r = S.apply_abrupt_covariate_shift(df, np.random.default_rng(0), ["temp"], SCALES, 1.5, 0.3, 0.0)
    cp = r.metadata["change_point"]
    assert cp == 300
    d = r.df["temp"].to_numpy() - df["temp"].to_numpy()
    assert np.allclose(d[:cp], 0) and np.allclose(d[cp:], 1.5 * 0.2)
    assert np.allclose(r.df["cnt"], df["cnt"])  # target untouched


def test_gradual_ramp(df):
    r = S.apply_gradual_covariate_shift(df, np.random.default_rng(0), ["hum"], SCALES, 2.0, 0.1, 0.9)
    d = r.df["hum"].to_numpy() - df["hum"].to_numpy()
    assert np.allclose(d[:100], 0) and np.allclose(d[900:], 0.2)
    assert np.all(np.diff(d) >= -1e-12)


def test_noise_magnitude(df):
    r = S.apply_noise_shift(df, np.random.default_rng(0), "cnt", 1.0, 50.0, 0.3, 0.0)
    eps = r.df["cnt"].to_numpy() - df["cnt"].to_numpy()
    assert np.allclose(eps[:300], 0)
    assert 40 < eps[300:].std() < 60
    assert np.allclose(r.df["temp"], df["temp"])


def test_conditional_subset(df):
    r = S.apply_conditional_shift(df, np.random.default_rng(0), "cnt", 2.0, 0.5, 0.0, {"workingday": [1]})
    ratio = r.df["cnt"].to_numpy() / df["cnt"].replace(0, 1).to_numpy()
    aff = r.affected
    assert aff[:500].sum() == 0
    assert np.all(df["workingday"].to_numpy()[aff] == 1)
    assert np.allclose(ratio[aff & (df["cnt"].to_numpy() > 0)], 2.0)
    assert np.allclose(r.df["cnt"].to_numpy()[~aff], df["cnt"].to_numpy()[~aff])


def test_resample_keeps_order_and_prefix(df):
    r = S.apply_covariate_resample_shift(df, np.random.default_rng(0), "hr", [7, 8, 17], 0.2, 0.3, 0.0)
    cp = r.metadata["change_point"]
    pd.testing.assert_frame_equal(r.df.iloc[:cp], df.iloc[:cp].reset_index(drop=True))
    assert r.metadata["favored_share_post"] > r.metadata["favored_share_pre"]


def test_metadata_records_parameters(df):
    for r in _all(df, 0):
        assert "type" in r.metadata and "change_point" in r.metadata
