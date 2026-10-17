"""Leakage and split-protocol tests (use a small synthetic time series; no download)."""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (FeatureSpec, build_preprocessor, check_feature_spec, chronological_split, iid_split,
                                 nested_dev_split, validate_split)


def _df(n=500):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
                         "temp": rng.normal(size=n), "hr": np.arange(n) % 24, "weathersit": rng.integers(1, 4, n),
                         "cnt": rng.poisson(10, n)})


def test_chronological_split_order_and_sizes():
    df = _df()
    s = chronological_split(len(df), {"train": 0.6, "calibration": 0.2, "test": 0.2})
    assert (len(s.train), len(s.calibration), len(s.test)) == (300, 100, 100)
    validate_split(s, df["timestamp"])
    assert s.train.max() < s.calibration.min() < s.test.min()


def test_validate_detects_temporal_leakage():
    df = _df()
    s = iid_split(len(df), {"train": 0.6, "calibration": 0.2, "test": 0.2}, np.random.default_rng(0))
    s.kind = "chronological"
    with pytest.raises(ValueError):
        validate_split(s, df["timestamp"])


def test_validate_rejects_equal_timestamp_boundary():
    ts = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]))
    s = type("S", (), {"train": np.array([0, 1]), "calibration": np.array([2]),
                        "test": np.array([3]), "kind": "chronological"})()
    with pytest.raises(ValueError, match="strict time order"):
        validate_split(s, ts)


def test_nested_dev_split_stays_inside_training():
    outer = chronological_split(1000, {"train": 0.6, "calibration": 0.2, "test": 0.2})
    dev = nested_dev_split(outer.train, {"train": 0.6, "calibration": 0.2, "test": 0.2})
    assert set(np.concatenate([dev.train, dev.calibration, dev.test])) <= set(outer.train)
    assert dev.test.max() < outer.calibration.min()


def test_bad_ratios():
    with pytest.raises(ValueError):
        chronological_split(100, {"train": 0.7, "calibration": 0.2, "test": 0.2})


def test_forbidden_features_rejected():
    spec = FeatureSpec(numeric=["temp", "casual"])
    with pytest.raises(ValueError):
        check_feature_spec(spec, "cnt", ["casual", "registered"])
    with pytest.raises(ValueError):
        check_feature_spec(FeatureSpec(numeric=["cnt"]), "cnt")


def test_preprocessor_fitted_on_train_only():
    df = _df()
    spec = FeatureSpec(numeric=["temp"], categorical=["weathersit"], cyclic={"hr": 24})
    pre = build_preprocessor(spec).fit(df.iloc[:300])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(df["temp"].iloc[:300].mean())
    X = pre.transform(df.iloc[300:])
    assert X.shape[0] == 200 and np.isfinite(X).all()


def test_feature_spec_allows_intentional_cyclic_duplicate():
    spec = FeatureSpec(categorical=["weekday"], cyclic={"weekday": 7})
    check_feature_spec(spec, "cnt")


def test_feature_spec_rejects_conflicting_encoding_groups():
    with pytest.raises(ValueError):
        check_feature_spec(FeatureSpec(numeric=["x"], categorical=["x"]), "cnt")


def test_cyclic_encoding_continuity():
    df = _df(48)
    spec = FeatureSpec(cyclic={"hr": 24})
    X = build_preprocessor(spec).fit_transform(df)
    # hour 23 and hour 0 are neighbours on the circle
    assert np.linalg.norm(X[23] - X[24]) == pytest.approx(np.linalg.norm(X[0] - X[1]))
