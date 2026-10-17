"""Feature construction, chronological splitting and split validation.

Leakage rules enforced here
---------------------------
* All fitted transformations (imputation, scaling, one-hot categories) live
  inside an sklearn ``ColumnTransformer`` that is fitted on the *training
  split only* (it is the first stage of the model pipeline).
* Cyclic encodings sin/cos(2*pi*v/period) are stateless (no fitting).
* No feature uses future targets (no lags/rolling statistics of the target).
* Columns that trivially leak the target (``casual`` + ``registered`` = ``cnt``)
  are never listed as features; :func:`check_feature_spec` raises if they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass
class FeatureSpec:
    numeric: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    cyclic: dict[str, float] = field(default_factory=dict)  # column -> period
    onehot_extra: list[str] = field(default_factory=list)   # e.g. hour as one-hot as well

    @classmethod
    def from_config(cls, features_cfg: dict, drop: list[str] | None = None) -> "FeatureSpec":
        drop = set(drop or [])
        return cls(
            numeric=[c for c in features_cfg.get("numeric", []) if c not in drop],
            binary=[c for c in features_cfg.get("binary", []) if c not in drop],
            categorical=[c for c in features_cfg.get("categorical", []) if c not in drop],
            cyclic={c: float(p) for c, p in (features_cfg.get("cyclic") or {}).items() if c not in drop},
            onehot_extra=[c for c in features_cfg.get("onehot_extra", []) if c not in drop],
        )

    @property
    def raw_columns(self) -> list[str]:
        cols = self.numeric + self.binary + self.categorical + list(self.cyclic) + self.onehot_extra
        return list(dict.fromkeys(cols))


def check_feature_spec(spec: FeatureSpec, target: str, forbidden: list[str] | None = None) -> None:
    forbidden = set(forbidden or []) | {target}
    bad = forbidden.intersection(spec.raw_columns)
    if bad:
        raise ValueError(f"Feature list contains target/leaking columns: {sorted(bad)}")
    if any(not np.isfinite(period) or period <= 0 for period in spec.cyclic.values()):
        raise ValueError("cyclic feature periods must be finite and positive")
    groups = {
        "numeric": spec.numeric,
        "binary": spec.binary,
        "categorical": spec.categorical,
    }
    seen: dict[str, str] = {}
    for group, columns in groups.items():
        for column in columns:
            previous = seen.get(column)
            if previous is not None and previous != group:
                # A raw column may intentionally be represented both as a
                # categorical feature and as a cyclic feature (and
                # ``onehot_extra`` is explicitly an additional representation).
                # Conflicts among the ordinary groups, however, are almost
                # certainly a configuration mistake.
                raise ValueError(f"feature '{column}' appears in both {previous} and {group}")
            seen[column] = group


def _cyclic_transform(X: pd.DataFrame | np.ndarray, periods: tuple[float, ...]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = []
    for j, p in enumerate(periods):
        out.append(np.sin(2 * np.pi * X[:, j] / p))
        out.append(np.cos(2 * np.pi * X[:, j] / p))
    return np.column_stack(out)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """ColumnTransformer; must be fitted on training data only (done inside the model pipeline)."""
    transformers = []
    if spec.numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                              ("scale", StandardScaler())]), spec.numeric))
    if spec.binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), spec.binary))
    cat_cols = spec.categorical + [c for c in spec.onehot_extra if c not in spec.categorical]
    if cat_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                             cat_cols))
    if spec.cyclic:
        periods = tuple(spec.cyclic.values())
        transformers.append(("cyc", FunctionTransformer(_cyclic_transform, kw_args={"periods": periods}),
                             list(spec.cyclic)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


@dataclass
class Split:
    """Index positions (into the chronologically ordered DataFrame) of each split."""
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    kind: str = "chronological"


def _boundaries(n: int, ratios: dict) -> tuple[int, int]:
    r_tr, r_cal, r_te = float(ratios["train"]), float(ratios["calibration"]), float(ratios["test"])
    if min(r_tr, r_cal, r_te) <= 0 or abs(r_tr + r_cal + r_te - 1.0) > 1e-8:
        raise ValueError(f"split ratios must be positive and sum to 1, got {ratios}")
    a = int(round(n * r_tr))
    b = int(round(n * (r_tr + r_cal)))
    return a, b


def chronological_split(n: int, ratios: dict) -> Split:
    """First ``train`` fraction -> train, next -> calibration, final -> test. No shuffling."""
    a, b = _boundaries(n, ratios)
    idx = np.arange(n)
    return Split(idx[:a], idx[a:b], idx[b:], kind="chronological")


def iid_split(n: int, ratios: dict, rng: np.random.Generator) -> Split:
    """Random permutation split (exchangeable CONTROL experiment only, never the primary protocol).

    Within each split, indices are re-sorted so the online methods still see
    the (permuted-membership) points in time order; exchangeability holds
    because split membership is random.
    """
    a, b = _boundaries(n, ratios)
    perm = rng.permutation(n)
    return Split(np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:]), kind="iid")


def nested_dev_split(outer_train: np.ndarray, ratios: dict) -> Split:
    """Chronological train/cal/test split *inside* the outer training portion (development protocol)."""
    inner = chronological_split(len(outer_train), ratios)
    return Split(outer_train[inner.train], outer_train[inner.calibration], outer_train[inner.test], kind="dev")


def validate_split(split: Split, timestamps: pd.Series | None = None) -> None:
    """Raise if splits overlap, are empty or (for chronological splits) are not time ordered."""
    parts = [split.train, split.calibration, split.test]
    if any(len(p) == 0 for p in parts):
        raise ValueError("empty split")
    all_idx = np.concatenate(parts)
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("splits overlap")
    if split.kind in ("chronological", "dev") and timestamps is not None:
        ts = pd.Series(timestamps).reset_index(drop=True)
        # Strict inequalities prevent rows sharing a timestamp from being
        # split across train/calibration/test.  Treating equal timestamps as
        # ordered would allow same-time information to cross a boundary.
        if not (ts.iloc[split.train].max() < ts.iloc[split.calibration].min()
                and ts.iloc[split.calibration].max() < ts.iloc[split.test].min()):
            raise ValueError("chronological split violates strict time order (possible temporal leakage)")


def dataset_statistics(df: pd.DataFrame, split: Split, target: str) -> pd.DataFrame:
    rows = []
    for name, idx in [("train", split.train), ("calibration", split.calibration), ("test", split.test), ("all", np.arange(len(df)))]:
        sub = df.iloc[idx]
        y = sub[target]
        rows.append({
            "split": name, "n": len(sub),
            "start": str(sub["timestamp"].min()), "end": str(sub["timestamp"].max()),
            "target_mean": y.mean(), "target_std": y.std(), "target_median": y.median(),
            "target_min": y.min(), "target_max": y.max(),
            "temp_mean": sub["temp"].mean() if "temp" in sub else np.nan,
        })
    return pd.DataFrame(rows)
