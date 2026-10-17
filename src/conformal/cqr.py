"""Conformalized Quantile Regression (Romano, Patterson & Candes, 2019) - benchmark.

Two quantile regressors are fitted on the training split at levels
alpha/2 and 1 - alpha/2.  On calibration data the score is

    s_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

and the interval is [q_lo(x) - q, q_hi(x) + q], where q is produced by any
``Calibrator`` (static split quantile, rolling, weighted, DA-ACP ...).  The
same calibrator classes as for absolute residuals are reused - only the score
function changes - which keeps the conformal logic in one place.

CQR produces asymmetric, input-dependent intervals; this matters for the
heteroscedastic, right-skewed bike counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.conformal.base import cqr_scores
from src.data.preprocess import FeatureSpec
from src.models.regressors import build_quantile_model


@dataclass
class CQRBands:
    lo_cal: np.ndarray
    hi_cal: np.ndarray
    lo_test: np.ndarray
    hi_test: np.ndarray

    def cal_scores(self, y_cal: np.ndarray) -> np.ndarray:
        return cqr_scores(y_cal, self.lo_cal, self.hi_cal)

    def test_scores(self, y_test: np.ndarray) -> np.ndarray:
        return cqr_scores(y_test, self.lo_test, self.hi_test)


def fit_cqr_bands(X_train, y_train, X_cal, X_test, alpha: float, spec: FeatureSpec, params: dict, seed: int,
                  kind: str = "hgb") -> CQRBands:
    lo_m = build_quantile_model(alpha / 2, params, spec, seed, kind)
    hi_m = build_quantile_model(1 - alpha / 2, params, spec, seed + 1, kind)
    lo_m.fit(X_train, y_train)
    hi_m.fit(X_train, y_train)
    lo_cal, hi_cal = lo_m.predict(X_cal), hi_m.predict(X_cal)
    lo_te, hi_te = lo_m.predict(X_test), hi_m.predict(X_test)
    # Quantile crossing fix (sort per point) - standard, does not use labels.
    lo_cal, hi_cal = np.minimum(lo_cal, hi_cal), np.maximum(lo_cal, hi_cal)
    lo_te, hi_te = np.minimum(lo_te, hi_te), np.maximum(lo_te, hi_te)
    return CQRBands(lo_cal, hi_cal, lo_te, hi_te)
