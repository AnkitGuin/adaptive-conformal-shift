"""Method 1 - static split conformal (the baseline).

q = s_(ceil((n+1)(1-alpha))) over the n calibration scores, fixed for the
whole test stream.  Under exchangeability of calibration and test points,
P(Y in C(X)) >= 1 - alpha (marginally).  No feedback is used.
"""
from __future__ import annotations

import numpy as np

from src.conformal.base import Calibrator
from src.conformal.quantiles import conformal_quantile


class StaticSplitCalibrator(Calibrator):
    name = "static"

    def fit(self, cal_scores: np.ndarray) -> "StaticSplitCalibrator":
        cal_scores = np.asarray(cal_scores, float)
        self._register(cal_scores)
        self.n_cal = len(cal_scores)
        self.q = conformal_quantile(cal_scores, self.alpha)
        return self

    def _raw_threshold(self) -> float:
        return self.q
