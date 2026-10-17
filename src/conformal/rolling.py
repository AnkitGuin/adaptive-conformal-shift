"""Method 2 - rolling-window conformal.

The buffer starts with the calibration scores; after each test label is
observed its score is appended and only the most recent W scores are kept.
q_t is the unweighted finite-sample conformal quantile of the buffer.

Small W adapts quickly but q_t is noisy (the order statistic is estimated
from few points); large W is stable but slow.  No coverage guarantee holds
under shift; under exchangeability each q_t is a valid split-conformal
quantile for the next point.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.conformal.base import Calibrator
from src.conformal.quantiles import conformal_quantile


class RollingCalibrator(Calibrator):
    name = "rolling"

    def __init__(self, alpha: float, window: int, **kw) -> None:
        super().__init__(alpha, **kw)
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.buffer: deque[float] = deque(maxlen=self.window)

    def fit(self, cal_scores: np.ndarray) -> "RollingCalibrator":
        cal_scores = np.asarray(cal_scores, float)
        self._register(cal_scores)
        self.buffer.clear()
        self.buffer.extend(cal_scores[-self.window:])
        return self

    def _raw_threshold(self) -> float:
        return conformal_quantile(np.fromiter(self.buffer, float, len(self.buffer)), self.alpha)

    def _update(self, score: float, covered: bool) -> None:
        self.buffer.append(score)
