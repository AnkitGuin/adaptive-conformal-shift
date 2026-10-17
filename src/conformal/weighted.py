"""Method 3 - exponentially weighted conformal.

Buffer (oldest -> newest) of at most ``max_buffer`` scores.  The score of age
k (newest stored score: k = 1) receives weight w_k = exp(-lambda * k); the
future test point has age 0 and weight 1.  q_t is the weighted conformal
quantile (``quantiles.weighted_conformal_quantile``), i.e. the smallest score
s such that the normalised weight of scores <= s reaches 1 - alpha, with the
test point's own mass placed at +inf.

This is a fixed-weight instance of non-exchangeable conformal prediction
(Barber et al., 2023).  Its coverage gap relative to 1 - alpha is bounded by a
weighted total-variation term that is small only when the recent past is
distributed like the present; it is NOT guaranteed under arbitrary shifts.

Note: residuals are **not** multiplied by the weights - weights enter the
CDF of the scores, not the scores themselves.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.conformal.base import Calibrator
from src.conformal.quantiles import effective_sample_size, exponential_weights, weighted_conformal_quantile


class ExpWeightedCalibrator(Calibrator):
    name = "weighted"

    def __init__(self, alpha: float, decay: float, max_buffer: int | None = None, **kw) -> None:
        super().__init__(alpha, **kw)
        if decay < 0:
            raise ValueError("decay must be >= 0")
        self.decay = float(decay)
        self.max_buffer = None if max_buffer in (None, 0) else int(max_buffer)
        self.buffer: deque[float] = deque(maxlen=self.max_buffer)

    def fit(self, cal_scores: np.ndarray) -> "ExpWeightedCalibrator":
        cal_scores = np.asarray(cal_scores, float)
        self._register(cal_scores)
        self.buffer.clear()
        self.buffer.extend(cal_scores if self.max_buffer is None else cal_scores[-self.max_buffer:])
        return self

    def current_decay(self) -> float:
        return self.decay

    def _raw_threshold(self) -> float:
        s = np.fromiter(self.buffer, float, len(self.buffer))
        w = exponential_weights(len(s), self.current_decay())
        return weighted_conformal_quantile(s, w, self.alpha, test_weight=1.0)

    def effective_n(self) -> float:
        return effective_sample_size(exponential_weights(len(self.buffer), self.current_decay()))

    def _update(self, score: float, covered: bool) -> None:
        self.buffer.append(score)
