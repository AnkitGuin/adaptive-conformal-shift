"""Point and interval metrics."""
from __future__ import annotations

import numpy as np


def coverage_indicator(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    return (np.asarray(lower, float) <= y) & (y <= np.asarray(upper, float))


def empirical_coverage(y, lower, upper) -> float:
    return float(np.mean(coverage_indicator(y, lower, upper)))


def mean_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def coverage_gap(coverage: float, nominal: float) -> float:
    return abs(coverage - nominal)


def interval_score(y, lower, upper, alpha: float) -> np.ndarray:
    """Winkler / Gneiting-Raftery (2007) interval score; lower is better. Returns per-point scores.

    S = (U - L) + (2/alpha)(L - y) 1{y < L} + (2/alpha)(y - U) 1{y > U}
    """
    y, lo, hi = (np.asarray(a, float) for a in (y, lower, upper))
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def mean_interval_score(y, lower, upper, alpha: float) -> float:
    return float(np.mean(interval_score(y, lower, upper, alpha)))


def mae(y, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(y_pred, float))))


def rmse(y, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(y_pred, float)) ** 2)))


def r2(y, y_pred) -> float:
    y, p = np.asarray(y, float), np.asarray(y_pred, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
