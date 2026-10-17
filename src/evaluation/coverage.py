"""Time-local coverage and adaptation metrics.

All functions take a boolean/0-1 coverage sequence ordered in time.
"""
from __future__ import annotations

import numpy as np


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean; entries before a full window are NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def rolling_coverage(covered: np.ndarray, window: int = 100) -> np.ndarray:
    return rolling_mean(np.asarray(covered, float), window)


def worst_window_coverage(covered: np.ndarray, window: int = 200) -> float:
    rc = rolling_coverage(covered, window)
    return float(np.nanmin(rc)) if np.any(~np.isnan(rc)) else float("nan")


def post_shift_coverage(covered: np.ndarray, change_point: int | None, horizon: int | None = None) -> float:
    """Coverage over [change_point, change_point + horizon) (or to the end if horizon is None)."""
    if change_point is None:
        return float("nan")
    seg = np.asarray(covered, float)[change_point: None if horizon is None else change_point + horizon]
    return float(seg.mean()) if len(seg) else float("nan")


def recovery_time(covered: np.ndarray, change_point: int | None, nominal: float, window: int = 100,
                  tol: float = 0.05, sustain: int = 50) -> float:
    """Steps after the change point until the trailing ``window``-coverage is within ``tol``
    of nominal for ``sustain`` consecutive steps.

    The rolling window is computed on post-change observations only (so pre-change
    coverage cannot make a method look recovered).  Returns the step (relative to the
    change point) at which the sustained run *starts*, measured at the window end.
    Returns NaN if it never recovers before the end of the stream.

    Note: a method that is never disturbed (e.g. one that is conservative)
    still needs ``window`` steps to produce a first post-change estimate, so the
    minimum possible value is ``window - 1``.
    """
    if change_point is None:
        return float("nan")
    post = np.asarray(covered, float)[change_point:]
    rc = rolling_mean(post, window)
    ok = np.abs(rc - nominal) <= tol
    run = 0
    for i, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run >= sustain:
            return float(i - sustain + 1)
    return float("nan")


def pre_recovery_coverage(covered: np.ndarray, change_point: int | None, rec_time: float, window: int) -> float:
    """Coverage between the change point and the moment of recovery (Section 19 'post-shift coverage')."""
    if change_point is None:
        return float("nan")
    end = len(covered) if np.isnan(rec_time) else change_point + int(rec_time) + window
    seg = np.asarray(covered, float)[change_point:end]
    return float(seg.mean()) if len(seg) else float("nan")


def detection_delay(detections: np.ndarray, change_point: int | None) -> tuple[float, int, int]:
    """(delay to first alarm at/after change point, #false alarms before it, #alarms after it).

    ``detections`` is a boolean array of alarm flags per step.
    """
    idx = np.flatnonzero(np.asarray(detections, bool))
    if change_point is None:
        return float("nan"), int(len(idx)), 0
    before = idx[idx < change_point]
    after = idx[idx >= change_point]
    delay = float(after[0] - change_point) if len(after) else float("nan")
    return delay, int(len(before)), int(len(after))
