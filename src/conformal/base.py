"""Shared conformal machinery: nonconformity scores, intervals and the online protocol.

Design
------
Every method in this project is split into two independent pieces:

1. a **score function** that maps (x, y) to a nonconformity score and maps a
   score threshold q back to an interval
   (absolute residual -> symmetric interval; CQR score -> asymmetric interval);
2. a **calibrator** that, given the history of past scores, returns the
   threshold q_t to be used at time t.

This avoids duplicating conformal logic: static, rolling, weighted, ACI and
DA-ACP differ *only* in how they turn past scores into q_t.

The online protocol (``run_online``) enforces the causal ordering required by
Section 12 of the project specification:

    for t = 1..T:
        q_t      <- calibrator.threshold()          # uses scores s_1..s_{t-1} only
        interval <- score_fn.interval(x_t, q_t)      # reported interval
        (observe y_t)  s_t <- score(x_t, y_t)
        calibrator.update(s_t, covered_t)            # only now is y_t used

``tests/test_conformal.py::test_online_protocol_is_causal`` verifies that
changing future scores never changes past thresholds.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- scores
def absolute_residual_scores(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """s_i = |y_i - f_hat(x_i)|."""
    return np.abs(np.asarray(y, float) - np.asarray(y_pred, float))


def cqr_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """CQR score (Romano et al., 2019): s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)).

    Negative when y lies strictly inside the quantile band.
    """
    y = np.asarray(y, float)
    return np.maximum(np.asarray(q_lo, float) - y, y - np.asarray(q_hi, float))


def build_intervals(lo_center: np.ndarray, hi_center: np.ndarray, q: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """[lo_center - q, hi_center + q].

    Absolute residual: lo_center = hi_center = f_hat(x).
    CQR: lo_center = q_lo(x), hi_center = q_hi(x).
    For CQR a negative q can make lower > upper when the band is very narrow;
    we then collapse the *display bounds* to their midpoint so plotting and
    interval-width code never sees inverted bounds.  The exact CQR set can be
    empty in that case.  ``run_online`` therefore evaluates coverage from the
    score rule ``s(x, y) <= q`` rather than from these display bounds.
    """
    lo = np.asarray(lo_center, float) - q
    hi = np.asarray(hi_center, float) + q
    bad = lo > hi
    if np.any(bad):
        mid = 0.5 * (lo + hi)
        lo = np.where(bad, mid, lo)
        hi = np.where(bad, mid, hi)
    return lo, hi


# ----------------------------------------------------------------------------- calibrators
class Calibrator(ABC):
    """Maps a history of scores to a threshold q_t. Subclasses implement ``_raw_threshold``.

    Infinite thresholds.  With too little (effective) calibration mass the
    finite-sample conformal quantile is +inf, i.e. the prediction set is the
    whole real line.  Infinite widths make averages meaningless, so by default
    we replace +inf with the largest score currently available to the
    calibrator (``inf_fallback='max_score'``) and count how often this
    happened (``n_infinite``).  This is a documented deviation from the exact
    conformal set and is reported in every results table.
    """

    name: str = "base"

    def __init__(self, alpha: float, inf_fallback: str = "max_score") -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = float(alpha)
        self.inf_fallback = inf_fallback
        self.n_infinite = 0
        self._max_seen = -math.inf
        self._signed: float | None = None

    # -- to implement
    @abstractmethod
    def fit(self, cal_scores: np.ndarray) -> "Calibrator":
        ...

    @abstractmethod
    def _raw_threshold(self) -> float:
        ...

    def _update(self, score: float, covered: bool) -> None:  # static methods ignore feedback
        return None

    # -- public API
    def _register(self, scores: np.ndarray) -> None:
        if len(scores):
            self._max_seen = max(self._max_seen, float(np.max(scores)))

    def threshold(self) -> float:
        q = self._raw_threshold()
        if math.isinf(q) and q > 0:
            self.n_infinite += 1
            if self.inf_fallback == "max_score":
                return self._max_seen
        return q

    def update(self, score: float, covered: bool, signed: float | None = None) -> None:
        """Feed back the realised score (and optionally the signed residual y - f_hat(x))."""
        self._max_seen = max(self._max_seen, float(score))
        self._signed = None if signed is None else float(signed)
        self._update(float(score), bool(covered))

    def diagnostics(self) -> dict:
        """Per-step diagnostics recorded by ``run_online`` (override to add drift info)."""
        return {}


@dataclass
class OnlineResult:
    q: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    covered: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    n_infinite: int = 0


def run_online(
    calibrator: Calibrator,
    test_scores: np.ndarray,
    lo_center: np.ndarray,
    hi_center: np.ndarray,
    y_true: np.ndarray,
    signed_residuals: np.ndarray | None = None,
) -> OnlineResult:
    """Run the causal predict-then-observe protocol over a test stream.

    ``test_scores[t]`` is only *read* after the prediction set for step t has
    been fixed (see module docstring).  Membership is evaluated using the
    method's defining rule, ``score_t <= q_t``, rather than by reverse-
    engineering membership from the rendered bounds.  These are equivalent
    for absolute-residual intervals, but not always for CQR when a negative
    correction would make the nominal bounds cross.  ``y_true`` is retained in
    the API for shape checking and backwards compatibility.
    """
    test_scores = np.asarray(test_scores, dtype=float).ravel()
    lo_center = np.asarray(lo_center, dtype=float).ravel()
    hi_center = np.asarray(hi_center, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    T = len(test_scores)
    if not (len(lo_center) == len(hi_center) == len(y_true) == T):
        raise ValueError("test_scores, interval centers, and y_true must have the same length")
    if (np.any(~np.isfinite(test_scores)) or np.any(~np.isfinite(y_true))
            or np.any(~np.isfinite(lo_center)) or np.any(~np.isfinite(hi_center))):
        raise ValueError("test scores, y_true, and interval centers must be finite")
    q = np.empty(T)
    lower = np.empty(T)
    upper = np.empty(T)
    covered = np.empty(T, dtype=bool)
    diag: dict[str, list] = {}
    for t in range(T):
        # (1)-(2) interval from information strictly before t
        q_t = calibrator.threshold()
        lo_t, hi_t = build_intervals(lo_center[t:t + 1], hi_center[t:t + 1], q_t)
        q[t], lower[t], upper[t] = q_t, lo_t[0], hi_t[0]
        for k, v in calibrator.diagnostics().items():
            diag.setdefault(k, []).append(v)
        # (3)-(6) observe y_t, score, then update the calibrator / drift detector
        # This is the actual prediction-set definition.  In particular, for
        # CQR a negative q can describe an empty set even though
        # build_intervals() keeps its display bounds ordered for downstream
        # plotting/metrics.  Using the score avoids silently changing the
        # coverage event (and ACI's feedback) in that edge case.
        covered[t] = bool(test_scores[t] <= q_t)
        calibrator.update(test_scores[t], covered[t],
                          None if signed_residuals is None else signed_residuals[t])
    return OnlineResult(q, lower, upper, covered, {k: np.asarray(v) for k, v in diag.items()}, calibrator.n_infinite)
