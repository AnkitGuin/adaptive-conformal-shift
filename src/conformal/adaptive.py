"""Method 4 - DA-ACP: Drift-Aware Adaptive Conformal Prediction (+ ACI reference baseline).

DA-ACP is a transparent combination of known ingredients, not a new theory:

    rolling buffer of recent scores          (window W = ``max_buffer``)
  + exponential recency weights               w_i = exp(-lambda_t * age_i)
    (or, for ablation B, a box kernel 1{age_i <= 2/lambda_t} with matched effective size)
  + Page-Hinkley drift detector on the score stream
  + a configurable drift response.

State at step t: buffer of past scores (with their time stamps), the last
drift event (time tau and estimated onset t0) and the detector state.

Threshold at step t (computed before y_t is seen):

    lambda_t = lambda_0                                        (no recent drift)
    lambda_t = lambda_0 + (lambda_fast - lambda_0) * r(t - tau) (after drift at tau)
        with r = 1 on [0, boost_duration) for anneal='step',
        or r(u) = max(0, 1 - u / boost_duration) for anneal='linear'.

    w_i = exp(-lambda_t * age_i) * g_i,
        g_i = pre_drift_weight  if score i was observed before the estimated onset t0
                                and is not among the ``min_keep`` newest scores,
        g_i = 1                 otherwise.

    q_t = weighted conformal quantile of the buffer with weights w_i and
          test weight 1 (see ``quantiles.weighted_conformal_quantile``).

After observing y_t:  s_t is scored, appended to the buffer (unless
``rolling_updates`` is False, see below), and the detector is updated with
z_t = s_t / scale (``scale`` = mean calibration score, or signed residual /
scale when ``drift_stream='signed'``).  If the detector fires, (tau, t0) are
updated, which changes lambda_t and g_i for all later steps.

Every mechanism is exposed and can be switched off independently:
  * ``drift_enabled=False``               -> exponentially weighted conformal (ablation A)
  * ``kernel='box'``                      -> no exponential weighting (ablation B): uniform
                                             weights over the most recent 2/lambda_t scores, so
                                             the drift response shortens a hard window instead
  * ``rolling_updates=False``             -> calibration buffer frozen; after a detected
                                             drift, it is refilled with post-onset scores
                                             (up to ``max_buffer``) and then frozen again
                                             (ablation C, "detect-then-recalibrate")
  * ``pre_drift_weight=1``                -> no (partial) reset
  * ``decay_fast=decay``                  -> no weight boost

Coverage caveat (must not be over-claimed): with data-dependent weights and
temporally dependent, shifting data, DA-ACP has NO finite-sample coverage
guarantee.  It is evaluated purely empirically.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.conformal.base import Calibrator
from src.conformal.quantiles import conformal_quantile, conformal_rank, weighted_conformal_quantile
from src.drift.page_hinkley import PageHinkley


class DAACPCalibrator(Calibrator):
    name = "da_acp"

    def __init__(
        self,
        alpha: float,
        kernel: str = "exponential",
        decay: float = 0.002,
        decay_fast: float = 0.02,
        boost_duration: int = 200,
        anneal: str = "linear",
        pre_drift_weight: float = 1.0,
        min_keep: int = 50,
        max_buffer: int | None = 3000,
        rolling_updates: bool = True,
        drift_enabled: bool = True,
        drift_stream: str = "abs",
        ph_delta: float = 0.1,
        ph_threshold: float = 50.0,
        ph_min_instances: int = 30,
        ph_direction: str = "increase",
        **kw,
    ) -> None:
        super().__init__(alpha, **kw)
        if kernel not in ("exponential", "box"):
            raise ValueError("kernel must be 'exponential' or 'box'")
        self.kernel = kernel
        if anneal not in ("step", "linear"):
            raise ValueError("anneal must be 'step' or 'linear'")
        if drift_stream not in ("abs", "signed"):
            raise ValueError("drift_stream must be 'abs' or 'signed'")
        if not 0.0 <= pre_drift_weight <= 1.0:
            raise ValueError("pre_drift_weight must be in [0, 1]")
        self.decay, self.decay_fast = float(decay), float(decay_fast)
        self.boost_duration, self.anneal = int(boost_duration), anneal
        self.pre_drift_weight, self.min_keep = float(pre_drift_weight), int(min_keep)
        self.max_buffer = None if max_buffer in (None, 0) else int(max_buffer)
        self.rolling_updates, self.drift_enabled, self.drift_stream = bool(rolling_updates), bool(drift_enabled), drift_stream
        self.detector = PageHinkley(ph_delta, ph_threshold, ph_min_instances, ph_direction) if drift_enabled else None
        self.scores: deque[float] = deque(maxlen=self.max_buffer)
        self.times: deque[int] = deque(maxlen=self.max_buffer)
        self.t = 0                 # index of the next test step
        self.last_drift: int | None = None
        self.onset: int | None = None
        self.refill_remaining = 0  # used only when rolling_updates=False
        self.events: list = []
        self._last_event = False
        self._recent: deque = deque(maxlen=max(self.max_buffer or 5000, self.min_keep))

    # ----------------------------------------------------------------- setup
    def fit(self, cal_scores: np.ndarray) -> "DAACPCalibrator":
        cal_scores = np.asarray(cal_scores, float)
        self._register(cal_scores)
        keep = cal_scores if self.max_buffer is None else cal_scores[-self.max_buffer:]
        n = len(keep)
        self.scores.clear(); self.times.clear()
        self.scores.extend(keep)
        self.times.extend(range(-n, 0))  # calibration scores have negative time stamps
        # Detector scale: mean absolute calibration score, so that ph_delta / ph_threshold are
        # expressed in units of a typical calibration residual (dataset-unit free).
        self.scale = float(np.mean(np.abs(cal_scores))) or 1.0
        return self

    # ----------------------------------------------------------------- threshold
    def current_decay(self) -> float:
        if self.last_drift is None:
            return self.decay
        u = self.t - self.last_drift
        if self.anneal == "step":
            r = 1.0 if u < self.boost_duration else 0.0
        else:
            r = max(0.0, 1.0 - u / self.boost_duration) if self.boost_duration > 0 else 0.0
        return self.decay + (self.decay_fast - self.decay) * r

    def current_weights(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.fromiter(self.scores, float, len(self.scores))
        ts = np.fromiter(self.times, float, len(self.times))
        ages = self.t - ts  # newest score has age 1, test point has age 0
        lam = self.current_decay()
        if self.kernel == "exponential":
            w = np.exp(-lam * ages)
        else:  # box kernel with the same effective sample size as exp(-lam*age): W = 2 / lam
            horizon = self.box_horizon(lam)
            w = (ages <= horizon).astype(float)
        if self.onset is not None and self.pre_drift_weight < 1.0:
            pre = ts < self.onset
            if self.min_keep > 0:
                pre[-self.min_keep:] = False
            w = np.where(pre, w * self.pre_drift_weight, w)
        return s, w

    def box_horizon(self, lam: float) -> float:
        """Window length of the box kernel matched to exp(-lam*age).

        The Kish effective sample size of exponential weights is
        (sum w)^2 / sum w^2 ~= 2 / lam for small lam, and a box of length W has
        effective size W, so W = 2 / lam (capped by the buffer length).
        lam = 0 means the whole buffer.
        """
        cap = float(self.max_buffer or len(self.scores))
        return cap if lam <= 0 else min(cap, 2.0 / lam)

    def _raw_threshold(self) -> float:
        s, w = self.current_weights()
        return weighted_conformal_quantile(s, w, self.alpha, test_weight=1.0)

    # ----------------------------------------------------------------- update
    def _update(self, score: float, covered: bool) -> None:
        t = self.t
        self._recent.append((score, t))  # history of observed test scores (for ablation C)
        if self.rolling_updates or self.refill_remaining > 0:
            self.scores.append(score); self.times.append(t)
            if not self.rolling_updates:
                self.refill_remaining -= 1
        self._last_event = False
        if self.detector is not None:
            if self.drift_stream == "signed":
                if self._signed is None:
                    raise ValueError("drift_stream='signed' requires signed residuals in update()")
                z = self._signed / self.scale
            else:
                z = score / self.scale
            ev = self.detector.update(z)
            if ev is not None:
                self._last_event = True
                self.events.append(ev)
                # alarm raised after observing y_t -> takes effect from step t+1 onwards
                self.last_drift, self.onset = t + 1, ev.onset
                if not self.rolling_updates:
                    self._refresh_frozen_buffer(ev.onset)
        self.t += 1

    def _refresh_frozen_buffer(self, onset: int) -> None:
        """Ablation C: replace the frozen buffer by post-onset scores, then refill up to max_buffer and freeze."""
        post = [(s, tt) for s, tt in self._recent if tt >= onset]
        if len(post) < self.min_keep:
            post = list(self._recent)[-self.min_keep:]
        self.scores.clear(); self.times.clear()
        for s, tt in post:
            self.scores.append(s); self.times.append(tt)
        cap = self.max_buffer or 10**9
        self.refill_remaining = max(0, cap - len(self.scores))

    def diagnostics(self) -> dict:
        return {
            "drift_detected": self._last_event,
            "drift_score": self.detector.stat if self.detector is not None else float("nan"),
            "decay_t": self.current_decay(),
        }


class ACICalibrator(Calibrator):
    """Adaptive Conformal Inference (Gibbs & Candes, 2021) - reference online baseline.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1{y_t not in C_t},
    q_t = finite-sample conformal quantile at level alpha_t of the last W scores.
    alpha_t <= 0 gives q_t = +inf (handled by the base-class fallback);
    alpha_t >= 1 gives the smallest score.  Gibbs & Candes prove a long-run
    miscoverage statement for the ideal recursion with the appropriate
    infinite endpoints.  The repository's default ``max_score`` fallback
    replaces an infinite threshold by a finite one, so this capped
    implementation is an empirical ACI reference and does not inherit that
    theorem literally (it is still an average-over-time, not a per-step,
    target).
    """

    name = "aci"

    def __init__(self, alpha: float, gamma: float = 0.005, window: int = 1000, **kw) -> None:
        super().__init__(alpha, **kw)
        self.gamma, self.window = float(gamma), int(window)
        self.alpha_t = float(alpha)
        self.buffer: deque[float] = deque(maxlen=self.window)

    def fit(self, cal_scores: np.ndarray) -> "ACICalibrator":
        cal_scores = np.asarray(cal_scores, float)
        self._register(cal_scores)
        self.buffer.clear(); self.buffer.extend(cal_scores[-self.window:])
        return self

    def _raw_threshold(self) -> float:
        s = np.fromiter(self.buffer, float, len(self.buffer))
        if self.alpha_t <= 0:
            return math.inf
        if self.alpha_t >= 1:
            return float(np.min(s))
        k = conformal_rank(len(s), self.alpha_t)
        if k <= 0:
            return float(np.min(s))
        return conformal_quantile(s, self.alpha_t)

    def _update(self, score: float, covered: bool) -> None:
        self.alpha_t += self.gamma * (self.alpha - (0.0 if covered else 1.0))
        self.buffer.append(score)

    def diagnostics(self) -> dict:
        return {"alpha_t": self.alpha_t}
