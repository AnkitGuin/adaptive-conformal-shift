"""Controlled, deterministic distribution-shift generators.

All shifts are applied to the **test stream only** (the model and the
calibration scores are always computed on clean data), so the shift is by
construction something that happens *after* calibration.

Every function

* takes the test DataFrame and returns a *new* DataFrame (the input is never
  mutated – see ``tests/test_shifts.py``);
* takes an explicit ``numpy.random.Generator`` (or needs no randomness);
* returns a metadata dict recording every parameter, the resolved change
  point (as a position in the returned stream) and any per-row flags.

Interpretation of each scenario (important for the report)
-----------------------------------------------------------
gradual_covariate / abrupt_covariate
    ``x'_t = x_t + delta(t)`` on selected numeric features while ``y`` is kept.
    Because the target is left unchanged, the *observed* covariates drift away
    from the covariates that generated ``y``: this models sensor drift /
    mis-calibration.  P(X) changes AND the relationship between the observed
    X' and Y changes.  It is therefore not a "pure" covariate shift.
covariate_resample
    A pure covariate shift: after the change point, rows are sub-sampled with a
    keep-probability that depends only on X.  P(Y | X) is untouched, P(X)
    changes.
noise
    ``y' = y + eps_t``, ``eps_t ~ N(0, sigma_t^2)`` with ``sigma_t`` jumping
    (or ramping) from 0 to ``magnitude * scale`` after the change point.
    Aleatoric noise increases; E[Y | X] is unchanged.
conditional
    ``y'_t = c * y_t`` for a controlled subset of post-change rows.  The
    conditional distribution P(Y | X) itself changes, in a subgroup-specific
    way, which a marginal method cannot fully repair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ShiftResult:
    df: pd.DataFrame
    metadata: dict[str, Any]
    affected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # per-row flag


def resolve_change_point(n: int, frac: float, jitter: float, rng: np.random.Generator) -> int:
    """Change point = round(n * (frac + U(-jitter, jitter))), clipped to [1, n-1]."""
    f = frac + (rng.uniform(-jitter, jitter) if jitter > 0 else 0.0)
    return int(np.clip(round(n * f), 1, n - 1))


def _ramp(n: int, start: int, end: int) -> np.ndarray:
    """0 before ``start``, linear 0->1 between ``start`` and ``end``, 1 afterwards."""
    t = np.arange(n, dtype=float)
    if end <= start:
        return (t >= start).astype(float)
    return np.clip((t - start) / (end - start), 0.0, 1.0)


def apply_no_shift(df: pd.DataFrame, rng: np.random.Generator | None = None, **_: Any) -> ShiftResult:
    return ShiftResult(df.copy(), {"type": "none", "change_point": None}, np.zeros(len(df), dtype=bool))


def apply_gradual_covariate_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    features: list[str],
    feature_scales: dict[str, float],
    magnitude: float,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    clip: dict[str, list[float]] | None = None,
    **_: Any,
) -> ShiftResult:
    """x'_t = x_t + magnitude * scale_j * ramp(t); ramp rises linearly from start_frac to end_frac.

    ``feature_scales`` must be computed on the TRAINING split (e.g. std), so the
    magnitude is expressed in training-standard-deviation units.
    """
    out = df.copy()
    n = len(out)
    start, end = int(round(n * start_frac)), int(round(n * end_frac))
    ramp = _ramp(n, start, end)
    deltas = {}
    for col in features:
        delta = magnitude * float(feature_scales[col]) * ramp
        vals = out[col].to_numpy(dtype=float) + delta
        if clip and col in clip:
            vals = np.clip(vals, *clip[col])
        out[col] = vals
        deltas[col] = float(magnitude * feature_scales[col])
    meta = {"type": "gradual_covariate", "features": list(features), "magnitude_std_units": magnitude,
            "max_delta_raw_units": deltas, "start": start, "end": end,
            # For a gradual shift the "change point" used for delay/recovery metrics is the ramp start.
            "change_point": start if start > 0 else None, "ramp_start": start, "ramp_end": end}
    return ShiftResult(out, meta, ramp > 0)


def apply_abrupt_covariate_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    features: list[str],
    feature_scales: dict[str, float],
    magnitude: float,
    change_frac: float = 0.3,
    change_jitter: float = 0.0,
    clip: dict[str, list[float]] | None = None,
    **_: Any,
) -> ShiftResult:
    """x'_t = x_t + magnitude * scale_j for t >= change point (distribution A -> B)."""
    out = df.copy()
    n = len(out)
    cp = resolve_change_point(n, change_frac, change_jitter, rng)
    mask = np.arange(n) >= cp
    deltas = {}
    for col in features:
        vals = out[col].to_numpy(dtype=float)
        vals = vals + magnitude * float(feature_scales[col]) * mask
        if clip and col in clip:
            vals = np.clip(vals, *clip[col])
        out[col] = vals
        deltas[col] = float(magnitude * feature_scales[col])
    meta = {"type": "abrupt_covariate", "features": list(features), "magnitude_std_units": magnitude,
            "delta_raw_units": deltas, "change_point": cp}
    return ShiftResult(out, meta, mask)


def apply_covariate_resample_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    favor_col: str,
    favor_values: list[Any],
    keep_prob_other: float,
    change_frac: float = 0.3,
    change_jitter: float = 0.0,
    **_: Any,
) -> ShiftResult:
    """Pure covariate shift by X-dependent thinning after the change point.

    Rows before the change point are kept.  After it, rows with
    ``favor_col in favor_values`` are kept, other rows are kept with
    probability ``keep_prob_other``.  Chronological order is preserved.
    P(Y|X) is unchanged because selection depends on X only.
    """
    n = len(df)
    cp = resolve_change_point(n, change_frac, change_jitter, rng)
    post = np.arange(n) >= cp
    favored = df[favor_col].isin(favor_values).to_numpy()
    u = rng.uniform(size=n)
    keep = ~post | favored | (u < keep_prob_other)
    out = df.loc[keep].copy().reset_index(drop=True)
    meta = {"type": "covariate_resample", "favor_col": favor_col, "favor_values": list(favor_values),
            "keep_prob_other": keep_prob_other, "change_point": cp,  # rows before cp are all kept
            "n_before": n, "n_after": int(keep.sum()),
            "favored_share_pre": float(favored[~post].mean()), "favored_share_post": float(favored[post & keep].mean()) if (post & keep).any() else None}
    return ShiftResult(out, meta, (np.arange(len(out)) >= cp))


def apply_noise_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target: str,
    magnitude: float,
    scale: float,
    change_frac: float = 0.3,
    change_jitter: float = 0.0,
    ramp_length: int = 0,
    **_: Any,
) -> ShiftResult:
    """y' = y + eps, eps ~ N(0, sigma_t^2); sigma_t = 0 before cp and magnitude*scale after.

    ``scale`` should be computed on the training split (we use the training
    target standard deviation).  With ``ramp_length > 0`` sigma increases
    linearly over that many steps.  Values are NOT clipped at zero (documented
    limitation: synthetic counts may become negative).
    """
    out = df.copy()
    n = len(out)
    cp = resolve_change_point(n, change_frac, change_jitter, rng)
    sigma = magnitude * scale * _ramp(n, cp, cp + ramp_length)
    eps = rng.normal(0.0, 1.0, size=n) * sigma
    out[target] = out[target].to_numpy(dtype=float) + eps
    meta = {"type": "noise", "magnitude_rel": magnitude, "scale": scale, "sigma_post": magnitude * scale,
            "change_point": cp, "ramp_length": ramp_length}
    return ShiftResult(out, meta, sigma > 0)


def apply_conditional_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target: str,
    factor: float,
    change_frac: float = 0.3,
    change_jitter: float = 0.0,
    subset: dict[str, list[Any]] | None = None,
    subset_fraction: float = 1.0,
    **_: Any,
) -> ShiftResult:
    """y'_t = factor * y_t for post-change rows in a controlled subset.

    ``subset`` is a dict {column: allowed values}; a row is eligible if it
    satisfies all conditions (empty/None -> all rows).  Each eligible row is
    affected with probability ``subset_fraction``.
    """
    out = df.copy()
    n = len(out)
    cp = resolve_change_point(n, change_frac, change_jitter, rng)
    eligible = np.ones(n, dtype=bool)
    for col, vals in (subset or {}).items():
        eligible &= out[col].isin(vals).to_numpy()
    u = rng.uniform(size=n)
    affected = (np.arange(n) >= cp) & eligible & (u < subset_fraction)
    y = out[target].to_numpy(dtype=float)
    out[target] = np.where(affected, factor * y, y)
    meta = {"type": "conditional", "factor": factor, "subset": subset or {}, "subset_fraction": subset_fraction,
            "change_point": cp, "n_affected": int(affected.sum()),
            "affected_share_post": float(affected[cp:].mean())}
    return ShiftResult(out, meta, affected)


SHIFT_FUNCTIONS = {
    "none": apply_no_shift,
    "gradual_covariate": apply_gradual_covariate_shift,
    "abrupt_covariate": apply_abrupt_covariate_shift,
    "covariate_resample": apply_covariate_resample_shift,
    "noise": apply_noise_shift,
    "conditional": apply_conditional_shift,
}


def apply_shift(df: pd.DataFrame, shift_cfg: dict, rng: np.random.Generator, context: dict) -> ShiftResult:
    """Dispatch on ``shift_cfg['type']``; ``context`` supplies training-derived scales and the target name."""
    cfg = dict(shift_cfg)
    kind = cfg.pop("type")
    fn = SHIFT_FUNCTIONS[kind]
    kwargs = {**cfg}
    if kind in ("gradual_covariate", "abrupt_covariate"):
        kwargs.setdefault("feature_scales", context["feature_scales"])
    if kind in ("noise", "conditional"):
        kwargs.setdefault("target", context["target"])
    if kind == "noise":
        kwargs.setdefault("scale", context["target_scale"])
    res = fn(df, rng, **kwargs)
    res.metadata["config"] = shift_cfg
    return res
