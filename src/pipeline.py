"""End-to-end experiment pipeline shared by every experiment script.

One *run* = (protocol, split kind, feature set, base model, seed, scenario,
alpha, calibration size) evaluated for a set of conformal methods.

Protocol ('test' or 'dev')
    'test' : outer chronological split  train | calibration | test.
    'dev'  : a chronological train | calibration | test split nested INSIDE
             the outer training portion.  All hyperparameter choices are made
             on 'dev' runs; the outer test split is only touched by final
             evaluation runs.

Order of operations for one run (no leakage)
    1. split indices (chronological, or random for the iid control);
    2. fit preprocessing + model on the training split ONLY;
    3. compute calibration scores on the calibration split (clean data);
    4. apply the synthetic shift to a COPY of the test stream (seeded);
    5. predict on the (shifted) test covariates;
    6. run each calibrator through the causal online protocol;
    7. compute metrics.
"""
from __future__ import annotations

import copy
import functools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.conformal.adaptive import ACICalibrator, DAACPCalibrator
from src.conformal.base import Calibrator, absolute_residual_scores, cqr_scores, run_online
from src.conformal.rolling import RollingCalibrator
from src.conformal.split import StaticSplitCalibrator
from src.conformal.weighted import ExpWeightedCalibrator
from src.data.load import load_dataset
from src.data.preprocess import (FeatureSpec, Split, check_feature_spec, chronological_split, iid_split,
                                 nested_dev_split, validate_split)
from src.data.shifts import apply_shift
from src.evaluation import coverage as cov
from src.evaluation import metrics as M
from src.models.regressors import build_model
from src.utils.config import load_config, resolve_path
from src.utils.seeds import derive_seed, make_rng

INTERVAL_METHOD_KINDS = ("static", "rolling", "weighted", "aci", "da_acp")


# ============================================================================ configuration helpers
def load_full_config(path: str = "configs/default.yaml", use_selected_hparams: bool = True) -> dict:
    """Load default config + shift scenarios + experiment settings (+ frozen tuned hyperparameters)."""
    cfg = load_config(path)
    cfg.setdefault("shifts", load_config("configs/shifts.yaml"))
    cfg.setdefault("experiments", load_config("configs/experiments.yaml"))
    hp_file = resolve_path(cfg["experiment"].get("hparams_file", "configs/selected_hparams.yaml"))
    cfg["hparams_source"] = "defaults"
    if use_selected_hparams and hp_file.exists():
        hp = load_config(hp_file)
        for m, params in (hp.get("methods") or {}).items():
            cfg["methods"].setdefault(m, {}).update(params)
        cfg["hparams_source"] = str(hp_file.relative_to(resolve_path(".")))
    return cfg


def resolve_method_cfg(methods_cfg: dict, name: str) -> dict:
    """Resolve ``inherit_from`` (used e.g. by cqr_da_acp to reuse da_acp's hyperparameters)."""
    m = dict(methods_cfg[name])
    parent = m.pop("inherit_from", None)
    if parent:
        base = resolve_method_cfg(methods_cfg, parent)
        base.update({k: v for k, v in m.items()})
        m = base
    return m


def make_calibrator(method_cfg: dict, alpha: float, inf_fallback: str = "max_score") -> Calibrator:
    m = dict(method_cfg)
    kind = m.pop("kind")
    m.pop("score", None)
    m.pop("inherit_from", None)
    kw = {"inf_fallback": inf_fallback}
    if kind == "static":
        return StaticSplitCalibrator(alpha, **kw)
    if kind == "rolling":
        return RollingCalibrator(alpha, window=m["window"], **kw)
    if kind == "weighted":
        return ExpWeightedCalibrator(alpha, decay=m["decay"], max_buffer=m.get("max_buffer"), **kw)
    if kind == "aci":
        return ACICalibrator(alpha, gamma=m["gamma"], window=m["window"], **kw)
    if kind == "da_acp":
        if m.get("decay_fast") == "__decay__":
            m["decay_fast"] = m["decay"]
        return DAACPCalibrator(alpha, **m, **kw)
    raise KeyError(f"unknown method kind '{kind}'")


# ============================================================================ data + models (cached)
@functools.lru_cache(maxsize=4)
def _load_df(data_cfg_key: str) -> pd.DataFrame:
    import json

    return load_dataset(json.loads(data_cfg_key))


def get_dataframe(cfg: dict) -> pd.DataFrame:
    import json

    return _load_df(json.dumps(cfg["data"], sort_keys=True))


def feature_spec(cfg: dict, feature_set: str | None = None) -> FeatureSpec:
    fs = feature_set or cfg.get("primary_feature_set", "full")
    spec = FeatureSpec.from_config(cfg["features"], cfg["feature_sets"][fs]["drop"])
    check_feature_spec(spec, cfg["data"]["target"], cfg["data"].get("forbidden_features"))
    return spec


def make_split(cfg: dict, n: int, protocol: str, split_kind: str, seed: int, timestamps: pd.Series) -> Split:
    if split_kind == "iid":
        outer = iid_split(n, cfg["split"], make_rng(seed, "iid_split"))
    else:
        outer = chronological_split(n, cfg["split"])
    validate_split(outer, timestamps)
    if protocol == "test":
        return outer
    if protocol == "dev":
        if split_kind == "iid":
            rng = make_rng(seed, "iid_split_dev")
            tr = outer.train
            inner = iid_split(len(tr), cfg["split"]["dev"], rng)
            dev = Split(tr[inner.train], tr[inner.calibration], tr[inner.test], kind="iid")
        else:
            dev = nested_dev_split(outer.train, cfg["split"]["dev"])
        validate_split(dev, timestamps)
        # the dev split must never touch outer calibration/test rows
        assert set(np.concatenate([dev.train, dev.calibration, dev.test])) <= set(outer.train)
        return dev
    raise ValueError(protocol)


@dataclass
class Fitted:
    split: Split
    spec: FeatureSpec
    model: Any
    y_pred_cal: np.ndarray
    cqr: Any
    feature_scales: dict
    target_scale: float
    fit_seconds: float


_FIT_CACHE: dict = {}


def fit_base(cfg: dict, protocol: str, split_kind: str, feature_set: str, model_name: str, seed: int,
             alpha: float, need_cqr: bool) -> Fitted:
    """Fit the base regressor (and optionally the CQR quantile models) on the training split; cached."""
    key = (protocol, split_kind, feature_set, model_name, seed, alpha if need_cqr else None, need_cqr)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    df = get_dataframe(cfg)
    target = cfg["data"]["target"]
    split = make_split(cfg, len(df), protocol, split_kind, seed, df["timestamp"])
    spec = feature_spec(cfg, feature_set)
    cols = spec.raw_columns
    tr, ca = df.iloc[split.train], df.iloc[split.calibration]
    t0 = time.time()
    model = build_model(model_name, cfg["models"]["params"].get(model_name), spec, derive_seed(seed, "model", model_name))
    model.fit(tr[cols], tr[target].to_numpy(float))
    y_pred_cal = model.predict(ca[cols])
    cqr = None
    if need_cqr:
        # CQR bands on calibration data; test bands are computed after shifting (see run_scenario)
        from src.models.regressors import build_quantile_model

        qc = cfg["models"]["cqr"]
        lo_m = build_quantile_model(alpha / 2, qc["params"], spec, derive_seed(seed, "cqr_lo"), qc["kind"])
        hi_m = build_quantile_model(1 - alpha / 2, qc["params"], spec, derive_seed(seed, "cqr_hi"), qc["kind"])
        lo_m.fit(tr[cols], tr[target].to_numpy(float))
        hi_m.fit(tr[cols], tr[target].to_numpy(float))
        cqr = (lo_m, hi_m)
    # training-split scales used to express shift magnitudes (never test statistics)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_scales = {c: float(tr[c].std()) for c in numeric}
    fitted = Fitted(split, spec, model, y_pred_cal, cqr, feature_scales, float(tr[target].std()), time.time() - t0)
    if len(_FIT_CACHE) >= 6:  # bounded: random forests are ~80 MB each
        _FIT_CACHE.clear()
    _FIT_CACHE[key] = fitted
    return fitted


def clear_cache() -> None:
    _FIT_CACHE.clear()


def _cqr_band(cqr, X) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = cqr[0].predict(X), cqr[1].predict(X)
    return np.minimum(lo, hi), np.maximum(lo, hi)


# ============================================================================ one run
def run_scenario(
    cfg: dict,
    scenario: str,
    seed: int,
    methods: dict[str, dict],
    model_name: str | None = None,
    alpha: float | None = None,
    protocol: str = "test",
    feature_set: str | None = None,
    cal_size: int | None = None,
    shift_override: dict | None = None,
    save_predictions: bool = False,
    save_traces: bool = False,
    tags: dict | None = None,
) -> tuple[list[dict], pd.DataFrame | None, dict]:
    """Run all ``methods`` on one (scenario, seed). Returns (metric rows, predictions, metadata)."""
    alpha = float(alpha if alpha is not None else cfg["conformal"]["alpha"])
    model_name = model_name or cfg["models"]["primary"]
    feature_set = feature_set or cfg.get("primary_feature_set", "full")
    scen_cfg = copy.deepcopy(cfg["shifts"]["scenarios"][scenario])
    if shift_override:
        scen_cfg["shift"].update(shift_override)
    split_kind = scen_cfg.get("split", "chronological")
    need_cqr = any(m.get("score") == "cqr" for m in methods.values())
    fit = fit_base(cfg, protocol, split_kind, feature_set, model_name, seed, alpha, need_cqr)
    df = get_dataframe(cfg)
    target = cfg["data"]["target"]
    cols = fit.spec.raw_columns

    cal_idx = fit.split.calibration
    y_cal = df[target].to_numpy(float)[cal_idx]
    y_pred_cal = fit.y_pred_cal
    if cal_size is not None and cal_size < len(cal_idx):
        # keep the MOST RECENT cal_size calibration points (closest in time to the test period)
        y_cal, y_pred_cal, cal_idx = y_cal[-cal_size:], y_pred_cal[-cal_size:], cal_idx[-cal_size:]

    # ---- shift the test stream (copy; seeded; metadata recorded)
    test_df = df.iloc[fit.split.test].reset_index(drop=True)
    shift_rng = make_rng(seed, "shift", scenario, protocol)
    context = {"feature_scales": fit.feature_scales, "target": target, "target_scale": fit.target_scale}
    sres = apply_shift(test_df, scen_cfg["shift"], shift_rng, context)
    test_df = sres.df
    cp = sres.metadata.get("change_point")
    y_test = test_df[target].to_numpy(float)
    y_pred = fit.model.predict(test_df[cols])
    signed_test = y_test - y_pred

    abs_cal = absolute_residual_scores(y_cal, y_pred_cal)
    abs_test = absolute_residual_scores(y_test, y_pred)
    if need_cqr:
        lo_cal_b, hi_cal_b = _cqr_band(fit.cqr, df.iloc[cal_idx][cols])
        lo_te_b, hi_te_b = _cqr_band(fit.cqr, test_df[cols])
        cqr_cal, cqr_test = cqr_scores(y_cal, lo_cal_b, hi_cal_b), cqr_scores(y_test, lo_te_b, hi_te_b)

    ev = cfg["evaluation"]
    nominal = 1 - alpha
    base_row = {"scenario": scenario, "seed": seed, "model": model_name, "alpha": alpha, "protocol": protocol,
                "feature_set": feature_set, "cal_size": len(y_cal), "n_test": len(y_test),
                "change_point": cp, **(tags or {})}
    rows, preds = [], []
    meta = {"shift": sres.metadata, "drift_events": {}, "fit_seconds": fit.fit_seconds, "traces": {}}
    point_mae, point_rmse = M.mae(y_test, y_pred), M.rmse(y_test, y_pred)

    for name, mcfg_raw in methods.items():
        mcfg = dict(mcfg_raw)
        t0 = time.time()
        if mcfg["kind"] == "point":
            rows.append({**base_row, "method": name, "mae": point_mae, "rmse": point_rmse, "r2": M.r2(y_test, y_pred)})
            continue
        use_cqr = mcfg.get("score") == "cqr"
        cal_scores, test_scores = (cqr_cal, cqr_test) if use_cqr else (abs_cal, abs_test)
        lo_c, hi_c = (lo_te_b, hi_te_b) if use_cqr else (y_pred, y_pred)
        calib = make_calibrator(mcfg, alpha, cfg["conformal"].get("inf_fallback", "max_score"))
        calib.fit(cal_scores)
        res = run_online(calib, test_scores, lo_c, hi_c, y_test, signed_residuals=signed_test)
        covered = res.covered
        width = res.upper - res.lower
        drift_flags = res.diagnostics.get("drift_detected", np.zeros(len(covered), bool))
        # drift_detected[t] is recorded BEFORE the update at t (i.e. refers to an alarm raised after t-1);
        # shift by one so that flag[t] = "alarm raised after observing y_t".
        alarm_after = np.zeros(len(covered), bool)
        alarm_after[:-1] = drift_flags[1:]
        if isinstance(calib, DAACPCalibrator) and calib.events and calib.events[-1].t == len(covered) - 1:
            alarm_after[-1] = True
        delay, n_false, n_after = cov.detection_delay(alarm_after, cp)
        rec = cov.recovery_time(covered, cp, nominal, ev["rolling_window"], ev["recovery_tol"], ev["recovery_sustain"])
        row = {
            **base_row, "method": name,
            "coverage": float(covered.mean()),
            "coverage_gap": M.coverage_gap(float(covered.mean()), nominal),
            "width": float(width.mean()), "median_width": float(np.median(width)),
            "interval_score": M.mean_interval_score(y_test, res.lower, res.upper, alpha),
            "mae": point_mae, "rmse": point_rmse,
            "worst_window_coverage": cov.worst_window_coverage(covered, ev["worst_window"]),
            "pre_shift_coverage": float(covered[:cp].mean()) if cp else np.nan,
            "post_shift_coverage": cov.post_shift_coverage(covered, cp, ev["post_shift_horizon"]),
            "post_shift_width": float(width[cp:cp + ev["post_shift_horizon"]].mean()) if cp else np.nan,
            "late_coverage": float(covered[cp + ev["post_shift_horizon"]:].mean()) if cp and cp + ev["post_shift_horizon"] < len(covered) else np.nan,
            "recovery_time": rec,
            "recovered": float(not np.isnan(rec)) if cp else np.nan,
            "pre_recovery_coverage": cov.pre_recovery_coverage(covered, cp, rec, ev["rolling_window"]),
            "detection_delay": delay if isinstance(calib, DAACPCalibrator) and calib.detector is not None else np.nan,
            "false_alarms": n_false if isinstance(calib, DAACPCalibrator) and calib.detector is not None else np.nan,
            "n_alarms": int(alarm_after.sum()),
            "n_infinite": res.n_infinite,
            "seconds": time.time() - t0,
        }
        # subgroup coverage (conditional coverage diagnostics; working days vs non-working days)
        if "workingday" in test_df:
            wd = test_df["workingday"].to_numpy() == 1
            row["coverage_workingday"] = float(covered[wd].mean()) if wd.any() else np.nan
            row["coverage_nonworkingday"] = float(covered[~wd].mean()) if (~wd).any() else np.nan
        if sres.affected.any() and len(sres.affected) == len(covered):
            row["coverage_affected"] = float(covered[sres.affected].mean())
            post_unaff = (~sres.affected) & (np.arange(len(covered)) >= (cp or 0))
            row["coverage_unaffected_post"] = float(covered[post_unaff].mean()) if post_unaff.any() else np.nan
        rows.append(row)
        if isinstance(calib, DAACPCalibrator):
            meta["drift_events"][name] = [e.__dict__ for e in calib.events]
        if save_traces:
            meta["traces"][name] = {"covered": covered.astype(np.uint8), "width": width.astype(np.float32),
                                    "alarm": alarm_after.astype(np.uint8)}
        if save_predictions:
            p = pd.DataFrame({
                "t": np.arange(len(y_test)), "timestamp": test_df["timestamp"].astype(str).to_numpy(),
                "y_true": y_test, "y_pred": y_pred, "lower": res.lower, "upper": res.upper,
                "residual": signed_test, "score": test_scores, "q": res.q, "covered": covered.astype(int),
                "drift_detected": alarm_after.astype(int),
                "drift_score": res.diagnostics.get("drift_score", np.full(len(y_test), np.nan)),
                "method": name, "experiment": scenario, "seed": seed,
            })
            preds.append(p)
    pred_df = pd.concat(preds, ignore_index=True) if preds else None
    return rows, pred_df, meta


def interval_methods(cfg: dict, names: list[str] | None = None) -> dict[str, dict]:
    names = names or list(cfg["methods"])
    return {n: resolve_method_cfg(cfg["methods"], n) for n in names}
