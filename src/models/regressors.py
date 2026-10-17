"""Classical point predictors and quantile regressors (CPU-only, scikit-learn).

Every model is an sklearn ``Pipeline(preprocessor -> estimator)`` so that the
preprocessing is fitted on exactly the data the estimator is fitted on (the
training split), which rules out preprocessing leakage by construction.

Hyperparameters are fixed a priori in ``configs/default.yaml`` (sensible
defaults, no test-set tuning); the research question concerns the conformal
layer, not leaderboard performance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.data.preprocess import FeatureSpec, build_preprocessor

MODEL_REGISTRY = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
    "hgb": HistGradientBoostingRegressor,
}
_STOCHASTIC = {"random_forest", "gbr", "hgb"}


def build_model(name: str, params: dict[str, Any] | None, spec: FeatureSpec, seed: int) -> Pipeline:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {sorted(MODEL_REGISTRY)}")
    params = dict(params or {})
    if name in _STOCHASTIC:
        params["random_state"] = int(seed)
    return Pipeline([("pre", build_preprocessor(spec)), ("model", MODEL_REGISTRY[name](**params))])


def build_quantile_model(quantile: float, params: dict[str, Any] | None, spec: FeatureSpec, seed: int,
                         kind: str = "hgb") -> Pipeline:
    """Quantile regressor used by CQR (pinball loss)."""
    params = dict(params or {})
    if kind == "hgb":
        est = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=int(seed), **params)
    elif kind == "gbr":
        est = GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=int(seed), **params)
    else:
        raise KeyError(kind)
    return Pipeline([("pre", build_preprocessor(spec)), ("model", est)])


def fit_predict(model: Pipeline, X_train, y_train, *X_eval) -> list[np.ndarray]:
    model.fit(X_train, y_train)
    return [model.predict(X) for X in X_eval]
