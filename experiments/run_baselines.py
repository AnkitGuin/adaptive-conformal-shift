"""Baselines: point-model comparison, no-shift conformal behaviour and cross-model generalisation.

Produces:
  * point-model performance of Ridge / Random Forest / Gradient Boosting / HistGB on
    calibration and test splits (Table 2),
  * the full method comparison for every base regressor on a subset of scenarios
    (Question G: do conclusions depend on the base model?),
  * the time-identifying-feature experiment (feature set 'no_time_id', Section 7).

    python -m experiments.run_baselines [--quick]
"""
from __future__ import annotations

import numpy as np

from experiments.common import base_parser, collect, json_safe_meta, run_jobs, save_results, setup
from src.evaluation import metrics as M
from src.pipeline import clear_cache, fit_base, get_dataframe, interval_methods, run_scenario


def point_rows(cfg: dict, model: str, seed: int, feature_set: str, split_kind: str) -> list[dict]:
    fit = fit_base(cfg, "test", split_kind, feature_set, model, seed, cfg["conformal"]["alpha"], False)
    df = get_dataframe(cfg)
    tgt = cfg["data"]["target"]
    cols = fit.spec.raw_columns
    out = []
    for part, idx in [("train", fit.split.train), ("calibration", fit.split.calibration), ("test", fit.split.test)]:
        y = df[tgt].to_numpy(float)[idx]
        yp = fit.model.predict(df.iloc[idx][cols])
        out.append({"table": "point", "model": model, "seed": seed, "feature_set": feature_set, "split_kind": split_kind,
                    "part": part, "mae": M.mae(y, yp), "rmse": M.rmse(y, yp), "r2": M.r2(y, yp),
                    "fit_seconds": fit.fit_seconds})
    return out


def job(cfg: dict, j: tuple):
    """One (model, seed): point metrics + conformal methods. Fits are used then freed (RF ~80 MB each)."""
    model, seed = j
    g = cfg["experiments"]["generalisation"]
    methods = interval_methods(cfg, g["methods"])
    rows, meta = [], {}
    for fs in cfg["feature_sets"]:
        for sk in ("chronological", "iid"):
            rows.extend(point_rows(cfg, model, seed, fs, sk))
            for sc in g["scenarios"]:
                sc_kind = cfg["shifts"]["scenarios"][sc].get("split", "chronological")
                if sc_kind != sk or (fs != cfg["primary_feature_set"] and sc not in ("iid_control", "natural")):
                    continue
                r, _, m = run_scenario(cfg, sc, seed, methods, model_name=model, feature_set=fs, protocol="test",
                                       tags={"table": "conformal"})
                rows.extend(r)
                meta[f"{model}|{fs}|{sc}|{seed}"] = json_safe_meta(m)["shift"]
            clear_cache()
    return rows, meta, [], {}


def main(argv=None) -> None:
    args = base_parser(__doc__).parse_args(argv)
    cfg, seeds, out_dir, log = setup(args, "baselines")
    models = cfg["experiments"]["generalisation"]["models"]
    jobs = [(m, s) for m in models for s in seeds]
    rows, meta, _, _ = collect(run_jobs(cfg, jobs, job, log))
    save_results(out_dir, cfg, rows, meta, extra={"seeds": seeds, "models": models})


if __name__ == "__main__":
    main()
