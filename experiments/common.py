"""Shared CLI + job runner + result storage for all experiment scripts.

Every experiment writes to ``<output>/raw/<experiment_name>/``:

    config.json          fully resolved configuration (incl. tuned hyperparameters)
    metrics.csv          one row per (scenario, seed, method, variant ...)
    metadata.json        shift metadata (change points, parameters) and drift events
    predictions.csv.gz   raw per-step predictions (see ``save_predictions``)
    traces.npz           per-step coverage/width/alarm arrays for every seed (for figures)
    run_info.json        timing, package versions, git-free provenance
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.pipeline import load_full_config
from src.utils.config import apply_overrides, resolve_path, save_json
from src.utils.logging import get_logger


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml", help="configuration file (relative to repo root)")
    p.add_argument("--seed", type=int, default=None, help="run a single seed instead of the configured list")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit list of seeds")
    p.add_argument("--output", default=None, help="output root directory (default: experiment.output_dir)")
    p.add_argument("--n-jobs", type=int, default=None, help="parallel worker processes")
    p.add_argument("--quick", action="store_true", help="smoke-test mode: 2 seeds")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--no-hparams", action="store_true", help="ignore configs/selected_hparams.yaml (use defaults)")
    return p


def setup(args: argparse.Namespace, name: str) -> tuple[dict, list[int], Path, object]:
    cfg = load_full_config(args.config, use_selected_hparams=not args.no_hparams)
    cfg = apply_overrides(cfg, args.set)
    seeds = cfg["experiment"]["seeds"]
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    elif args.quick:
        seeds = seeds[:2]
    if args.n_jobs is not None:
        cfg["experiment"]["n_jobs"] = args.n_jobs
    out_root = resolve_path(args.output or cfg["experiment"]["output_dir"])
    out_dir = out_root / "raw" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"acp.{name}", out_root / "logs" / f"{name}.log")
    logger.info("experiment=%s seeds=%s hparams=%s output=%s", name, seeds, cfg["hparams_source"], out_dir)
    return cfg, list(seeds), out_dir, logger


def run_jobs(cfg: dict, jobs: list, job_fn: Callable, logger) -> list:
    n_jobs = int(cfg["experiment"].get("n_jobs", 1))
    t0 = time.time()
    if n_jobs == 1 or len(jobs) == 1:
        results = [job_fn(cfg, j) for j in jobs]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(delayed(job_fn)(cfg, j) for j in jobs)
    logger.info("%d jobs finished in %.1fs", len(jobs), time.time() - t0)
    return results


def save_results(out_dir: Path, cfg: dict, rows: list[dict], metadata: dict, preds: list[pd.DataFrame] | None = None,
                 traces: dict | None = None, extra: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    save_json(cfg, out_dir / "config.json")
    save_json(metadata, out_dir / "metadata.json")
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    if traces:
        np.savez_compressed(out_dir / "traces.npz", **traces)
    import sklearn, scipy

    info = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_rows": len(df), **(extra or {})}
    save_json(info, out_dir / "run_info.json")
    return df


def collect(results: list) -> tuple[list, dict, list, dict]:
    """Merge job outputs of the form (rows, metadata, preds, traces)."""
    rows, meta, preds, traces = [], {}, [], {}
    for r, m, p, tr in results:
        rows.extend(r)
        meta.update(m)
        preds.extend(p)
        traces.update(tr)
    return rows, meta, preds, traces


def want_predictions(cfg: dict, seed: int, seeds: list[int]) -> bool:
    mode = cfg["experiment"].get("save_predictions", "first_seed")
    return mode == "all" or (mode == "first_seed" and seed == seeds[0])


def json_safe_meta(meta: dict) -> dict:
    m = dict(meta)
    m.pop("traces", None)
    return json.loads(json.dumps(m, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
