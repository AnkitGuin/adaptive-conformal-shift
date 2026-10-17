"""Hyperparameter selection on the DEVELOPMENT protocol only (never the test split).

The development protocol is a chronological train | calibration | test split
nested inside the outer training portion (first 60% of the data), with the
same synthetic shift scenarios.  Each adaptive method gets its own small grid
(fairness: baselines are tuned just like DA-ACP).

Selection criterion: the *relative* interval score, i.e. the method's mean
interval score divided by that of static split conformal on the same
(scenario, seed), averaged over dev scenarios and dev seeds.  Normalising by
the static baseline stops high-variance scenarios from dominating.  The
interval score is a proper scoring rule for central (1 - alpha) intervals, so
it rewards both sharpness and coverage.

Output: configs/selected_hparams.yaml (frozen and then used by every
test-protocol experiment) + results/tuning/tuning_results.csv.

    python -m experiments.run_tuning
"""
from __future__ import annotations

import itertools

import pandas as pd
import yaml

from experiments.common import base_parser, run_jobs, setup
from src.pipeline import interval_methods, run_scenario
from src.utils.config import resolve_path


def grid_methods(cfg: dict) -> dict[str, dict]:
    grids = cfg["experiments"]["tuning"]["grids"]
    base = interval_methods(cfg, ["static", "rolling", "weighted", "aci", "da_acp"])
    out = {"static": base["static"]}
    for mname, grid in grids.items():
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            if mname == "da_acp" and params.get("decay_fast", 1) < params.get("decay", 0):
                continue
            label = mname + "|" + ",".join(f"{k}={v}" for k, v in params.items())
            out[label] = {**base[mname], **params}
    return out


def job(cfg: dict, j: tuple) -> list[dict]:
    scenario, seed = j
    rows, _, _ = run_scenario(cfg, scenario, seed, grid_methods(cfg), protocol="dev")
    return rows


def main(argv=None) -> None:
    args = base_parser(__doc__).parse_args(argv)
    cfg, _, out_dir, log = setup(args, "tuning")
    tcfg = cfg["experiments"]["tuning"]
    seeds = args.seeds or tcfg["seeds"]
    log.info("tuning (dev protocol) seeds=%s scenarios=%s", seeds, tcfg["scenarios"])
    jobs = [(s, seed) for seed in seeds for s in tcfg["scenarios"]]
    log.info("%d candidate methods x %d jobs", len(grid_methods(cfg)), len(jobs))
    rows = [r for rs in run_jobs(cfg, jobs, job, log) for r in rs]
    df = pd.DataFrame(rows)
    static = df[df.method == "static"].set_index(["scenario", "seed"])["interval_score"]
    df["rel_interval_score"] = df.apply(lambda r: r["interval_score"] / static.loc[(r["scenario"], r["seed"])], axis=1)
    df["family"] = df.method.str.split("|").str[0]
    agg = (df.groupby(["family", "method"])
             .agg(rel_interval_score=("rel_interval_score", "mean"), coverage=("coverage", "mean"),
                  worst_window_coverage=("worst_window_coverage", "mean"), width=("width", "mean"))
             .reset_index().sort_values(["family", "rel_interval_score"]))
    tdir = resolve_path(args.output or cfg["experiment"]["output_dir"]) / "tuning"
    tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "tuning_runs.csv", index=False)
    agg.to_csv(tdir / "tuning_results.csv", index=False)

    selected = {}
    for fam, g in agg[agg.family != "static"].groupby("family"):
        best = g.iloc[0]["method"]
        params = dict(kv.split("=") for kv in best.split("|")[1].split(","))
        selected[fam] = {k: yaml.safe_load(v) for k, v in params.items()}
        log.info("selected %s: %s (dev rel. IS %.4f)", fam, selected[fam], g.iloc[0]["rel_interval_score"])
    doc = {"_comment": "Selected on the development protocol by experiments/run_tuning.py; frozen for test runs.",
           "criterion": "mean relative interval score (method / static) over dev scenarios and seeds",
           "dev_seeds": list(seeds), "dev_scenarios": tcfg["scenarios"], "methods": selected}
    with open(resolve_path(cfg["experiment"]["hparams_file"]), "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(agg.groupby("family").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
