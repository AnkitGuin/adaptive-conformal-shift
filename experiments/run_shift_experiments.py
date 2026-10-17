"""Main experiment: all shift scenarios x all methods x all seeds (TEST protocol).

Answers RQ1-RQ3, RQ5, RQ6 and produces Tables 3-6 and Figures 1-6.
Hyperparameters are frozen from configs/selected_hparams.yaml (dev protocol).

    python -m experiments.run_shift_experiments [--quick] [--seeds 0 1 2]
"""
from __future__ import annotations

from experiments.common import base_parser, collect, json_safe_meta, run_jobs, save_results, setup, want_predictions
from src.pipeline import clear_cache, interval_methods, run_scenario

METHODS = ["point", "static", "rolling", "weighted", "aci", "da_acp", "cqr_static", "cqr_da_acp"]


def job(cfg: dict, j: tuple):
    seed, scenarios, save_pred = j
    rows, meta, preds, traces = [], {}, [], {}
    methods = interval_methods(cfg, METHODS)
    for sc in scenarios:
        r, p, m = run_scenario(cfg, sc, seed, methods, protocol="test", save_predictions=save_pred, save_traces=True)
        rows.extend(r)
        meta[f"{sc}|{seed}"] = json_safe_meta(m)
        if p is not None:
            preds.append(p)
        for mname, tr in m["traces"].items():
            for k, v in tr.items():
                traces[f"{sc}|{seed}|{mname}|{k}"] = v
    clear_cache()
    return rows, meta, preds, traces


def main(argv=None) -> None:
    p = base_parser(__doc__)
    p.add_argument("--scenarios", nargs="+", default=None)
    args = p.parse_args(argv)
    cfg, seeds, out_dir, log = setup(args, "shift_experiments")
    scenarios = args.scenarios or list(cfg["shifts"]["scenarios"])
    jobs = [(s, scenarios, want_predictions(cfg, s, seeds)) for s in seeds]
    rows, meta, preds, traces = collect(run_jobs(cfg, jobs, job, log))
    df = save_results(out_dir, cfg, rows, meta, preds, traces, {"seeds": seeds, "scenarios": scenarios})
    log.info("saved %d metric rows to %s", len(df), out_dir)


if __name__ == "__main__":
    main()
