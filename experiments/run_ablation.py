"""Ablation study of DA-ACP (Section 22): remove / vary one component at a time.

Variants (configs/experiments.yaml -> ablation.variants) are applied on top of
the frozen, dev-selected DA-ACP configuration:

  A  no drift detection                 (drift_enabled = false)
  B  no exponential weighting           (kernel = box with matched effective size)
  C  no rolling updates                 (frozen buffer, refreshed only after an alarm)
  +  no weight boost, partial reset on/off, step anneal, signed-residual detector,
     two-sided detector
  D  decay rate lambda in {small, medium, large}
  E  calibration buffer length (max_buffer)

Ablation F (shift magnitude) is part of run_sensitivity.py.

Follow-up group ``slow_base`` (results/raw/ablation_slow_base): weighted(lambda) vs
DA-ACP(lambda_0 = lambda, lambda_fast = 0.02) with/without partial reset for slow
lambda, i.e. the regime where a detector *could* add value.

    python -m experiments.run_ablation [--quick]
"""
from __future__ import annotations

from experiments.common import base_parser, collect, json_safe_meta, run_jobs, save_results, setup
from src.pipeline import clear_cache, interval_methods, run_scenario


def variants(cfg: dict) -> dict[str, dict]:
    ab = cfg["experiments"]["ablation"]
    base = interval_methods(cfg, ["da_acp"])["da_acp"]
    out = {"static": interval_methods(cfg, ["static"])["static"]}
    for name, over in ab["variants"].items():
        out[name] = {**base, **over}
    for lvl, lam in ab["decay_levels"].items():
        # D: vary lambda_0; the post-drift lambda keeps the same ratio decay_fast / decay
        ratio = base["decay_fast"] / base["decay"] if base["decay"] > 0 else 1.0
        out[f"D_decay_{lvl}"] = {**base, "decay": lam, "decay_fast": lam * ratio}
    for w in ab["windows"]:
        out[f"E_buffer_{w}"] = {**base, "max_buffer": w}
    return out


def slow_base_variants(cfg: dict) -> dict[str, dict]:
    """Follow-up ablation: value of the detector when the baseline decay is slow."""
    sb = cfg["experiments"]["ablation"]["slow_base"]
    base = interval_methods(cfg, ["da_acp"])["da_acp"]
    out = {}
    for lam in sb["decays"]:
        out[f"weighted_l{lam}"] = {**base, "decay": lam, "decay_fast": lam, "drift_enabled": False}
        out[f"da_acp_l{lam}"] = {**base, "decay": lam, "decay_fast": sb["decay_fast"], "pre_drift_weight": 1.0}
        out[f"da_acp_reset_l{lam}"] = {**base, "decay": lam, "decay_fast": sb["decay_fast"],
                                       "pre_drift_weight": sb["pre_drift_weight_reset"]}
    return out


def job(cfg: dict, j: tuple):
    seed, scenarios, group = j
    rows, meta = [], {}
    vs = variants(cfg) if group == "main" else slow_base_variants(cfg)
    for sc in scenarios:
        r, _, m = run_scenario(cfg, sc, seed, vs, protocol="test")
        rows.extend(r)
        meta[f"{sc}|{seed}"] = json_safe_meta(m)["drift_events"]
    clear_cache()
    return rows, meta, [], {}


def main(argv=None) -> None:
    p = base_parser(__doc__)
    p.add_argument("--group", choices=["main", "slow_base", "all"], default="all")
    args = p.parse_args(argv)
    groups = ["main", "slow_base"] if args.group == "all" else [args.group]
    for group in groups:
        name = "ablation" if group == "main" else "ablation_slow_base"
        cfg, seeds, out_dir, log = setup(args, name)
        scenarios = cfg["experiments"]["ablation"]["scenarios"]
        rows, meta, _, _ = collect(run_jobs(cfg, [(s, scenarios, group) for s in seeds], job, log))
        vs = variants(cfg) if group == "main" else slow_base_variants(cfg)
        save_results(out_dir, cfg, rows, meta, extra={"seeds": seeds, "variants": list(vs)})


if __name__ == "__main__":
    main()
