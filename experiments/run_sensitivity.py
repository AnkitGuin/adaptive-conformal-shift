"""Sensitivity analysis (Section 23) on the TEST protocol with frozen hyperparameters.

One factor is varied at a time; everything else stays at the frozen values.

  alpha            nominal coverage 80 / 90 / 95 %          (all methods)
  cal_size         most recent n calibration points           (all methods)
  window           rolling W and DA-ACP max_buffer            (rolling, da_acp)
  decay            lambda for weighted and DA-ACP             (weighted, da_acp)
  ph_threshold     Page-Hinkley threshold (1e9 = never fires) (da_acp)
  magnitude        covariate / noise / conditional strength   (all methods + DA-ACP without detector)

These runs are *descriptive*: they show how results would change with other
choices, they are NOT used to re-select hyperparameters (that would be
test-set tuning).

    python -m experiments.run_sensitivity [--quick]
"""
from __future__ import annotations

from experiments.common import base_parser, collect, run_jobs, save_results, setup
from src.pipeline import clear_cache, interval_methods, run_scenario

HP_SCENARIOS = ["natural", "abrupt_covariate", "noise", "conditional"]


def job(cfg: dict, j: tuple):
    seed = j
    s = cfg["experiments"]["sensitivity"]
    base = interval_methods(cfg, s["methods"])
    rows = []

    def run(sc, methods, factor, value, **kw):
        r, _, _ = run_scenario(cfg, sc, seed, methods, protocol="test", tags={"factor": factor, "value": value}, **kw)
        rows.extend(r)

    for sc in HP_SCENARIOS:
        for a in cfg["conformal"]["alphas_sensitivity"]:
            run(sc, base, "alpha", a, alpha=a)
        for n in s["calibration_sizes"]:
            run(sc, base, "cal_size", n if n else "full", cal_size=n)
        for w in s["windows"]:
            run(sc, {"rolling": {**base["rolling"], "window": w}, "da_acp": {**base["da_acp"], "max_buffer": w}}, "window", w)
        for lam in s["decays"]:
            ratio = base["da_acp"]["decay_fast"] / base["da_acp"]["decay"]
            run(sc, {"weighted": {**base["weighted"], "decay": lam},
                     "da_acp": {**base["da_acp"], "decay": lam, "decay_fast": lam * ratio}}, "decay", lam)
        for th in s["ph_thresholds"]:
            run(sc, {"da_acp": {**base["da_acp"], "ph_threshold": th}}, "ph_threshold", th)

    mag_methods = {**base, "da_acp_no_drift": {**base["da_acp"], "drift_enabled": False}}
    for m in s["covariate_magnitudes"]:
        run("abrupt_covariate", mag_methods, "covariate_magnitude", m, shift_override={"magnitude": m})
    for m in s["noise_magnitudes"]:
        run("noise", mag_methods, "noise_magnitude", m, shift_override={"magnitude": m})
    for c in s["conditional_factors"]:
        run("conditional", mag_methods, "conditional_factor", c, shift_override={"factor": c})
    clear_cache()
    return rows, {}, [], {}


def main(argv=None) -> None:
    args = base_parser(__doc__).parse_args(argv)
    cfg, seeds, out_dir, log = setup(args, "sensitivity")
    rows, _, _, _ = collect(run_jobs(cfg, seeds, job, log))
    save_results(out_dir, cfg, rows, {}, extra={"seeds": seeds})


if __name__ == "__main__":
    main()
