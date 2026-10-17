"""Run the complete pipeline from scratch.

    python run_all.py                 # full study (10 seeds; ~1-2 h on a 2-core laptop)
    python run_all.py --quick         # smoke test (2 seeds, few minutes)
    python run_all.py --skip-tuning   # reuse the frozen configs/selected_hparams.yaml

Steps
  1. download data if missing
  2. hyperparameter selection on the development protocol (writes configs/selected_hparams.yaml)
  3. main shift experiments (test protocol)
  4. baselines / point models / cross-model generalisation
  5. ablation study
  6. sensitivity analysis
  7. tables + figures (+ statistical tests)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("data", ["-m", "experiments.download_data"]),
    ("tuning", ["-m", "experiments.run_tuning", "--no-hparams"]),
    ("shift", ["-m", "experiments.run_shift_experiments"]),
    ("baselines", ["-m", "experiments.run_baselines"]),
    ("ablation", ["-m", "experiments.run_ablation"]),
    ("sensitivity", ["-m", "experiments.run_sensitivity"]),
    ("analysis", ["-m", "experiments.analyze"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--only", nargs="+", choices=[s for s, _ in STEPS])
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    for name, cmd in STEPS:
        if args.only and name not in args.only:
            continue
        if name == "tuning" and args.skip_tuning:
            continue
        extra = []
        if name not in ("data",):
            extra += ["--config", args.config]
            if args.output:
                extra += ["--output", args.output]
            if name != "analysis":
                if args.quick:
                    extra += ["--quick"] if name != "tuning" else ["--seeds", "100"]
                if args.n_jobs:
                    extra += ["--n-jobs", str(args.n_jobs)]
        t0 = time.time()
        print(f"\n=== [{name}] python {' '.join(cmd + extra)}", flush=True)
        subprocess.run([sys.executable, *cmd, *extra], check=True)
        print(f"=== [{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
