# Adaptive Conformal Prediction under Distribution Shift

**An empirical study of drift-aware calibration and uncertainty quantification for regression**

This repository is a small, reproducible research project. It asks one question:

> Can adaptive recalibration and drift-aware weighting make conformal prediction intervals more reliable after a distribution shift, while keeping them reasonably narrow?

The project uses classical, CPU-only methods: scikit-learn regressors, conformal prediction implemented from scratch, and a Page-Hinkley drift detector. The data are the UCI Bike Sharing hourly records, kept in chronological order, plus controlled synthetic shifts applied to the test stream.

> **What is guaranteed and what is not.** Split conformal prediction guarantees marginal coverage ≥ 1 − α when calibration and test points are *exchangeable*. Our chronological experiments break that assumption on purpose: the data are temporally dependent, and we add shifts. So none of the adaptive methods here, DA-ACP included, has a finite-sample coverage guarantee in these experiments. We only investigate whether adaptive calibration improves **empirical** coverage under controlled distribution shifts. The ideal ACI recursion has a long-run *average* miscoverage result for arbitrary sequences, but this repository caps infinite thresholds at the largest observed score for finite-width reporting; the capped ACI reference should therefore be interpreted empirically rather than as a literal reproduction of that theorem.

The full write-up is in [`report/report.md`](report/report.md). Issues found and decisions made along the way are logged in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

---

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate       # optional
pip install -r requirements.txt                         # numpy, pandas, scikit-learn, scipy, matplotlib, PyYAML, joblib, pytest
python -m experiments.download_data                     # UCI Bike Sharing -> data/raw/hour.csv (checksum verified)
python -m pytest -q                                     # unit tests
python run_all.py --skip-tuning                         # reproduce all results with the frozen hyperparameters
```

`python run_all.py` (without `--skip-tuning`) also re-runs hyperparameter selection on the development split. That step overwrites `configs/selected_hparams.yaml`. `python run_all.py --quick` is a smoke test that finishes in a few minutes.

Individual steps:

| Command | What it does | Time (2 CPU cores) |
|---|---|---|
| `python -m experiments.run_tuning --no-hparams` | Grid search on the **development** split (inside the training period); writes `configs/selected_hparams.yaml` | ~9 min |
| `python -m experiments.run_shift_experiments` | Main study: 7 scenarios × 8 methods × 10 seeds | ~5 min |
| `python -m experiments.run_ablation` | DA-ACP ablations A–E, plus the slow-base follow-up | ~15 min |
| `python -m experiments.run_sensitivity` | α, calibration size, window, decay, PH threshold, shift magnitude | ~25 min |
| `python -m experiments.run_baselines` | Point models, cross-model generalisation, removal of time-identifying features | ~15 min |
| `python -m experiments.analyze` | All tables (CSV + Markdown) and figures, regenerated from `results/raw/` | ~30 s |

Common CLI flags: `--config configs/default.yaml`, `--seed 42`, `--seeds 0 1 2`, `--output results/`, `--n-jobs 2`, `--quick`, and `--set key.sub=value` for dotted config overrides.

---

## 2. Methods

The symmetric methods use the same fitted point predictor f̂ and nonconformity score s = |y − f̂(x)|. CQR is the deliberate exception: it uses two fitted quantile predictors and its asymmetric score. The calibrators differ only in how past scores are turned into a threshold q_t.

| Method | Threshold q_t | File |
|---|---|---|
| Static split | Order statistic s_(⌈(n+1)(1−α)⌉) of the calibration scores, fixed for all test points | `src/conformal/split.py` |
| Rolling window | The same conformal quantile over the most recent W observed scores | `src/conformal/rolling.py` |
| Exp. weighted | Weighted conformal quantile with weights w = exp(−λ·age); the test point has weight 1 at +∞ | `src/conformal/weighted.py` |
| ACI (reference) | Gibbs & Candès (2021): α_t ← α_t + γ(α − err_t) | `src/conformal/adaptive.py` |
| **DA-ACP** | Exp. weighted plus a Page-Hinkley detector on the residual stream. After an alarm, λ is boosted to λ_fast and annealed back to λ₀ over `boost_duration` steps. Optionally, scores from before the estimated onset are down-weighted | `src/conformal/adaptive.py` |
| CQR (benchmark) | Romano et al. (2019) score max(q̂_lo − y, y − q̂_hi) with the static or DA-ACP calibrator | `src/conformal/cqr.py` |

The quantile conventions are implemented and documented in [`src/conformal/quantiles.py`](src/conformal/quantiles.py):

- **Unweighted quantile:** the k-th smallest score, with k = ⌈(n+1)(1−α)⌉ (+∞ if k > n). NumPy interpolation is never used.
- **Weighted quantile:** inf{s : Σ_{s_i ≤ s} w_i / (Σ w_j + w_test) ≥ 1 − α}. With equal weights this reduces exactly to the unweighted rule, and a unit test checks that.
- **Infinite thresholds:** a +∞ threshold is replaced by the largest available score and counted (`n_infinite` column). This finite-width fallback is a documented implementation choice, not the exact infinite prediction set used in the ideal conformal/ACI definition. It happens only for ACI in our runs.

**Causal online protocol** (`src/conformal/base.py::run_online`). At each step t:

1. Compute q_t from scores observed before t.
2. Report the interval.
3. Observe y_t.
4. Score it and update the buffer and the detector.

A unit test checks this ordering: changing future scores never changes past thresholds.

### DA-ACP in one paragraph

DA-ACP keeps a rolling buffer (W = 3000) of calibration and observed test scores. In calm periods it weights them by exp(−λ₀·age). It feeds the standardised absolute residuals r_t / mean(r_cal) into a one-sided Page-Hinkley test (δ = 0.1, threshold h). When the test fires, the decay is raised to λ_fast and linearly annealed back to λ₀. Optionally, scores from before the detector's estimated change onset are down-weighted by `pre_drift_weight`. The threshold at time t is the weighted conformal quantile of the buffer. Every mechanism can be switched off in the config, which is how the ablations are run.

---

## 3. Experimental protocol and leakage prevention

- **Chronological split:** 60% train (2011-01-01 → 2012-03-15), 20% calibration (→ 2012-08-07), 20% test (→ 2012-12-31). There is no shuffling. `validate_split` checks time order.
- **Fitting:** preprocessing (imputation, scaling, one-hot encoding) sits inside the sklearn pipeline and is fitted on the training split only. Cyclic sin/cos features are stateless. `casual` and `registered` sum to the target, so they are rejected by `check_feature_spec`.
- **Shifts:** synthetic shifts are applied only to a copy of the **test** stream. They are seeded, and their parameters are recorded in `metadata.json`. Shift magnitudes use training-split standard deviations.
- **Hyperparameters:** chosen by `run_tuning.py` on a *development* train/cal/test split nested inside the outer training period. Every method family gets the same kind of grid. The selection criterion is interval score relative to static. The selected values are frozen in `configs/selected_hparams.yaml`. The test split never influences a choice. The sensitivity analysis on the test protocol is descriptive only.
- **Seeds:** 10 seeds (0–9) for all test-protocol experiments; dev seeds are 100–102. A seed controls the random change-point jitter, the noise draws, and the model randomness. All seeds share the same underlying test period, so seed-to-seed spread understates real-world variability.
- **Control experiment:** an exchangeable IID control (random split membership) checks the textbook guarantee.
- **Time-identifying features:** a `no_time_id` feature set drops `yr`, `season` and `mnth` (Table 10).

### Shift scenarios (`configs/shifts.yaml`)

| ID | Scenario | Transformation |
|---|---|---|
| A | IID control | Random split membership, no shift |
| A' | Natural | Chronological split, no synthetic shift (only the real 2012 trend) |
| B | Gradual covariate | temp, atemp, hum + 1.5 sd·ramp(t), ramp from 10% to 90% of the stream |
| C | Abrupt covariate | temp, atemp, hum + 1.5 sd from the change point (≈30% ± 5%) |
| C' | Covariate resampling | After the change point, off-peak hours are kept with p = 0.15 (pure P(X) shift) |
| D | Noise | y + N(0, (0.5·sd_train(y))²) after the change point |
| E | Conditional | y × 1.5 on working days after the change point |

---

## 4. Main results (10 seeds, α = 0.1, gradient boosting)

These numbers come from `results/tables/`. The full analysis is in the report.

See `results/tables/table_overview.md` for coverage and interval score per scenario and method. Main findings:

1. **No shift, exchangeable (A):** static split conformal reaches **0.900** coverage, as theory predicts. Every method is within ~0.01 of nominal.
2. **Real temporal drift alone breaks static conformal:** with the chronological split, static covers only **0.841**. Under synthetic shifts its coverage drops to 0.81 (abrupt), 0.80 (noise), 0.78 (resampling) and **0.68** (conditional, the largest degradation). Its worst 250-step window falls as low as 0.43.
3. **Every adaptive method restores ~0.90 marginal coverage** in every scenario. The cost is 23–43% wider intervals than static (111–126% under the conditional shift), because static is too narrow for the post-shift regime. Interval scores improve by 5–33%.
4. **DA-ACP is not better than a well-tuned exponentially weighted conformal predictor.** The ablation shows that removing the drift detector changes the interval score by < 0.3% on average. At the tuned λ₀ the recency weighting already adapts quickly, so the detector has little left to add. Among the adaptive methods, the dominant factor is the **effective memory length** (decay λ or window W).
5. Freezing the calibration buffer and updating it only after alarms (ablation C) over-covers and is 18% worse. Continuous updating matters more than detection.
6. The detector *does* help when the base memory is long: post-shift coverage improves by 6–14 points over weighted conformal with the same slow λ (follow-up ablation). Even so, it does not beat simply choosing a short memory.
7. **Failure modes:** conditional shift restricted to a subgroup. Marginal coverage recovers, but working days stay under-covered and non-working days over-covered. Also, slow memory settings under-cover badly (small λ gives a worst window of 0.72).

---

## 5. Repository layout

```
configs/            default.yaml (all settings), shifts.yaml, experiments.yaml, selected_hparams.yaml (frozen, from dev tuning)
src/data/           load.py (download/load), preprocess.py (features, splits, leakage checks), shifts.py (shift generators)
src/models/         regressors.py (Ridge, RF, GBR, HistGB, quantile HistGB)
src/conformal/      quantiles.py, base.py (scores, intervals, online protocol), split/rolling/weighted/adaptive/cqr.py
src/drift/          page_hinkley.py
src/evaluation/     metrics.py, coverage.py (rolling/worst-window/recovery/delay), statistics.py (bootstrap, Wilcoxon)
src/pipeline.py     one experiment run: split -> fit -> calibrate -> shift -> online conformal -> metrics
experiments/        run_tuning, run_shift_experiments, run_baselines, run_ablation, run_sensitivity, analyze, download_data
tests/              unit tests (quantiles, conformal, drift, metrics, shifts, data/leakage)
results/raw/<exp>/  config.json, metrics.csv, metadata.json, predictions.csv.gz, traces.npz, run_info.json
results/tables/     Table 1–10 as CSV + Markdown (ALL_TABLES.md)
results/figures/    Figures 0–12
report/report.md    research report
notebooks/          exploratory_analysis.ipynb (EDA only; generated by make_notebook.py)
```

Raw predictions are stored for the first seed (`save_predictions: first_seed`). The columns are timestamp, y_true, y_pred, lower, upper, residual, method, experiment, seed, drift_detected and drift_score. Per-step coverage, width and alarm traces are stored for **all** seeds in `traces.npz`, so every figure can be regenerated. Set `experiment.save_predictions: all` to store every seed (~90 MB).

## 6. Using another dataset

Set `data.source: csv`, `data.path`, `data.timestamp_col` (or `date_col` + `hour_col`), `data.target` and the `features` block in a config file. Then adapt the column names used in `configs/shifts.yaml`. See `data/README.md`.

## 7. Limitations (summary)

- There is one real dataset and one test period; seeds do not provide new data.
- The shifts are synthetic and stylised. Covariate shifts that leave y unchanged also change the observed X→Y relationship (a sensor-drift model).
- The point model is never retrained, so all adaptation happens in the calibration layer.
- Only marginal coverage is targeted. Subgroup (conditional) coverage is not guaranteed, and the conditional-shift experiment shows it can fail.
- Hyperparameters were selected on the development period, which contains its own shifts; the conclusions may depend on that choice.
- Grid v1 put several optima on the grid boundary, so the grid was extended once (v2) on the dev split. Both versions are kept in `results/tuning/`.

## References

The references are listed in the report. The project relies on: Vovk, Gammerman & Shafer (2005); Lei et al. (2018); Romano, Patterson & Candès (2019); Tibshirani et al. (2019); Gibbs & Candès (2021); Barber et al. (2023); Page (1954); Gneiting & Raftery (2007); Fanaee-T & Gama (2013).

License: MIT (code). Dataset: CC BY 4.0 (UCI).
