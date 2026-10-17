# Experiment log: issues found and decisions made

A chronological record of problems found during the study and how each was resolved. Items 1–6 describe the
original study protocol; item 7 is a post-study correctness hardening that can affect edge-case CQR coverage
and should be followed by regeneration of raw results if exact updated tables are required.

1. **Tuning grid v1 → v2 (development split only).** The first grid put the optimum for rolling (W=100),
   weighted (λ=0.01), ACI (γ=0.05) and DA-ACP (λ0=0.005) on the grid edge. The grids were extended once
   (W=50; λ up to 0.05; γ up to 0.2; λ0 up to 0.01) and tuning was re-run, still on the dev protocol only.
   Both versions are kept: `results/tuning/tuning_results_grid_v1.csv` and `tuning_results.csv`.
   The optima moved to W=50, λ=0.02, γ=0.05 (unchanged), λ0=0.01.
2. **Memory on a 2 GB machine.** Random forests pickle to ~80 MB each. The fit cache was bounded
   (≤ 6 entries, cleared after every job), and the baselines job frees each fit after use.
3. **YAML parsed `1.0e9` as a string** (YAML 1.1 requires a dot *and* a signed exponent), which crashed the
   first sensitivity run. Fixed in the config (`1000000000.0`), and `PageHinkley` now casts its parameters
   to float. The sensitivity experiment was then re-run in full.
4. **Follow-up ablation added after seeing results.** The main ablation showed no benefit from the drift
   detector at the tuned λ0. A follow-up (`ablation_slow_base`) was added to test whether the detector helps
   with a slow base memory. It is labelled exploratory in the report. It does not change any frozen
   hyperparameter or any headline number.
5. **Figure 6 artefact.** A centred moving average computed with `np.convolve(mode="same")` zero-pads at the
   end of the stream, which made coverage appear to collapse at the end of the shorter, resampled stream.
   Replaced by a NaN-padded centred window. Only the figure was affected; no metric was.
6. **ACI infinite thresholds.** α_t ≤ 0 gives q = +∞ (the whole real line). These are capped at the largest
   score seen, and the count is reported in the `n_infinite` column (2–6% of steps at α=0.1).
7. **Score-set coverage and split boundaries.** Online coverage is now evaluated from the defining event
   `score_t ≤ q_t`, which is important for negative CQR corrections whose exact set can be empty even when
   display bounds are kept ordered. Chronological validation now requires strict timestamp separation, so
   duplicate timestamps cannot straddle train/calibration/test boundaries. The Bike Sharing loader also verifies
   the canonical SHA-256 by default.
