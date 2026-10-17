# instructions.md
# Adaptive Conformal Prediction under Distribution Shift
## A research-oriented classical/statistical ML project

---

## 0. Role and objective

You are a senior ML research engineer and scientific-software developer.

Your task is to **fully implement, test, document, and package** the research project described in this file.

Do not merely create a demo or notebook. Build a reproducible research repository that could reasonably support a master's application and a small research report.

The project must emphasize:

- classical/statistical machine learning;
- uncertainty quantification;
- conformal prediction;
- distribution shift;
- calibration;
- rigorous experimental methodology;
- reproducibility;
- ablation studies;
- statistical evaluation;
- clear limitations.

Do NOT use large language models, RAG, agents, CNNs, transformers, or other unnecessarily heavy deep-learning components.

The entire project must be runnable on a normal laptop with CPU-only execution.

The implementation should be understandable enough that the student can defend every major algorithmic choice in an interview.

---

# 1. Project title

Use the working title:

> **Adaptive Conformal Prediction under Distribution Shift**

A more research-oriented subtitle may be used:

> **An Empirical Study of Drift-Aware Calibration and Uncertainty Quantification for Regression**

Do not claim that the proposed method is mathematically novel unless the implementation genuinely establishes a novel theoretical contribution. The project should instead frame its contribution as an experimental/algorithmic investigation.

---

# 2. Core research problem

Most supervised ML models output a point prediction:

    y_hat = f(x)

but a point prediction does not tell us how uncertain the prediction is.

Conformal prediction can convert a point predictor into a prediction interval:

    C(x) = [lower(x), upper(x)]

with a marginal coverage guarantee under the assumptions of the conformal method.

The problem is that the data distribution may change after calibration.

Let:

    P_train(X, Y) != P_test(X, Y)

Examples:

- sensor operating conditions change;
- weather changes over time;
- customer behavior changes;
- the relationship between inputs and targets drifts;
- noise level increases;
- a new operating regime appears.

A conformal predictor calibrated on historical data may therefore become:

- under-covering;
- unnecessarily wide;
- miscalibrated;
- slow to adapt.

The central research question is:

> **Can adaptive recalibration and distribution-shift-aware weighting improve the reliability of conformal prediction after distribution shift while keeping prediction intervals reasonably narrow?**

---

# 3. Main research hypothesis

The primary hypothesis is:

> When the test distribution differs from the calibration distribution, a drift-aware adaptive conformal strategy can recover empirical coverage more effectively than static split conformal prediction, while avoiding excessive interval width.

This is an empirical hypothesis, NOT a theorem.

The project must explicitly distinguish:

1. theoretical guarantees under exchangeability;
2. empirical behavior under distribution shift;
3. assumptions made by weighted/adaptive variants;
4. situations where the proposed method fails.

Do not write that conformal prediction "guarantees 90% coverage under distribution shift" unless the precise assumptions required by a specific method are satisfied.

---

# 4. Main research questions

The implementation and final report must answer the following questions.

### RQ1 — Baseline reliability

How well does ordinary split conformal prediction maintain empirical coverage when train/calibration/test data are exchangeable?

### RQ2 — Distribution shift

How does coverage change when the test distribution differs from the calibration distribution?

### RQ3 — Adaptation

Can rolling-window recalibration recover coverage after a distribution shift?

### RQ4 — Weighting

Does exponentially weighting recent calibration examples improve adaptation compared with an unweighted rolling window?

### RQ5 — Drift detection

Does explicitly detecting distribution drift before changing the calibration strategy improve the accuracy/efficiency trade-off?

### RQ6 — Efficiency

How much interval width is required to obtain the desired coverage?

### RQ7 — Robustness

How do the methods behave under:

- gradual drift;
- abrupt drift;
- increasing noise;
- feature distribution shift;
- target/conditional shift;
- different calibration-set sizes?

### RQ8 — Failure modes

When does adaptive conformal prediction fail or become unstable?

---

# 5. Recommended dataset

Use a real regression dataset with a meaningful ordering over time.

## Primary dataset: UCI Bike Sharing Dataset

Use the Bike Sharing Dataset, preferably the hourly data.

Target:

    cnt

Useful features include:

- season
- yr
- mnth
- hr
- holiday
- weekday
- workingday
- weathersit
- temp
- atemp
- hum
- windspeed

The natural temporal ordering makes it possible to create realistic train/calibration/test regimes.

Do NOT randomly shuffle the entire dataset before splitting the primary experiment.

The project should preserve chronological order.

If the dataset is unavailable or problematic, the implementation should support a configurable CSV dataset with:

- timestamp column;
- numeric/categorical predictors;
- continuous target.

The code must not depend permanently on one hard-coded absolute file path.

---

# 6. Dataset protocol

Create a chronological split.

A recommended default:

    first 60%  -> training
    next 20%   -> calibration
    final 20%  -> test

However, the exact split must be configurable.

The primary experiment should use:

    Train -> Calibration -> Test

with no leakage.

The model is trained only on the training portion.

The baseline conformal calibration scores are calculated only from the calibration portion.

The test set must remain untouched until final evaluation.

---

# 7. Preprocessing rules

Build preprocessing carefully.

### Numeric variables

Use a reproducible imputation/scaling pipeline if needed.

### Categorical variables

Use one-hot encoding or another simple classical encoding.

### Temporal variables

If using:

- hour;
- month;
- weekday;

consider cyclic representations:

    sin(2*pi*hour/24)
    cos(2*pi*hour/24)

and analogous transformations for periodic variables.

Do not use future information.

Be especially careful with the `yr` field and other variables that may trivially encode chronological position.

Run at least one experiment with potentially time-identifying variables removed, so that the model does not obtain an artificial shortcut.

---

# 8. Data leakage requirements

This is extremely important.

The implementation must NOT:

- fit preprocessing on the entire dataset;
- tune hyperparameters using the test set;
- choose the best method based on test performance and then report that same result as unbiased;
- use future target values to create features;
- randomly shuffle temporal data for the primary experiment;
- calculate calibration residuals using test labels;
- use test coverage to select the final method.

If hyperparameter tuning is performed, use:

- training-only validation;
- nested chronological validation;
- or a clearly documented development split.

The test set is for final evaluation.

---

# 9. Point prediction models

Implement at least three classical regression models.

Recommended:

### Model A
Linear Regression / Ridge Regression

### Model B
Random Forest Regressor

### Model C
Gradient Boosting Regressor

Optional:

### Model D
HistGradientBoostingRegressor

The primary conformal experiments can use one strong baseline model, preferably Gradient Boosting or Random Forest, to isolate the effect of the uncertainty method.

The other models should be used to establish that the conclusions are not entirely model-specific.

Do not spend excessive effort hyperparameter tuning.

The research question is about conformal calibration, not leaderboard optimization.

---

# 10. Conformal prediction fundamentals

Implement the core conformal machinery yourself rather than simply calling a high-level conformal prediction package.

The code should clearly implement the mathematics.

For split conformal regression:

Train:

    f_hat = train_model(D_train)

For each calibration sample:

    r_i = |y_i - f_hat(x_i)|

For target miscoverage level:

    alpha = 1 - coverage

Estimate a suitable upper quantile of the calibration residuals.

Then:

    lower(x) = f_hat(x) - q
    upper(x) = f_hat(x) + q

For example, for nominal 90% coverage:

    alpha = 0.10

The implementation must use a mathematically defensible finite-sample quantile rule rather than silently relying on an arbitrary NumPy interpolation convention.

Document the exact quantile convention used.

---

# 11. Required conformal methods

Implement at least the following methods.

## Method 1 — Static Split Conformal

This is the baseline.

Procedure:

    Train model once
    ↓
    Calibration residuals
    ↓
    Fixed conformal quantile
    ↓
    Apply same interval to all test points

This establishes the standard reference.

---

## Method 2 — Rolling-window Conformal

Maintain a recent calibration buffer.

At time t:

    use most recent W calibration/observed residuals

to compute the conformal quantile.

The window size W must be configurable.

Suggested values:

    100
    250
    500
    1000

depending on dataset size.

This method adapts to recent changes but may become statistically noisy when W is small.

---

## Method 3 — Exponentially Weighted Conformal

Assign larger weights to recent calibration examples.

For residual at age k:

    w_k = exp(-lambda * k)

Normalize weights.

Use a weighted quantile of the nonconformity scores.

The implementation must include a correct weighted quantile routine.

Test it separately.

Do not simply multiply residuals by weights and then take an ordinary quantile.

---

## Method 4 — Drift-Aware Adaptive Conformal Prediction

This is the project's main proposed method.

Working name:

> **DA-ACP — Drift-Aware Adaptive Conformal Prediction**

The method should combine:

1. recent calibration residuals;
2. exponentially decaying weights;
3. a drift detector;
4. adaptive calibration-window behavior.

A conceptual procedure:

    incoming observations
           |
           v
    calculate residual
           |
           v
    update residual buffer
           |
           v
      drift detector
        /       \
      no         yes
      |           |
      v           v
  normal       increase
  weighting    adaptation
                  |
                  v
        emphasize recent data
                  |
                  v
         recompute interval

The exact design is up to the implementation, but it must be simple, transparent, and reproducible.

---

# 12. Important distinction: residual availability

A real online conformal system does not know:

    y_t

before making the prediction at time t.

Therefore the system must operate as follows:

At time t:

    1. receive x_t
    2. produce prediction interval using information available before t
    3. observe y_t
    4. calculate residual
    5. update calibration buffer
    6. update drift detector

Never use y_t to construct the interval that is reported for the same t.

This ordering must be explicitly enforced in code.

---

# 13. Drift detection

Implement at least one simple drift detector.

Recommended primary detector:

> Page-Hinkley

It is appropriate because the goal is to detect changes in the residual/error stream.

The drift detector should operate on a stream such as:

    absolute residuals

or optionally:

    signed residuals

The project should investigate which is more appropriate.

Optional additional detectors:

- ADWIN;
- CUSUM;
- rolling KS test;
- Page-Hinkley.

Do not implement five detectors unless there is enough time.

One well-implemented detector + an ablation study is preferable to many poorly implemented detectors.

---

# 14. Proposed DA-ACP algorithm

Implement a transparent first version.

Maintain:

    residual_buffer
    timestamps
    drift_detector_state

At each prediction time t:

### Step 1

Use the current calibration buffer.

### Step 2

Compute exponentially decaying weights:

    w_i = exp(-lambda * age_i)

### Step 3

Compute the weighted conformal quantile.

### Step 4

Generate:

    [f_hat(x_t) - q_t,
     f_hat(x_t) + q_t]

### Step 5

Observe y_t.

### Step 6

Compute:

    r_t = |y_t - f_hat(x_t)|

### Step 7

Add r_t to the residual buffer.

### Step 8

Update the drift detector.

### Step 9

If drift is detected:

- temporarily shorten the effective calibration horizon;
- increase the weighting of recent samples;
- optionally reset/partially reset the calibration buffer;
- record the drift event.

The exact mechanism must be configurable.

Do not silently change multiple parameters at once.

Every adaptive mechanism must be exposed in the configuration.

---

# 15. Critical methodological issue

Do not claim that DA-ACP has the same formal coverage guarantee as ordinary exchangeable split conformal prediction.

Under distribution shift, ordinary exchangeability assumptions are violated.

The correct framing is:

> "We investigate whether adaptive calibration improves empirical coverage under controlled distribution shifts."

This distinction must appear in:

- README;
- report;
- code comments where appropriate;
- conclusions.

---

# 16. Distribution-shift scenarios

The project MUST contain controlled experiments.

Do not rely only on whatever naturally happens in the real dataset.

Create synthetic perturbations around the real dataset.

## Scenario A — No shift

Train, calibration, and test follow the same distribution.

Purpose:

Verify that standard split conformal behaves as expected.

---

## Scenario B — Gradual covariate drift

Modify selected test features gradually.

For example:

    x'_t = x_t + delta(t)

where:

    delta(t)

increases smoothly over time.

Purpose:

Test whether adaptive calibration follows slowly changing conditions.

---

## Scenario C — Abrupt covariate shift

At a known change point:

    distribution A → distribution B

Purpose:

Test how quickly the methods recover.

---

## Scenario D — Increased noise

Generate:

    y' = y + epsilon

where:

    epsilon ~ N(0, sigma_t^2)

and sigma_t increases after a change point.

Purpose:

Test whether intervals expand appropriately.

---

## Scenario E — Conditional shift

Change the relationship:

    y = f(x)

after a change point.

For example:

    y'_t = y_t * c

for a controlled subset of observations.

Purpose:

This is harder because the underlying conditional relationship itself changes.

Document this carefully.

---

# 17. Synthetic shift generation

The implementation should contain a dedicated module such as:

    src/data/shifts.py

with deterministic functions like:

    apply_gradual_covariate_shift(...)
    apply_abrupt_covariate_shift(...)
    apply_noise_shift(...)
    apply_conditional_shift(...)

Every transformation must:

- accept a random seed;
- record parameters;
- return metadata;
- avoid modifying the original dataset;
- preserve reproducibility.

Do not randomly alter data without recording exactly what was done.

---

# 18. Main metrics

Coverage alone is not enough.

Report at least:

## 1. Empirical coverage

    Coverage =
    fraction of observations where y is inside [L, U]

Target:

    90%

and optionally:

    80%
    95%

---

## 2. Mean interval width

    mean(U - L)

Lower is generally more efficient, conditional on acceptable coverage.

Do not collapse the project into "higher coverage = better."

Coverage and efficiency must be studied together.

---

## 3. Coverage gap

    |empirical_coverage - nominal_coverage|

---

## 4. Conditional/time-local coverage

Calculate coverage over rolling windows.

Example:

    100-observation rolling coverage

This is essential for visualizing adaptation.

---

## 5. Worst-window coverage

Report the minimum coverage over sufficiently large evaluation windows.

This helps identify periods of severe undercoverage.

---

## 6. Interval score

Implement a proper interval scoring metric.

For interval [L, U] with nominal miscoverage alpha:

    score =
        (U-L)
        + (2/alpha)(L-y) if y < L
        + (2/alpha)(y-U) if y > U

Otherwise:

    score = U-L

Lower is better.

Use this metric because it combines interval sharpness and penalties for missed observations.

---

## 7. Prediction error

Report:

- MAE;
- RMSE.

These evaluate the underlying point predictor and ensure that interval improvements are not caused by a catastrophic point model.

---

# 19. Adaptation metrics

The adaptive method should be evaluated on how quickly it responds to a known shift.

Implement:

### Detection delay

Number of observations between the true shift point and the detected shift.

### Recovery time

Number of observations required for rolling coverage to return within a specified tolerance of nominal coverage.

Example:

    |coverage - 0.90| <= 0.05

for a sustained window.

### Post-shift coverage

Coverage after the shift but before recovery.

These metrics should be reported separately.

---

# 20. Required baselines

At minimum compare:

1. No interval / point prediction
2. Static split conformal
3. Rolling conformal
4. Exponentially weighted conformal
5. DA-ACP

Optional:

6. CQR — Conformalized Quantile Regression

CQR is strongly recommended if implementation time permits.

---

# 21. CQR extension

Implement:

    lower_quantile_model
    upper_quantile_model

for:

    q_low = alpha/2
    q_high = 1-alpha/2

Then calibrate the interval using conformal scores.

This produces asymmetric intervals.

Compare:

    symmetric residual conformal
    vs
    CQR

especially when the target distribution is heteroscedastic.

Do not make CQR the primary novelty. It is a useful benchmark.

---

# 22. Ablation studies

This is essential for research quality.

The main proposed method contains several components.

Remove them one at a time.

### Ablation A

DA-ACP

vs

DA-ACP without drift detection.

### Ablation B

DA-ACP

vs

DA-ACP without exponential weighting.

### Ablation C

DA-ACP

vs

DA-ACP without rolling adaptation.

### Ablation D

Different decay rates:

    lambda ∈ {small, medium, large}

### Ablation E

Different calibration window sizes.

### Ablation F

Different shift magnitudes.

The purpose is to determine which components actually matter.

---

# 23. Sensitivity analysis

Study:

- nominal coverage: 80%, 90%, 95%;
- calibration size;
- window size;
- decay rate;
- drift threshold;
- shift magnitude;
- noise magnitude.

Do not tune every parameter until the test set looks best.

Choose reasonable values using training/development experiments, then freeze them for final test evaluation.

---

# 24. Statistical rigor

Do not report one lucky run.

For stochastic experiments:

- use multiple random seeds;
- preferably use at least 10 seeds for major synthetic experiments;
- report mean;
- report standard deviation;
- optionally report confidence intervals.

For example:

    Coverage:
    0.887 ± 0.021

Do not overclaim statistical significance.

If comparing methods, optionally use:

- paired bootstrap confidence intervals;
- Wilcoxon signed-rank test;
- paired permutation test.

The statistical test must match the experimental unit.

---

# 25. Important issue with temporal dependence

The Bike Sharing observations are not IID.

Therefore the report must distinguish:

1. classical conformal theory under exchangeability;
2. empirical evaluation on temporally ordered data.

Do not present the temporal experiment as if the standard IID guarantee automatically applies.

The chronological experiment is specifically intended to test robustness under realistic dependence and shift.

---

# 26. Experimental matrix

The final experiment should look approximately like:

    Dataset
       |
       +-- No shift
       |
       +-- Gradual shift
       |
       +-- Abrupt shift
       |
       +-- Noise shift
       |
       +-- Conditional shift

For each:

    Static Conformal
    Rolling Conformal
    Weighted Conformal
    DA-ACP
    Optional CQR

For each method calculate:

    Coverage
    Width
    Coverage gap
    Interval score
    MAE
    RMSE
    Detection delay
    Recovery time

---

# 27. Required visualizations

Generate publication-style figures.

At minimum:

## Figure 1

Prediction intervals on the no-shift test set.

Actual observations should be visually distinguishable from interval boundaries.

---

## Figure 2

Rolling coverage over time under gradual drift.

Include a horizontal nominal-coverage line.

---

## Figure 3

Rolling coverage after abrupt distribution shift.

Mark:

- true shift point;
- detected drift point.

---

## Figure 4

Interval width over time.

Compare:

    static
    rolling
    weighted
    DA-ACP

---

## Figure 5

Coverage vs average interval width.

This is one of the most important summary figures.

---

## Figure 6

Coverage trajectory after a shift.

Align different runs at the shift point if possible.

This allows comparison of recovery speed.

---

## Figure 7

Sensitivity to calibration-window size.

---

## Figure 8

Sensitivity to shift magnitude.

---

## Figure 9

Ablation comparison.

---

# 28. Tables

Generate machine-readable CSV tables and human-readable tables.

Required tables:

### Table 1
Dataset statistics.

### Table 2
Point-model performance.

### Table 3
No-shift conformal performance.

### Table 4
Performance under gradual drift.

### Table 5
Performance under abrupt drift.

### Table 6
Performance under noise shift.

### Table 7
Ablation study.

### Table 8
Sensitivity analysis.

---

# 29. Recommended repository structure

Create approximately:

    adaptive-conformal-shift/
    │
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── pyproject.toml
    ├── .gitignore
    ├── instructions.md
    │
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── README.md
    │
    ├── configs/
    │   ├── default.yaml
    │   ├── experiments.yaml
    │   └── shifts.yaml
    │
    ├── src/
    │   ├── data/
    │   │   ├── load.py
    │   │   ├── preprocess.py
    │   │   └── shifts.py
    │   │
    │   ├── models/
    │   │   └── regressors.py
    │   │
    │   ├── conformal/
    │   │   ├── base.py
    │   │   ├── split.py
    │   │   ├── rolling.py
    │   │   ├── weighted.py
    │   │   ├── cqr.py
    │   │   └── adaptive.py
    │   │
    │   ├── drift/
    │   │   └── page_hinkley.py
    │   │
    │   ├── evaluation/
    │   │   ├── metrics.py
    │   │   ├── coverage.py
    │   │   └── statistics.py
    │   │
    │   └── utils/
    │       ├── config.py
    │       ├── logging.py
    │       └── seeds.py
    │
    ├── experiments/
    │   ├── run_baselines.py
    │   ├── run_shift_experiments.py
    │   ├── run_ablation.py
    │   └── run_sensitivity.py
    │
    ├── tests/
    │   ├── test_quantiles.py
    │   ├── test_conformal.py
    │   ├── test_drift.py
    │   ├── test_metrics.py
    │   └── test_shifts.py
    │
    ├── notebooks/
    │   └── exploratory_analysis.ipynb
    │
    ├── results/
    │   ├── raw/
    │   ├── tables/
    │   └── figures/
    │
    └── report/
        ├── report.md
        └── figures/

The exact structure may be improved if a better architecture is justified, but maintain separation between:

- data;
- models;
- conformal algorithms;
- drift detection;
- evaluation;
- experiments.

---

# 30. Software engineering requirements

The implementation must be:

- modular;
- typed where practical;
- documented;
- reproducible;
- deterministic when a seed is supplied;
- testable.

Avoid:

- one giant Python script;
- notebook-only implementation;
- hidden global state;
- hard-coded Windows paths;
- hard-coded random seeds buried throughout the code;
- duplicated conformal logic;
- undocumented constants.

---

# 31. Command-line interface

Provide a simple CLI.

Examples:

    python -m experiments.run_baselines

    python -m experiments.run_shift_experiments

    python -m experiments.run_ablation

    python -m experiments.run_sensitivity

Ideally support:

    --config configs/default.yaml
    --seed 42
    --output results/

A single command should be able to reproduce the major experiment.

For example:

    python run_experiments.py --config configs/default.yaml

If useful, create a top-level:

    run_all.py

that executes the full pipeline.

---

# 32. Testing requirements

Write unit tests for mathematical components.

At minimum test:

### Weighted quantile

Test:

- uniform weights;
- highly concentrated weights;
- sorted/unsorted input;
- edge cases.

### Conformal quantile

Test finite-sample quantile behavior.

### Interval construction

Verify:

    lower <= upper

for all observations.

### Coverage

Use a hand-constructed toy example.

### Page-Hinkley

Construct an artificial residual stream:

    stable → shift

and verify that drift is eventually detected.

### Shift generators

Verify:

- deterministic behavior with seed;
- expected magnitude of shift;
- no accidental mutation of source data.

---

# 33. Reproducibility

Create a single configuration file containing:

- dataset path;
- target column;
- feature columns;
- split ratios;
- alpha values;
- model parameters;
- window sizes;
- decay rates;
- drift parameters;
- random seeds;
- experiment settings.

Every experiment should save its configuration alongside its results.

For example:

    results/raw/experiment_001/
        config.json
        metrics.csv
        predictions.csv
        metadata.json

This is important for research credibility.

---

# 34. Result storage

Never save only final plots.

Save raw predictions:

    timestamp
    y_true
    y_pred
    lower
    upper
    residual
    method
    experiment
    seed
    drift_detected
    drift_score

This allows every figure to be regenerated later.

---

# 35. Report requirements

Generate a proper research report.

Recommended structure:

## Abstract

150–250 words.

## 1. Introduction

Explain:

- prediction uncertainty;
- limitations of point predictions;
- conformal prediction;
- distribution shift.

## 2. Research Question

Clearly state RQs and hypothesis.

## 3. Background

Explain:

- regression;
- conformal prediction;
- exchangeability;
- calibration;
- distribution shift;
- drift detection.

## 4. Methodology

Describe:

- base predictors;
- split conformal;
- rolling conformal;
- weighted conformal;
- DA-ACP;
- drift detector.

Include equations.

## 5. Experimental Setup

Describe:

- dataset;
- splits;
- shift scenarios;
- metrics;
- seeds;
- hyperparameters.

## 6. Results

Include tables and figures.

## 7. Ablation Study

Determine whether each component is useful.

## 8. Failure Analysis

Discuss cases where adaptive methods fail.

## 9. Limitations

Be honest.

## 10. Conclusion

Summarize what the experiments actually establish.

## Future Work

Examples:

- online conformal methods;
- conformal prediction for time series;
- multivariate prediction;
- stronger distribution-shift assumptions;
- theoretical guarantees.

---

# 36. Research contribution framing

The project should NOT claim:

> "We invented a new conformal prediction algorithm."

Instead use language such as:

> "We develop and empirically evaluate a drift-aware adaptive calibration framework..."

or:

> "We investigate whether combining recent-residual weighting with drift detection improves empirical coverage under controlled distribution shifts."

This is credible for a student research project.

If experiments reveal a genuinely interesting result, document it rather than forcing the project to confirm the original hypothesis.

Negative results are acceptable.

---

# 37. Expected scientific conclusions

Do not pre-program the expected answer.

The experiment should determine whether:

- static conformal loses coverage under shift;
- rolling calibration adapts;
- exponential weighting helps;
- drift detection helps;
- adaptation makes intervals too wide;
- some shifts cannot be handled effectively.

It is perfectly acceptable if DA-ACP does not win everywhere.

A strong report should explain trade-offs.

---

# 38. Required final analysis

The final analysis should answer:

### Question A

Does static conformal maintain nominal coverage without shift?

### Question B

Which shift causes the largest degradation?

### Question C

How quickly does each adaptive method recover?

### Question D

Does improved coverage come at the cost of much wider intervals?

### Question E

Which component of DA-ACP contributes most?

### Question F

How sensitive is the method to its hyperparameters?

### Question G

Does the method generalize across different base regressors?

### Question H

What are the failure modes?

---

# 39. Optional extension: multiple real datasets

If the core project is complete, add one or two additional datasets.

Possible choices:

- a weather regression dataset;
- an energy-load dataset;
- an air-quality dataset;
- a traffic dataset.

The purpose is NOT to increase the number of datasets for appearance.

The purpose is to test whether the conclusions transfer beyond Bike Sharing.

One additional dataset with a clean experimental protocol is better than five poorly analyzed datasets.

---

# 40. Optional extension: online learning

Only after the core project works.

Investigate whether the point predictor itself can be updated online.

For example:

    static predictor
    vs
    incremental predictor

Then separately study:

    model adaptation
    vs
    conformal calibration adaptation

This distinction can lead to an interesting analysis:

> Is uncertainty recalibration sufficient, or must the underlying predictive model also adapt?

Do not implement this extension until the main experiments are complete.

---

# 41. What the final resume entry should eventually communicate

Do not write generic bullets like:

> "Built a conformal prediction model."

A strong final resume entry could eventually resemble:

> **Adaptive Conformal Prediction under Distribution Shift**  
> *Python, Scikit-learn, Statistical ML*
>
> - Developed a distribution-shift-aware conformal regression framework combining rolling calibration, exponentially weighted residuals, and Page-Hinkley drift detection; evaluated 90% prediction-interval coverage across controlled temporal, covariate, and noise shifts.
> - Implemented split, rolling, weighted, and CQR baselines with coverage, interval-width, interval-score, drift-delay, and recovery-time analysis across multiple random seeds.

Only use numerical claims after the experiments actually produce those numbers.

---

# 42. What NOT to do

Do not turn this into:

- a Streamlit dashboard as the main contribution;
- a generic "weather prediction" project;
- a Kaggle leaderboard competition;
- a huge deep-learning project;
- an LLM project;
- a collection of unrelated algorithms;
- a notebook with no reproducibility;
- an experiment that tunes everything on the test set;
- a project that reports only RMSE;
- a project that claims theoretical guarantees it does not possess.

A dashboard can be added at the end, but it is not the research contribution.

---

# 43. Implementation order

Follow this order exactly.

## Phase 1 — Project skeleton

Create:

- repository;
- environment;
- package structure;
- configuration system;
- logging;
- seed management.

Do not begin with a giant notebook.

## Phase 2 — Data

Implement:

- loading;
- preprocessing;
- chronological splitting;
- validation;
- exploratory analysis.

## Phase 3 — Point predictors

Implement:

- Ridge;
- Random Forest;
- Gradient Boosting.

Evaluate them independently.

## Phase 4 — Core conformal prediction

Implement and test:

- residual calculation;
- finite-sample conformal quantile;
- interval construction;
- coverage;
- interval width;
- interval score.

Verify no-shift behavior.

## Phase 5 — Baselines

Implement:

- static split conformal;
- rolling conformal;
- exponentially weighted conformal.

## Phase 6 — Drift detection

Implement:

- Page-Hinkley;
- tests;
- visualization.

## Phase 7 — DA-ACP

Implement the proposed adaptive method.

Ensure that prediction occurs BEFORE observing the current target.

## Phase 8 — Synthetic shifts

Implement:

- gradual shift;
- abrupt shift;
- noise shift;
- conditional shift.

## Phase 9 — Experiments

Run:

- baseline;
- shift;
- ablation;
- sensitivity;
- multiple seeds.

## Phase 10 — Analysis

Generate:

- tables;
- figures;
- raw CSV outputs;
- statistical summaries.

## Phase 11 — Report

Write the research report based ONLY on actual results.

## Phase 12 — Cleanup

Run:

- all tests;
- reproducibility check;
- clean-environment installation;
- complete experiment from scratch.

---

# 44. Definition of done

The project is complete only when:

- [ ] code runs from a clean environment;
- [ ] data acquisition instructions are documented;
- [ ] no absolute local paths exist;
- [ ] tests pass;
- [ ] static conformal is implemented;
- [ ] rolling conformal is implemented;
- [ ] weighted conformal is implemented;
- [ ] drift detection is implemented;
- [ ] DA-ACP is implemented;
- [ ] synthetic shift generators work;
- [ ] temporal leakage is prevented;
- [ ] no-shift experiment works;
- [ ] distribution-shift experiments work;
- [ ] ablation study works;
- [ ] sensitivity analysis works;
- [ ] multiple random seeds are supported;
- [ ] raw predictions are saved;
- [ ] figures are reproducible;
- [ ] tables are reproducible;
- [ ] report is complete;
- [ ] limitations are documented;
- [ ] README explains how to reproduce results.

---

# 45. Engineering principle

Prefer a smaller number of correct, transparent components over a huge codebase.

If a method can be implemented in 100 clear lines, do not turn it into 1000 opaque lines.

Every major mathematical operation should be easy to locate in the source code.

The student must be able to explain:

- what conformal prediction does;
- why exchangeability matters;
- why distribution shift breaks the standard assumptions;
- how the weighted quantile works;
- why recent residuals may be more useful;
- how drift detection works;
- why adaptive methods can fail;
- how coverage and interval width trade off.

---

# 46. Final instruction to the coding LLM

You are responsible for implementing the entire project described above.

Do not ask the student to manually implement individual modules unless absolutely necessary.

Proceed autonomously through:

1. repository creation;
2. architecture;
3. implementation;
4. tests;
5. experiment scripts;
6. analysis scripts;
7. documentation;
8. report generation.

If you need to make a reasonable implementation choice that is not explicitly specified, choose the simplest scientifically defensible option and document the choice.

Do not silently change the research question.

Do not replace classical methods with deep learning.

Do not fabricate experimental results.

Do not fabricate citations or papers.

Do not claim novelty without evidence.

Do not optimize for impressive-looking code; optimize for scientific correctness and reproducibility.

If an experiment fails, debug it and record the issue rather than hiding it.

If a proposed method performs worse than a baseline, report that honestly.

The final repository should look like a compact, reproducible research project rather than a software-demo project.

At the end, provide:

1. a complete implementation;
2. a clear README;
3. reproducible commands;
4. passing tests;
5. generated experimental results;
6. figures;
7. tables;
8. a research report;
9. a concise explanation of the final algorithm;
10. a summary of what was learned from the experiments.
