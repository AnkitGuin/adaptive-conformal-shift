"""Statistical summaries across seeds and paired comparisons.

The experimental unit for all comparisons is a **seed** (one realisation of
the random shift + model seeds).  Methods evaluated on the same seed are
paired, so we use paired tests on per-seed differences:

* paired bootstrap percentile CI of the mean difference;
* Wilcoxon signed-rank test (two-sided, scipy).

With 10 seeds these tests have limited power; p-values are reported as
descriptive evidence, not as confirmatory hypothesis tests, and no multiple-
comparison correction is claimed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_std_ci(x: np.ndarray, level: float = 0.95) -> dict:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    m, s = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * s / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": m, "std": s, "ci_lo": m - h, "ci_hi": m + h, "n": n}


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        for m in metrics:
            d = mean_std_ci(g[m].to_numpy())
            row[f"{m}_mean"], row[f"{m}_std"], row[f"{m}_n"] = d["mean"], d["std"], d["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, level: float = 0.95,
                        rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Mean of (a - b) and percentile bootstrap CI over paired units."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")
