"""Generate all tables (CSV + Markdown) and figures from saved raw results.

Nothing here re-runs a model: every number and figure is derived from files in
results/raw/, so results can be regenerated or audited at any time.

    python -m experiments.analyze [--output results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
import pandas as pd  # noqa: E402

from src.data.preprocess import chronological_split, dataset_statistics  # noqa: E402
from src.evaluation.coverage import rolling_mean  # noqa: E402
from src.evaluation.statistics import paired_bootstrap_ci, summarize, wilcoxon_p  # noqa: E402
from src.pipeline import get_dataframe, load_full_config  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

METHOD_ORDER = ["static", "rolling", "weighted", "aci", "da_acp", "cqr_static", "cqr_da_acp"]
LABEL = {"static": "Static split", "rolling": "Rolling window", "weighted": "Exp. weighted", "aci": "ACI",
         "da_acp": "DA-ACP", "cqr_static": "CQR (static)", "cqr_da_acp": "CQR + DA-ACP", "point": "Point only",
         "da_acp_no_drift": "DA-ACP w/o detector"}
COLOR = {"static": "#555555", "rolling": "#1f77b4", "weighted": "#2ca02c", "aci": "#9467bd", "da_acp": "#d62728",
         "cqr_static": "#8c564b", "cqr_da_acp": "#ff7f0e", "da_acp_no_drift": "#e377c2"}
SCEN_LABEL = {"iid_control": "A: IID control", "natural": "A': Natural (chronological)",
              "gradual_covariate": "B: Gradual covariate", "abrupt_covariate": "C: Abrupt covariate",
              "covariate_resample": "C': Covariate resampling", "noise": "D: Noise increase",
              "conditional": "E: Conditional shift"}
MAIN_METRICS = ["coverage", "coverage_gap", "width", "interval_score", "worst_window_coverage",
                "post_shift_coverage", "recovery_time", "recovered", "detection_delay", "false_alarms", "n_alarms",
                "mae", "rmse", "n_infinite", "coverage_affected", "coverage_unaffected_post",
                "coverage_workingday", "coverage_nonworkingday", "pre_recovery_coverage", "late_coverage"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


# ============================================================================ helpers
def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pm(mean, std, fmt=".3f"):
    if mean is None or np.isnan(mean):
        return "–"
    return f"{mean:{fmt}} ± {std:{fmt}}" if not np.isnan(std) else f"{mean:{fmt}}"


class Out:
    def __init__(self, root: Path):
        self.root = root
        self.tables = root / "tables"
        self.figs = root / "figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figs.mkdir(parents=True, exist_ok=True)
        self.md: list[str] = []

    def table(self, name: str, title: str, df_csv: pd.DataFrame, df_md: pd.DataFrame | None = None, note: str = "",
              floatfmt: str = ".3f"):
        df_csv.to_csv(self.tables / f"{name}.csv", index=False)
        md = f"### {title}\n\n{md_table(df_md if df_md is not None else df_csv, floatfmt)}\n"
        if note:
            md += f"\n{note}\n"
        (self.tables / f"{name}.md").write_text(md)
        self.md.append(md)

    def fig(self, fig, name: str):
        fig.savefig(self.figs / f"{name}.png", bbox_inches="tight")
        plt.close(fig)


def load_raw(root: Path, name: str):
    d = root / "raw" / name
    if not (d / "metrics.csv").exists():
        return None, None, None
    m = pd.read_csv(d / "metrics.csv")
    meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
    tr = np.load(d / "traces.npz") if (d / "traces.npz").exists() else None
    return m, meta, tr


def pretty_summary(m: pd.DataFrame, scenario: str, methods: list[str], cols: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = m[(m.scenario == scenario) & m.method.isin(methods)]
    s = summarize(sub, ["method"], [c for c, _, _ in cols])
    s["_o"] = s.method.map({k: i for i, k in enumerate(methods)})
    s = s.sort_values("_o").drop(columns="_o")
    pretty = pd.DataFrame({"Method": s.method.map(LABEL)})
    for c, label, f in cols:
        pretty[label] = [pm(a, b, f) for a, b in zip(s[f"{c}_mean"], s[f"{c}_std"])]
    return s, pretty


# ============================================================================ tables
def table_dataset(out: Out, cfg: dict):
    df = get_dataframe(cfg)
    split = chronological_split(len(df), cfg["split"])
    st = dataset_statistics(df, split, cfg["data"]["target"])
    out.table("table1_dataset", "Table 1 – Dataset statistics (UCI Bike Sharing, hourly; chronological split)", st,
              note="Target = hourly rental count `cnt`. The test period (Aug–Dec 2012) has a higher mean and variance "
                   "than the training period (2011 – Mar 2012): the real data already contain a trend/seasonal shift.")
    # EDA figure
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    daily = df.set_index("timestamp")[cfg["data"]["target"]].resample("D").mean()
    ax[0].plot(daily.index, daily.values, lw=0.7, color="k")
    ax[0].set_ylabel("daily mean cnt")
    tdaily = df.set_index("timestamp")["temp"].resample("D").mean()
    ax[1].plot(tdaily.index, tdaily.values, lw=0.7, color="tab:orange")
    ax[1].set_ylabel("temp (normalised)")
    ts = df["timestamp"]
    for a in ax:
        for idx, c, lab in [(split.calibration[0], "tab:blue", "calibration"), (split.test[0], "tab:red", "test")]:
            a.axvline(ts.iloc[idx], color=c, ls="--", lw=1)
    ax[0].text(ts.iloc[split.train[len(split.train) // 2]], daily.max() * 0.95, "train", ha="center")
    ax[0].text(ts.iloc[split.calibration[len(split.calibration) // 2]], daily.max() * 0.95, "cal", ha="center")
    ax[0].text(ts.iloc[split.test[len(split.test) // 2]], daily.max() * 0.95, "test", ha="center")
    ax[0].set_title("Bike Sharing data over time with the chronological 60/20/20 split")
    out.fig(fig, "fig0_dataset_overview")


def table_point_models(out: Out, base: pd.DataFrame | None):
    if base is None:
        return
    p = base[base.table == "point"]
    s = summarize(p, ["model", "feature_set", "split_kind", "part"], ["mae", "rmse", "r2"])
    out.table("table2_point_models_full", "Table 2 (full) – Point-model performance", s)
    main = s[(s.split_kind == "chronological") & (s.part.isin(["calibration", "test"]))]
    pretty = pd.DataFrame({
        "Model": main.model, "Features": main.feature_set, "Split part": main.part,
        "MAE": [pm(a, b, ".1f") for a, b in zip(main.mae_mean, main.mae_std)],
        "RMSE": [pm(a, b, ".1f") for a, b in zip(main.rmse_mean, main.rmse_std)],
        "R²": [pm(a, b, ".3f") for a, b in zip(main.r2_mean, main.r2_std)]})
    out.table("table2_point_models", "Table 2 – Point-model performance (chronological split, mean ± sd over seeds)",
              main, pretty, note="Models trained on the first 60% only. `no_time_id` removes yr, season and mnth.")


def tables_scenarios(out: Out, m: pd.DataFrame):
    cols_basic = [("coverage", "Coverage", ".3f"), ("width", "Width", ".1f"), ("interval_score", "Interval score", ".1f"),
                  ("worst_window_coverage", "Worst-window cov.", ".3f"), ("mae", "MAE", ".1f")]
    cols_shift = cols_basic[:4] + [("post_shift_coverage", "Post-shift cov. (500)", ".3f"),
                                   ("recovery_time", "Recovery (steps)", ".0f"), ("recovered", "Recovered frac.", ".2f"),
                                   ("detection_delay", "Detect. delay", ".0f")]
    specs = [("table3_no_shift_iid", "iid_control", "Table 3a – No shift, exchangeable control (random split membership)", cols_basic),
             ("table3_no_shift_natural", "natural", "Table 3b – No synthetic shift, chronological split (real temporal drift only)", cols_basic),
             ("table4_gradual", "gradual_covariate", "Table 4 – Gradual covariate drift", cols_shift),
             ("table5_abrupt", "abrupt_covariate", "Table 5 – Abrupt covariate shift", cols_shift),
             ("table5b_resample", "covariate_resample", "Table 5b – Pure covariate shift by X-dependent resampling", cols_shift),
             ("table6_noise", "noise", "Table 6 – Noise increase", cols_shift),
             ("table6b_conditional", "conditional", "Table 6b – Conditional shift (y × 1.5 on working days)", cols_shift + [("coverage_affected", "Cov. affected rows", ".3f"), ("coverage_unaffected_post", "Cov. unaffected post", ".3f")])]
    note = ("Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. "
            "Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 "
            "of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.")
    for name, sc, title, cols in specs:
        if sc not in set(m.scenario):
            continue
        s, pretty = pretty_summary(m, sc, METHOD_ORDER, cols)
        out.table(name, title, s, pretty, note)
    # compact overview: coverage & interval score for all scenarios
    rows = []
    for sc in [s for s in SCEN_LABEL if s in set(m.scenario)]:
        r = {"Scenario": SCEN_LABEL[sc]}
        g = m[m.scenario == sc]
        for meth in METHOD_ORDER:
            gg = g[g.method == meth]
            r[LABEL[meth]] = f"{gg.coverage.mean():.3f} / {gg.interval_score.mean():.0f}"
        rows.append(r)
    ov = pd.DataFrame(rows)
    out.table("table_overview", "Overview – coverage / mean interval score for every scenario (mean over seeds)", ov)


def table_stats(out: Out, m: pd.DataFrame):
    """Paired comparisons DA-ACP vs each baseline, unit = seed."""
    rows = []
    rng = np.random.default_rng(0)
    for sc in [s for s in SCEN_LABEL if s in set(m.scenario)]:
        g = m[m.scenario == sc].pivot_table(index="seed", columns="method", values=["interval_score", "coverage", "worst_window_coverage"])
        for base in ["static", "rolling", "weighted", "aci"]:
            for metric in ["interval_score", "coverage", "worst_window_coverage"]:
                a, b = g[metric]["da_acp"].to_numpy(), g[metric][base].to_numpy()
                d, lo, hi = paired_bootstrap_ci(a, b, rng=rng)
                rows.append({"scenario": sc, "baseline": base, "metric": metric, "mean_diff(da_acp-baseline)": d,
                             "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_p": wilcoxon_p(a, b), "n_seeds": len(a),
                             "da_acp_better_in": int(np.sum(a < b)) if metric == "interval_score" else int(np.sum(a > b))})
    df = pd.DataFrame(rows)
    view = df[df.metric == "interval_score"].copy()
    view["scenario"] = view.scenario.map(SCEN_LABEL)
    out.table("table_stats_paired", "Paired comparison of DA-ACP vs baselines (interval score; unit = seed)", df,
              view.drop(columns="metric"),
              note="Negative difference = DA-ACP has lower (better) interval score. 95% paired bootstrap CI over seeds; "
                   "two-sided Wilcoxon signed-rank p-value (descriptive; no multiplicity correction). "
                   "'da_acp_better_in' = number of seeds where DA-ACP was better. Note: the seeds share the same test "
                   "period, so seed-to-seed variation reflects model/shift randomness, not new data.")


def table_ablation(out: Out, ab: pd.DataFrame | None):
    if ab is None:
        return
    order = list(dict.fromkeys(ab.method))
    s = summarize(ab, ["method", "scenario"], ["coverage", "width", "interval_score", "worst_window_coverage",
                                                "post_shift_coverage", "recovery_time", "n_alarms", "detection_delay"])
    out.table("table7_ablation_full", "Table 7 (full) – Ablation per scenario", s)
    # relative interval score vs full DA-ACP, averaged over scenarios
    piv = ab.pivot_table(index=["scenario", "seed"], columns="method", values="interval_score")
    rel = piv.div(piv["da_acp_full"], axis=0)
    cov = ab.groupby("method").coverage.mean()
    wwc = ab.groupby("method").worst_window_coverage.mean()
    shifted = ab[ab.scenario != "natural"]
    rows = []
    for meth in order:
        r = {"Variant": meth, "Mean coverage": cov[meth], "Mean worst-window cov.": wwc[meth],
             "Rel. interval score (vs full)": rel[meth].mean(),
             "Seeds×scen. better than full": f"{int((rel[meth] < 1 - 1e-12).sum())}/{rel[meth].notna().sum()}",
             "Mean #alarms": ab[ab.method == meth].n_alarms.mean(),
             "Post-shift cov.": shifted[shifted.method == meth].post_shift_coverage.mean()}
        for sc in ["abrupt_covariate", "noise", "conditional"]:
            r[f"IS {sc.split('_')[0]}"] = ab[(ab.method == meth) & (ab.scenario == sc)].interval_score.mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    out.table("table7_ablation", "Table 7 – Ablation study (averaged over 6 scenarios × seeds)", t,
              note="Relative interval score = variant / full DA-ACP on the same (scenario, seed), then averaged; < 1 means "
                   "the variant is better. A: no drift detector; B: box kernel (hard window) instead of exponential "
                   "weights; C: frozen buffer refreshed only after alarms; D: decay λ0 small/medium/large; E: buffer length.")
    return t


def table_slow_base(out: Out, sb: pd.DataFrame | None):
    """Follow-up ablation: does the drift detector help when the baseline decay is slow?"""
    if sb is None:
        return
    rows = []
    rng = np.random.default_rng(1)
    for lam in sorted({float(x.split("_l")[-1]) for x in sb.method}):
        w = sb[sb.method == f"weighted_l{lam}"].set_index(["scenario", "seed"])
        for var in [f"da_acp_l{lam}", f"da_acp_reset_l{lam}"]:
            d = sb[sb.method == var].set_index(["scenario", "seed"])
            for sc in ["all"] + list(dict.fromkeys(sb.scenario)):
                a = d if sc == "all" else d.xs(sc, level="scenario", drop_level=False)
                b = w.loc[a.index]
                rel = (a.interval_score / b.interval_score).to_numpy()
                diff, lo, hi = paired_bootstrap_ci(a.interval_score.to_numpy(), b.interval_score.to_numpy(), rng=rng)
                rows.append({"lambda0": f"{lam:g}", "variant": "detector" if "reset" not in var else "detector+partial reset",
                             "scenario": sc, "rel_IS_vs_weighted": rel.mean(), "IS_diff": diff, "ci95_lo": lo, "ci95_hi": hi,
                             "cov_weighted": b.coverage.mean(), "cov_variant": a.coverage.mean(),
                             "wwc_weighted": b.worst_window_coverage.mean(), "wwc_variant": a.worst_window_coverage.mean(),
                             "post_cov_weighted": b.post_shift_coverage.mean(), "post_cov_variant": a.post_shift_coverage.mean(),
                             "better_in": f"{int((rel < 1).sum())}/{len(rel)}"})
    t = pd.DataFrame(rows)
    out.table("table7b_ablation_slow_base", "Table 7b – Follow-up: value of the drift detector when baseline weighting is slow",
              t, t[t.scenario == "all"].drop(columns="scenario"),
              note="Paired over (scenario, seed): DA-ACP with λ0 = λ, λ_fast = 0.02 vs exponentially weighted conformal with the "
                   "same λ and no detector. rel_IS < 1 means the detector helps. Full per-scenario rows in the CSV.")


def table_sensitivity(out: Out, se: pd.DataFrame | None):
    if se is None:
        return
    se = se.copy()
    se["value"] = se["value"].astype(str)
    s = summarize(se, ["factor", "value", "scenario", "method"], ["coverage", "width", "interval_score", "worst_window_coverage", "n_infinite"])
    out.table("table8_sensitivity_full", "Table 8 (full) – Sensitivity analysis", s)
    # compact: alpha and cal size
    rows = []
    for fac in ["alpha", "cal_size"]:
        g = s[s.factor == fac]
        for (val, meth), gg in g.groupby(["value", "method"], sort=False):
            rows.append({"Factor": fac, "Value": val, "Method": LABEL.get(meth, meth),
                         "Nominal": 1 - float(val) if fac == "alpha" else 0.9,
                         "Coverage (4 scen.)": gg.coverage_mean.mean(), "Worst-window": gg.worst_window_coverage_mean.mean(),
                         "Width": gg.width_mean.mean(), "#inf thresholds": gg.n_infinite_mean.sum()})
    t = pd.DataFrame(rows)
    out.table("table8_sensitivity", "Table 8 – Sensitivity to nominal level and calibration size "
              "(averaged over natural, abrupt, noise, conditional scenarios)", t)


def table_generalisation(out: Out, base: pd.DataFrame | None):
    if base is None:
        return
    c = base[base.table == "conformal"]
    s = summarize(c, ["model", "feature_set", "scenario", "method"], ["coverage", "width", "interval_score", "worst_window_coverage", "mae"])
    out.table("table9_generalisation_full", "Table 9 (full) – Cross-model generalisation", s)
    main = s[s.feature_set == "full"]
    rows = []
    for (model, sc), g in main.groupby(["model", "scenario"], sort=False):
        r = {"Model": model, "Scenario": SCEN_LABEL.get(sc, sc)}
        best_is = g[g.method != "point"].interval_score_mean.min()
        for meth in ["static", "rolling", "weighted", "aci", "da_acp"]:
            x = g[g.method == meth]
            if len(x):
                star = "*" if np.isclose(x.interval_score_mean.iloc[0], best_is) else ""
                r[LABEL[meth]] = f"{x.coverage_mean.iloc[0]:.3f} / {x.interval_score_mean.iloc[0]:.0f}{star}"
        rows.append(r)
    out.table("table9_generalisation", "Table 9 – Coverage / interval score across base regressors (mean over seeds; * = best IS)",
              pd.DataFrame(rows))
    ft = s[s.scenario.isin(["iid_control", "natural"]) & s.method.isin(["point", "static", "da_acp"])]
    out.table("table10_time_features", "Table 10 – Effect of removing time-identifying features (yr, season, mnth)",
              ft[["model", "feature_set", "scenario", "method", "coverage_mean", "width_mean", "interval_score_mean", "mae_mean"]])


# ============================================================================ figures
def fig_intervals(out: Out, root: Path):
    p = root / "raw" / "shift_experiments" / "predictions.csv.gz"
    if not p.exists():
        return
    pr = pd.read_csv(p)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharey=True)
    for ax, sc, title in [(axes[0], "iid_control", "IID control (random split) – 7 consecutive test days"),
                          (axes[1], "natural", "Chronological split, no synthetic shift – first 7 test days")]:
        sub = pr[(pr.experiment == sc) & (pr.seed == pr.seed.min())]
        for meth, alpha_fill in [("static", 0.25), ("da_acp", 0.0)]:
            s = sub[sub.method == meth].iloc[:168]
            x = np.arange(len(s))
            if meth == "static":
                ax.fill_between(x, s.lower, s.upper, color=COLOR[meth], alpha=alpha_fill, label="Static split 90% interval", step="mid")
            else:
                ax.plot(x, s.lower, color=COLOR[meth], lw=0.9, label="DA-ACP bounds")
                ax.plot(x, s.upper, color=COLOR[meth], lw=0.9)
        s = sub[sub.method == "static"].iloc[:168]
        x = np.arange(len(s))
        ax.plot(x, s.y_pred, color="k", lw=0.8, ls=":", label="prediction")
        miss = s.covered.to_numpy() == 0
        ax.scatter(x[~miss], s.y_true[~miss], s=6, color="tab:blue", label="observed (covered by static)", zorder=3)
        ax.scatter(x[miss], s.y_true[miss], s=14, marker="x", color="tab:red", label="observed (missed by static)", zorder=4)
        ax.set_title(title)
        ax.set_ylabel("cnt")
    axes[1].set_xlabel("hour index in test stream")
    axes[0].legend(ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout()
    out.fig(fig, "fig1_intervals_no_shift")


def _traces(tr, sc, meth, key, seeds):
    arrs = [tr[f"{sc}|{s}|{meth}|{key}"] for s in seeds if f"{sc}|{s}|{meth}|{key}" in tr]
    return arrs


def fig_rolling_gradual(out: Out, m, tr, seeds, window):
    sc = "gradual_covariate"
    if tr is None:
        return
    fig, ax = plt.subplots(figsize=(9, 3.4))
    meths = ["static", "rolling", "weighted", "aci", "da_acp"]
    for meth in meths:
        rc = np.array([rolling_mean(c.astype(float), window) for c in _traces(tr, sc, meth, "covered", seeds)])
        ax.plot(np.nanmean(rc, axis=0), color=COLOR[meth], lw=1.1, label=LABEL[meth])
    meta = m[(m.scenario == sc)].iloc[0]
    n = int(meta.n_test)
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.axvspan(0.1 * n, 0.9 * n, color="orange", alpha=0.08, label="drift ramp")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("test step (hours)")
    ax.set_ylabel(f"{window}-step rolling coverage")
    ax.set_title("Gradual covariate drift (temp, atemp, hum +1.5 sd over the ramp) – mean over seeds")
    ax.legend(ncol=6, fontsize=7, loc="lower left")
    out.fig(fig, "fig2_rolling_coverage_gradual")


def fig_rolling_abrupt(out: Out, m, tr, seeds, window, root: Path):
    sc = "abrupt_covariate"
    if tr is None:
        return
    seed = seeds[0]
    cp = int(m[(m.scenario == sc) & (m.seed == seed)].change_point.iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, sc2, title in [(axes[0], "abrupt_covariate", "Abrupt covariate shift"), (axes[1], "noise", "Noise increase")]:
        cp = int(m[(m.scenario == sc2) & (m.seed == seed)].change_point.iloc[0])
        for meth in ["static", "rolling", "weighted", "aci", "da_acp"]:
            c = tr[f"{sc2}|{seed}|{meth}|covered"].astype(float)
            ax.plot(rolling_mean(c, window), color=COLOR[meth], lw=1.0, label=LABEL[meth])
        alarms = np.flatnonzero(tr[f"{sc2}|{seed}|da_acp|alarm"])
        ax.axvline(cp, color="k", lw=1.5, label="true change point")
        for i, a in enumerate(alarms):
            ax.axvline(a, color=COLOR["da_acp"], ls=":", lw=1.0, label="DA-ACP drift alarm" if i == 0 else None)
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set_ylim(0.3, 1.02)
        ax.set_ylabel(f"{window}-step rolling cov.")
        ax.set_title(f"{title} (seed {seed})")
    axes[0].legend(ncol=4, fontsize=7, loc="lower left")
    axes[1].set_xlabel("test step (hours)")
    out.fig(fig, "fig3_rolling_coverage_abrupt")


def fig_width(out: Out, m, tr, seeds):
    if tr is None:
        return
    seed = seeds[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, sc in zip(axes, ["noise", "conditional"]):
        cp = int(m[(m.scenario == sc) & (m.seed == seed)].change_point.iloc[0])
        for meth in ["static", "rolling", "weighted", "aci", "da_acp"]:
            w = tr[f"{sc}|{seed}|{meth}|width"].astype(float)
            ax.plot(w, color=COLOR[meth], lw=0.8, label=LABEL[meth], alpha=0.9)
        ax.axvline(cp, color="k", lw=1.2)
        ax.set_title(f"Interval width over time – {SCEN_LABEL[sc]} (seed {seed})")
        ax.set_xlabel("test step")
        ax.set_ylabel("width = 2q")
    axes[0].legend(fontsize=7)
    out.fig(fig, "fig4_width_over_time")


def fig_cov_vs_width(out: Out, m):
    scs = [s for s in SCEN_LABEL if s in set(m.scenario)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, sc in zip(axes.ravel(), scs):
        g = m[(m.scenario == sc) & m.method.isin(METHOD_ORDER)]
        for meth in METHOD_ORDER:
            x = g[g.method == meth]
            ax.errorbar(x.width.mean(), x.coverage.mean(), xerr=x.width.std(), yerr=x.coverage.std(), fmt="o",
                        color=COLOR[meth], ms=5, capsize=2, label=LABEL[meth])
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set_title(SCEN_LABEL[sc])
        ax.set_xlabel("mean width")
        ax.set_ylabel("coverage")
    axes.ravel()[-1].axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center")
    fig.suptitle("Coverage vs mean interval width (mean ± sd over seeds)")
    fig.tight_layout()
    out.fig(fig, "fig5_coverage_vs_width")


def fig_aligned(out: Out, m, tr, seeds, window):
    if tr is None:
        return
    scs = ["abrupt_covariate", "covariate_resample", "noise", "conditional"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    pre, post = 300, 1200
    for ax, sc in zip(axes, scs):
        for meth in ["static", "rolling", "weighted", "aci", "da_acp"]:
            curves = []
            for s in seeds:
                key = f"{sc}|{s}|{meth}|covered"
                if key not in tr:
                    continue
                cp = int(m[(m.scenario == sc) & (m.seed == s)].change_point.iloc[0])
                c = tr[key].astype(float)
                # centred rolling coverage, aligned at the change point
                # centred rolling mean with NaN (not zero padding) where the window is incomplete
                rc = np.full(len(c), np.nan)
                trailing = rolling_mean(c, window)
                h = window // 2
                rc[: len(c) - h] = trailing[h:] if h else trailing
                seg = np.full(pre + post, np.nan)
                lo, hi = max(0, cp - pre), min(len(c), cp + post)
                seg[lo - (cp - pre): hi - (cp - pre)] = rc[lo:hi]
                curves.append(seg)
            if curves:
                ax.plot(np.arange(-pre, post), np.nanmean(curves, axis=0), color=COLOR[meth], lw=1.1, label=LABEL[meth])
        ax.axvline(0, color="k", lw=1)
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set_title(SCEN_LABEL[sc])
        ax.set_xlabel("steps relative to change point")
    axes[0].set_ylabel(f"centred {window}-step coverage")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Coverage trajectory aligned at the change point (mean over seeds)")
    fig.tight_layout()
    out.fig(fig, "fig6_aligned_recovery")


def fig_sensitivity(out: Out, se: pd.DataFrame | None):
    if se is None:
        return
    se = se.copy()
    # Figure 7: window sensitivity
    g = se[se.factor == "window"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, sc in zip(axes, ["natural", "abrupt_covariate", "noise", "conditional"]):
        for meth in ["rolling", "da_acp"]:
            x = g[(g.scenario == sc) & (g.method == meth)].groupby("value").agg(c=("coverage", "mean"), i=("interval_score", "mean"))
            x.index = x.index.astype(float)
            x = x.sort_index()
            ax.plot(x.index, x.i, "o-", color=COLOR[meth], label=f"{LABEL[meth]} (IS)")
            ax2 = ax.twinx() if meth == "rolling" else ax2
            ax2.plot(x.index, x.c, "s--", color=COLOR[meth], alpha=0.6, label=f"{LABEL[meth]} (coverage)")
            ax2.set_ylim(0.6, 1.0)
        ax.set_xscale("log")
        ax.set_title(SCEN_LABEL[sc])
        ax.set_xlabel("window W / DA-ACP buffer")
    axes[0].set_ylabel("interval score (solid)")
    ax2.set_ylabel("coverage (dashed)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out.fig(fig, "fig7_sensitivity_window")

    # Figure 8: shift magnitude
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, (fac, xl) in enumerate([("covariate_magnitude", "covariate shift (training sd)"),
                                   ("noise_magnitude", "noise sd / training target sd"),
                                   ("conditional_factor", "conditional factor c")]):
        g = se[se.factor == fac]
        for meth in ["static", "rolling", "weighted", "aci", "da_acp", "da_acp_no_drift"]:
            x = g[g.method == meth].groupby("value").agg(c=("coverage", "mean"), cs=("coverage", "std"),
                                                          w=("worst_window_coverage", "mean"), i=("interval_score", "mean"))
            x.index = x.index.astype(float)
            x = x.sort_index()
            ls = ":" if meth == "da_acp_no_drift" else "-"
            axes[0, j].plot(x.index, x.c, "o" + ls, color=COLOR[meth], label=LABEL[meth], ms=4)
            axes[1, j].plot(x.index, x.w, "o" + ls, color=COLOR[meth], label=LABEL[meth], ms=4)
        axes[0, j].axhline(0.9, color="k", ls="--", lw=0.8)
        axes[0, j].set_title(fac.replace("_", " "))
        axes[1, j].set_xlabel(xl)
    axes[0, 0].set_ylabel("overall coverage")
    axes[1, 0].set_ylabel("worst-window coverage")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out.fig(fig, "fig8_sensitivity_magnitude")

    # extra: decay + PH threshold + calibration size + alpha
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    g = se[se.factor == "decay"]
    for meth in ["weighted", "da_acp"]:
        x = g[g.method == meth].groupby("value").agg(i=("interval_score", "mean"), c=("coverage", "mean"))
        x.index = x.index.astype(float)
        axes[0].plot(x.sort_index().index, x.sort_index().i, "o-", color=COLOR[meth], label=LABEL[meth])
    axes[0].set_xscale("log"); axes[0].set_xlabel("decay λ (λ0 for DA-ACP)"); axes[0].set_ylabel("mean IS (4 scenarios)")
    axes[0].legend(fontsize=7)
    g = se[(se.factor == "ph_threshold") & (se.method == "da_acp")]
    x = g.groupby("value").agg(i=("interval_score", "mean"), a=("n_alarms", "mean"), w=("worst_window_coverage", "mean"))
    x.index = x.index.astype(float)
    x = x.sort_index()
    labels = [("∞" if v > 1e8 else f"{v:g}") for v in x.index]
    axes[1].plot(range(len(x)), x.i, "o-", color=COLOR["da_acp"])
    axes[1].set_xticks(range(len(x))); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Page-Hinkley threshold"); axes[1].set_ylabel("mean IS")
    b = axes[1].twinx(); b.bar(range(len(x)), x.a, alpha=0.2, color="k"); b.set_ylabel("mean #alarms")
    g = se[se.factor == "cal_size"]
    for meth in ["static", "rolling", "weighted", "aci", "da_acp"]:
        x = g[g.method == meth].groupby("value").agg(c=("coverage", "mean"))
        order = sorted(x.index, key=lambda v: 1e9 if v == "full" else float(v))
        axes[2].plot(range(len(order)), x.loc[order].c, "o-", color=COLOR[meth], label=LABEL[meth])
    axes[2].set_xticks(range(len(order))); axes[2].set_xticklabels(order)
    axes[2].axhline(0.9, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("calibration size (most recent n)"); axes[2].set_ylabel("coverage")
    g = se[se.factor == "alpha"]
    for meth in ["static", "rolling", "weighted", "aci", "da_acp"]:
        x = g[g.method == meth].groupby("value").agg(c=("coverage", "mean"))
        x.index = x.index.astype(float)
        axes[3].plot(1 - x.index, x.c, "o-", color=COLOR[meth], label=LABEL[meth])
    axes[3].plot([0.78, 0.97], [0.78, 0.97], "k--", lw=0.8)
    axes[3].set_xlabel("nominal coverage"); axes[3].set_ylabel("empirical coverage")
    fig.tight_layout()
    out.fig(fig, "fig10_sensitivity_hparams")


def fig_ablation(out: Out, ab: pd.DataFrame | None):
    if ab is None:
        return
    piv = ab.pivot_table(index=["scenario", "seed"], columns="method", values="interval_score")
    rel = piv.div(piv["da_acp_full"], axis=0)
    order = [c for c in dict.fromkeys(ab.method) if c != "da_acp_full"]
    scs = list(dict.fromkeys(ab.scenario))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    mat = np.array([[rel[o].xs(sc, level="scenario").mean() for sc in scs] for o in order])
    im = axes[0].imshow(np.log2(mat), cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=7)
    axes[0].set_xticks(range(len(scs))); axes[0].set_xticklabels([SCEN_LABEL[s].split(":")[0] for s in scs])
    for i in range(len(order)):
        for j in range(len(scs)):
            axes[0].text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6)
    axes[0].set_title("Interval score relative to full DA-ACP (>1 = worse)")
    plt.colorbar(im, ax=axes[0], label="log2 ratio")
    cov = ab.groupby("method").coverage.mean()
    wwc = ab.groupby("method").worst_window_coverage.mean()
    for o in ["da_acp_full"] + order:
        col = "k" if o == "da_acp_full" else ("tab:red" if o.startswith(("A_", "B_", "C_")) else "tab:gray")
        axes[1].scatter(cov[o], wwc[o], color=col, s=18)
        axes[1].annotate(o, (cov[o], wwc[o]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    axes[1].set_xlabel("mean coverage"); axes[1].set_ylabel("mean worst-window coverage")
    axes[1].set_title("Ablation variants (avg. over scenarios × seeds)")
    fig.tight_layout()
    out.fig(fig, "fig9_ablation")


def fig_generalisation(out: Out, base: pd.DataFrame | None):
    if base is None:
        return
    c = base[(base.table == "conformal") & (base.feature_set == "full") & (base.method != "point")]
    scs = [s for s in SCEN_LABEL if s in set(c.scenario)]
    models = list(dict.fromkeys(c.model))
    fig, axes = plt.subplots(1, len(scs), figsize=(3.2 * len(scs), 3.2), sharey=True)
    meths = ["static", "rolling", "weighted", "aci", "da_acp"]
    for ax, sc in zip(axes, scs):
        for k, meth in enumerate(meths):
            vals = [c[(c.scenario == sc) & (c.model == mo) & (c.method == meth)].coverage.mean() for mo in models]
            ax.bar(np.arange(len(models)) + (k - 2) * 0.16, vals, width=0.16, color=COLOR[meth], label=LABEL[meth])
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, fontsize=7)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(SCEN_LABEL[sc], fontsize=8)
    axes[0].set_ylabel("coverage")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    out.fig(fig, "fig11_generalisation_models")


def fig_detection(out: Out, m: pd.DataFrame):
    g = m[(m.method == "da_acp") & m.change_point.notna()]
    fig, ax = plt.subplots(figsize=(7, 3))
    scs = [s for s in SCEN_LABEL if s in set(g.scenario)]
    data = [g[g.scenario == s].detection_delay.dropna().to_numpy() for s in scs]
    ax.boxplot(data, tick_labels=[SCEN_LABEL[s].split(":")[0] for s in scs], showfliers=True)
    for i, s in enumerate(scs):
        nd = g[g.scenario == s].detection_delay.isna().sum()
        fa = g[g.scenario == s].false_alarms.mean()
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f"missed {nd}\nFA {fa:.1f}", ha="center", fontsize=7)
    ax.set_ylabel("detection delay (steps)")
    ax.set_title("Page-Hinkley detection delay of DA-ACP (per seed)")
    out.fig(fig, "fig12_detection_delay")


# ============================================================================ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    cfg = load_full_config(args.config)
    root = resolve_path(args.output or cfg["experiment"]["output_dir"])
    out = Out(root)
    window = cfg["evaluation"]["rolling_window"]

    table_dataset(out, cfg)
    m, meta, tr = load_raw(root, "shift_experiments")
    base, _, _ = load_raw(root, "baselines")
    ab, _, _ = load_raw(root, "ablation")
    se, _, _ = load_raw(root, "sensitivity")
    table_point_models(out, base)
    if m is not None:
        seeds = sorted(set(m.seed))
        tables_scenarios(out, m)
        table_stats(out, m)
        fig_intervals(out, root)
        fig_rolling_gradual(out, m, tr, seeds, window)
        fig_rolling_abrupt(out, m, tr, seeds, window, root)
        fig_width(out, m, tr, seeds)
        fig_cov_vs_width(out, m)
        fig_aligned(out, m, tr, seeds, window)
        fig_detection(out, m)
    table_ablation(out, ab)
    fig_ablation(out, ab)
    sb, _, _ = load_raw(root, "ablation_slow_base")
    table_slow_base(out, sb)
    table_sensitivity(out, se)
    fig_sensitivity(out, se)
    table_generalisation(out, base)
    fig_generalisation(out, base)
    tun = root / "tuning" / "tuning_results.csv"
    if tun.exists():
        t = pd.read_csv(tun)
        out.table("table_tuning", "Development-split hyperparameter selection (top 3 per family)",
                  t.groupby("family").head(3), note="Selected on the nested development split inside the training period; never on test data.", floatfmt=".4f")
    (out.tables / "ALL_TABLES.md").write_text("# All tables\n\n" + "\n\n".join(out.md))
    print(f"wrote {len(out.md)} tables to {out.tables} and figures to {out.figs}")


if __name__ == "__main__":
    main()
