"""Builds notebooks/exploratory_analysis.ipynb (kept as code so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s))  # noqa: E731
md("# Exploratory analysis – Bike Sharing (hourly)\n\nEDA only: this notebook makes **no modelling decisions** "
   "and is not part of the experimental pipeline. It uses the same loader and split code as the experiments.")
code("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
     "from src.pipeline import load_full_config, get_dataframe\nfrom src.data.preprocess import chronological_split, dataset_statistics\n"
     "cfg = load_full_config('configs/default.yaml')\ndf = get_dataframe(cfg)\nsplit = chronological_split(len(df), cfg['split'])\ndf.head()")
md("## Split statistics\nThe test period (Aug–Dec 2012) has a higher mean and variance than the training period: "
   "the data are **not** exchangeable across the chronological split.")
code("dataset_statistics(df, split, 'cnt')")
md("## Hourly profile: working vs non-working days\nStrong, regime-dependent daily pattern (commuting peaks on working days).")
code("fig, ax = plt.subplots(figsize=(7,3))\nfor wd, g in df.groupby('workingday'):\n    ax.plot(g.groupby('hr').cnt.mean(), label=f'workingday={wd}')\nax.legend(); ax.set_xlabel('hour'); ax.set_ylabel('mean cnt'); plt.show()")
md("## Heteroscedasticity\nThe spread of `cnt` grows with its level → symmetric constant-width intervals are inefficient; motivates CQR.")
code("g = df.groupby('hr').cnt.agg(['mean','std'])\nplt.figure(figsize=(4,3)); plt.scatter(g['mean'], g['std']); plt.xlabel('mean cnt by hour'); plt.ylabel('sd'); plt.show()")
md("## Temporal gaps and trend")
code("gaps = df.timestamp.diff().dt.total_seconds().div(3600)\nprint('missing hours:', int((gaps-1).clip(lower=0).sum()))\n"
     "df.set_index('timestamp').cnt.resample('W').mean().plot(figsize=(8,3), title='weekly mean cnt'); plt.show()")
md("## Time-identifying variables\n`yr` alone separates 2011 from 2012 and correlates with the upward trend; "
   "`season`/`mnth` encode position in the year. The experiments therefore include a `no_time_id` feature set.")
code("df.groupby('yr').cnt.describe()")
nb["cells"] = C
nbf.write(nb, "notebooks/exploratory_analysis.ipynb")
