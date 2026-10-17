### Table 3b – No synthetic shift, chronological split (real temporal drift only)

| Method | Coverage | Width | Interval score | Worst-window cov. | MAE |
|---|---|---|---|---|---|
| Static split | 0.841 ± 0.004 | 300.8 ± 6.2 | 498.0 ± 6.6 | 0.709 ± 0.009 | 76.0 ± 1.0 |
| Rolling window | 0.905 ± 0.001 | 372.3 ± 6.0 | 465.4 ± 6.0 | 0.866 ± 0.004 | 76.0 ± 1.0 |
| Exp. weighted | 0.915 ± 0.001 | 385.3 ± 6.5 | 465.1 ± 6.3 | 0.880 ± 0.005 | 76.0 ± 1.0 |
| ACI | 0.900 ± 0.000 | 373.2 ± 6.4 | 470.6 ± 6.6 | 0.877 ± 0.002 | 76.0 ± 1.0 |
| DA-ACP | 0.910 ± 0.001 | 378.7 ± 6.1 | 464.7 ± 6.1 | 0.859 ± 0.007 | 76.0 ± 1.0 |
| CQR (static) | 0.849 ± 0.004 | 356.1 ± 6.2 | 553.6 ± 7.6 | 0.696 ± 0.014 | 76.0 ± 1.0 |
| CQR + DA-ACP | 0.920 ± 0.001 | 427.0 ± 5.5 | 512.5 ± 6.1 | 0.875 ± 0.006 | 76.0 ± 1.0 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
