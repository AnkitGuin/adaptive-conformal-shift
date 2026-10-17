### Table 5 – Abrupt covariate shift

| Method | Coverage | Width | Interval score | Worst-window cov. | Post-shift cov. (500) | Recovery (steps) | Recovered frac. | Detect. delay |
|---|---|---|---|---|---|---|---|---|
| Static split | 0.810 ± 0.007 | 300.8 ± 6.2 | 627.0 ± 19.9 | 0.565 ± 0.018 | 0.644 ± 0.022 | 944 ± 86 | 1.00 ± 0.00 | – |
| Rolling window | 0.896 ± 0.002 | 409.9 ± 7.7 | 537.4 ± 10.4 | 0.854 ± 0.003 | 0.898 ± 0.006 | 119 ± 29 | 1.00 ± 0.00 | – |
| Exp. weighted | 0.909 ± 0.002 | 427.5 ± 9.1 | 538.3 ± 10.6 | 0.868 ± 0.008 | 0.908 ± 0.005 | 127 ± 24 | 1.00 ± 0.00 | – |
| ACI | 0.900 ± 0.000 | 430.6 ± 7.0 | 544.9 ± 8.3 | 0.878 ± 0.002 | 0.901 ± 0.003 | 99 ± 0 | 1.00 ± 0.00 | – |
| DA-ACP | 0.906 ± 0.001 | 425.5 ± 9.2 | 542.1 ± 10.8 | 0.856 ± 0.005 | 0.904 ± 0.006 | 137 ± 32 | 1.00 ± 0.00 | 22 ± 17 |
| CQR (static) | 0.857 ± 0.006 | 378.1 ± 6.6 | 610.1 ± 11.7 | 0.608 ± 0.026 | 0.707 ± 0.029 | 1030 ± 102 | 1.00 ± 0.00 | – |
| CQR + DA-ACP | 0.916 ± 0.001 | 442.5 ± 6.8 | 536.7 ± 8.2 | 0.869 ± 0.005 | 0.910 ± 0.006 | 114 ± 25 | 1.00 ± 0.00 | 37 ± 28 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
