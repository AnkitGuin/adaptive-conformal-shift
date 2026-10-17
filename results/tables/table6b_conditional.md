### Table 6b – Conditional shift (y × 1.5 on working days)

| Method | Coverage | Width | Interval score | Worst-window cov. | Post-shift cov. (500) | Recovery (steps) | Recovered frac. | Detect. delay | Cov. affected rows | Cov. unaffected post |
|---|---|---|---|---|---|---|---|---|---|---|
| Static split | 0.682 ± 0.009 | 300.8 ± 6.2 | 1231.4 ± 40.7 | 0.425 ± 0.005 | 0.503 ± 0.014 | 2351 ± 76 | 0.40 ± 0.52 | – | 0.501 ± 0.008 | 0.847 ± 0.010 |
| Rolling window | 0.899 ± 0.002 | 642.0 ± 14.6 | 852.8 ± 19.2 | 0.844 ± 0.004 | 0.888 ± 0.011 | 136 ± 42 | 1.00 ± 0.00 | – | 0.854 ± 0.003 | 0.981 ± 0.005 |
| Exp. weighted | 0.916 ± 0.001 | 680.9 ± 15.2 | 847.2 ± 19.3 | 0.870 ± 0.008 | 0.904 ± 0.005 | 132 ± 32 | 1.00 ± 0.00 | – | 0.880 ± 0.002 | 0.996 ± 0.003 |
| ACI | 0.901 ± 0.000 | 635.2 ± 11.1 | 830.1 ± 16.9 | 0.872 ± 0.000 | 0.901 ± 0.004 | 99 ± 0 | 1.00 ± 0.00 | – | 0.865 ± 0.001 | 0.974 ± 0.002 |
| DA-ACP | 0.915 ± 0.001 | 681.3 ± 15.4 | 851.3 ± 19.7 | 0.855 ± 0.005 | 0.903 ± 0.005 | 134 ± 32 | 1.00 ± 0.00 | 24 ± 24 | 0.884 ± 0.002 | 0.996 ± 0.003 |
| CQR (static) | 0.683 ± 0.010 | 356.1 ± 6.2 | 1290.6 ± 41.7 | 0.408 ± 0.015 | 0.502 ± 0.020 | – | 0.00 ± 0.00 | – | 0.498 ± 0.014 | 0.882 ± 0.013 |
| CQR + DA-ACP | 0.919 ± 0.001 | 740.4 ± 13.3 | 921.1 ± 17.9 | 0.870 ± 0.005 | 0.900 ± 0.006 | 134 ± 33 | 1.00 ± 0.00 | 16 ± 17 | 0.883 ± 0.003 | 0.997 ± 0.002 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
