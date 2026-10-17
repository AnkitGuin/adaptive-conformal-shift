### Table 6 – Noise increase

| Method | Coverage | Width | Interval score | Worst-window cov. | Post-shift cov. (500) | Recovery (steps) | Recovered frac. | Detect. delay |
|---|---|---|---|---|---|---|---|---|
| Static split | 0.802 ± 0.005 | 300.8 ± 6.2 | 573.7 ± 9.2 | 0.663 ± 0.020 | 0.715 ± 0.014 | 1488 ± 325 | 1.00 ± 0.00 | – |
| Rolling window | 0.904 ± 0.002 | 412.4 ± 6.6 | 516.1 ± 6.8 | 0.868 ± 0.007 | 0.905 ± 0.006 | 100 ± 3 | 1.00 ± 0.00 | – |
| Exp. weighted | 0.913 ± 0.001 | 425.4 ± 7.0 | 516.2 ± 6.8 | 0.879 ± 0.006 | 0.910 ± 0.004 | 108 ± 21 | 1.00 ± 0.00 | – |
| ACI | 0.900 ± 0.000 | 421.1 ± 8.3 | 529.4 ± 7.0 | 0.879 ± 0.002 | 0.900 ± 0.003 | 99 ± 0 | 1.00 ± 0.00 | – |
| DA-ACP | 0.908 ± 0.002 | 417.3 ± 6.9 | 515.0 ± 6.6 | 0.859 ± 0.007 | 0.903 ± 0.006 | 114 ± 26 | 1.00 ± 0.00 | 70 ± 47 |
| CQR (static) | 0.808 ± 0.005 | 356.1 ± 6.2 | 626.0 ± 7.3 | 0.643 ± 0.019 | 0.708 ± 0.014 | 1368 ± 225 | 1.00 ± 0.00 | – |
| CQR + DA-ACP | 0.915 ± 0.002 | 472.5 ± 6.1 | 569.1 ± 7.6 | 0.871 ± 0.007 | 0.908 ± 0.004 | 105 ± 19 | 1.00 ± 0.00 | 89 ± 144 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
