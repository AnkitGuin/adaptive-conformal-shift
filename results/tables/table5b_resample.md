### Table 5b – Pure covariate shift by X-dependent resampling

| Method | Coverage | Width | Interval score | Worst-window cov. | Post-shift cov. (500) | Recovery (steps) | Recovered frac. | Detect. delay |
|---|---|---|---|---|---|---|---|---|
| Static split | 0.777 ± 0.005 | 300.8 ± 6.2 | 583.3 ± 9.6 | 0.540 ± 0.020 | 0.608 ± 0.019 | 762 ± 148 | 1.00 ± 0.00 | – |
| Rolling window | 0.902 ± 0.003 | 410.8 ± 9.1 | 497.8 ± 9.3 | 0.869 ± 0.007 | 0.897 ± 0.007 | 124 ± 31 | 1.00 ± 0.00 | – |
| Exp. weighted | 0.914 ± 0.003 | 423.3 ± 9.7 | 493.0 ± 8.7 | 0.877 ± 0.003 | 0.917 ± 0.005 | 125 ± 31 | 1.00 ± 0.00 | – |
| ACI | 0.902 ± 0.000 | 424.1 ± 10.4 | 508.1 ± 10.1 | 0.882 ± 0.002 | 0.905 ± 0.002 | 99 ± 0 | 1.00 ± 0.00 | – |
| DA-ACP | 0.909 ± 0.002 | 419.6 ± 9.3 | 495.3 ± 8.6 | 0.858 ± 0.006 | 0.912 ± 0.009 | 139 ± 35 | 1.00 ± 0.00 | 46 ± 27 |
| CQR (static) | 0.785 ± 0.006 | 388.0 ± 5.5 | 684.7 ± 10.9 | 0.563 ± 0.011 | 0.630 ± 0.022 | 634 ± 47 | 1.00 ± 0.00 | – |
| CQR + DA-ACP | 0.923 ± 0.001 | 518.9 ± 7.0 | 599.0 ± 7.0 | 0.875 ± 0.009 | 0.917 ± 0.006 | 107 ± 15 | 1.00 ± 0.00 | 25 ± 16 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
