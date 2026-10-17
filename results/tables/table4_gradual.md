### Table 4 – Gradual covariate drift

| Method | Coverage | Width | Interval score | Worst-window cov. | Post-shift cov. (500) | Recovery (steps) | Recovered frac. | Detect. delay |
|---|---|---|---|---|---|---|---|---|
| Static split | 0.844 ± 0.004 | 300.8 ± 6.2 | 510.4 ± 8.5 | 0.712 ± 0.010 | 0.834 ± 0.010 | 153 ± 79 | 1.00 ± 0.00 | – |
| Rolling window | 0.897 ± 0.001 | 369.1 ± 6.7 | 477.4 ± 7.7 | 0.853 ± 0.006 | 0.907 ± 0.005 | 100 ± 3 | 1.00 ± 0.00 | – |
| Exp. weighted | 0.909 ± 0.001 | 384.0 ± 7.3 | 476.9 ± 8.2 | 0.871 ± 0.007 | 0.908 ± 0.003 | 99 ± 0 | 1.00 ± 0.00 | – |
| ACI | 0.900 ± 0.000 | 377.6 ± 7.4 | 478.4 ± 8.5 | 0.879 ± 0.003 | 0.898 ± 0.001 | 99 ± 0 | 1.00 ± 0.00 | – |
| DA-ACP | 0.905 ± 0.001 | 378.7 ± 7.5 | 479.5 ± 8.2 | 0.857 ± 0.006 | 0.895 ± 0.004 | 99 ± 0 | 1.00 ± 0.00 | 487 ± 31 |
| CQR (static) | 0.872 ± 0.004 | 372.6 ± 6.1 | 560.3 ± 7.7 | 0.684 ± 0.016 | 0.810 ± 0.007 | 210 ± 104 | 1.00 ± 0.00 | – |
| CQR + DA-ACP | 0.917 ± 0.001 | 414.9 ± 5.3 | 505.9 ± 6.2 | 0.881 ± 0.004 | 0.907 ± 0.005 | 99 ± 0 | 1.00 ± 0.00 | 38 ± 7 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
