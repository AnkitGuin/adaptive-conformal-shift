### Table 3a – No shift, exchangeable control (random split membership)

| Method | Coverage | Width | Interval score | Worst-window cov. | MAE |
|---|---|---|---|---|---|
| Static split | 0.900 ± 0.005 | 131.1 ± 2.6 | 209.2 ± 4.1 | 0.792 ± 0.016 | 28.0 ± 0.3 |
| Rolling window | 0.899 ± 0.003 | 136.1 ± 2.0 | 206.9 ± 4.3 | 0.866 ± 0.007 | 28.0 ± 0.3 |
| Exp. weighted | 0.913 ± 0.002 | 141.6 ± 2.4 | 204.8 ± 3.4 | 0.876 ± 0.007 | 28.0 ± 0.3 |
| ACI | 0.900 ± 0.001 | 150.2 ± 5.7 | 218.8 ± 6.2 | 0.874 ± 0.004 | 28.0 ± 0.3 |
| DA-ACP | 0.908 ± 0.002 | 137.1 ± 2.9 | 204.4 ± 3.6 | 0.860 ± 0.008 | 28.0 ± 0.3 |
| CQR (static) | 0.900 ± 0.005 | 141.6 ± 2.2 | 179.6 ± 2.2 | 0.847 ± 0.008 | 28.0 ± 0.3 |
| CQR + DA-ACP | 0.908 ± 0.002 | 145.1 ± 2.0 | 180.2 ± 2.2 | 0.872 ± 0.005 | 28.0 ± 0.3 |

Mean ± sd over seeds; nominal coverage 0.90. Worst-window = minimum 250-step rolling coverage. Recovery = steps after the change point until the 100-step post-change rolling coverage stays within ±0.05 of 0.90 for 50 steps (mean over runs that recovered; see 'Recovered frac.'). Minimum possible recovery is 99 steps.
