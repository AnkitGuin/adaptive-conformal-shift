### Paired comparison of DA-ACP vs baselines (interval score; unit = seed)

| scenario | baseline | mean_diff(da_acp-baseline) | ci95_lo | ci95_hi | wilcoxon_p | n_seeds | da_acp_better_in |
|---|---|---|---|---|---|---|---|
| A: IID control | static | -4.782 | -5.457 | -4.175 | 0.002 | 10 | 10 |
| A: IID control | rolling | -2.467 | -3.093 | -1.751 | 0.002 | 10 | 10 |
| A: IID control | weighted | -0.404 | -0.700 | -0.144 | 0.014 | 10 | 8 |
| A: IID control | aci | -14.412 | -17.074 | -11.966 | 0.002 | 10 | 10 |
| A': Natural (chronological) | static | -33.290 | -34.539 | -32.176 | 0.002 | 10 | 10 |
| A': Natural (chronological) | rolling | -0.605 | -1.012 | -0.213 | 0.020 | 10 | 8 |
| A': Natural (chronological) | weighted | -0.307 | -0.799 | 0.157 | 0.375 | 10 | 6 |
| A': Natural (chronological) | aci | -5.855 | -7.205 | -4.593 | 0.002 | 10 | 10 |
| B: Gradual covariate | static | -30.935 | -32.117 | -29.884 | 0.002 | 10 | 10 |
| B: Gradual covariate | rolling | 2.149 | 1.366 | 2.894 | 0.004 | 10 | 1 |
| B: Gradual covariate | weighted | 2.563 | 2.181 | 2.892 | 0.002 | 10 | 0 |
| B: Gradual covariate | aci | 1.066 | 0.032 | 2.014 | 0.084 | 10 | 2 |
| C: Abrupt covariate | static | -84.931 | -91.941 | -77.788 | 0.002 | 10 | 10 |
| C: Abrupt covariate | rolling | 4.661 | 3.846 | 5.401 | 0.002 | 10 | 0 |
| C: Abrupt covariate | weighted | 3.784 | 3.451 | 4.078 | 0.002 | 10 | 0 |
| C: Abrupt covariate | aci | -2.859 | -5.966 | 0.215 | 0.232 | 10 | 5 |
| C': Covariate resampling | static | -87.944 | -91.567 | -84.208 | 0.002 | 10 | 10 |
| C': Covariate resampling | rolling | -2.483 | -3.626 | -1.282 | 0.006 | 10 | 9 |
| C': Covariate resampling | weighted | 2.370 | 2.115 | 2.606 | 0.002 | 10 | 0 |
| C': Covariate resampling | aci | -12.758 | -15.599 | -9.938 | 0.002 | 10 | 10 |
| D: Noise increase | static | -58.702 | -62.086 | -55.462 | 0.002 | 10 | 10 |
| D: Noise increase | rolling | -1.162 | -1.841 | -0.485 | 0.027 | 10 | 7 |
| D: Noise increase | weighted | -1.209 | -1.789 | -0.706 | 0.002 | 10 | 10 |
| D: Noise increase | aci | -14.385 | -16.278 | -12.659 | 0.002 | 10 | 10 |
| E: Conditional shift | static | -380.145 | -393.596 | -365.966 | 0.002 | 10 | 10 |
| E: Conditional shift | rolling | -1.536 | -2.692 | -0.474 | 0.020 | 10 | 9 |
| E: Conditional shift | weighted | 4.062 | 3.722 | 4.378 | 0.002 | 10 | 0 |
| E: Conditional shift | aci | 21.195 | 18.604 | 23.873 | 0.002 | 10 | 0 |

Negative difference = DA-ACP has lower (better) interval score. 95% paired bootstrap CI over seeds; two-sided Wilcoxon signed-rank p-value (descriptive; no multiplicity correction). 'da_acp_better_in' = number of seeds where DA-ACP was better. Note: the seeds share the same test period, so seed-to-seed variation reflects model/shift randomness, not new data.
