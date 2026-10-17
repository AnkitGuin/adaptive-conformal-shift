### Table 7b – Follow-up: value of the drift detector when baseline weighting is slow

| lambda0 | variant | rel_IS_vs_weighted | IS_diff | ci95_lo | ci95_hi | cov_weighted | cov_variant | wwc_weighted | wwc_variant | post_cov_weighted | post_cov_variant | better_in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | detector | 0.965 | -26.139 | -35.064 | -18.014 | 0.871 | 0.897 | 0.703 | 0.800 | 0.756 | 0.868 | 60/60 |
| 0.0005 | detector+partial reset | 0.958 | -29.854 | -38.533 | -22.001 | 0.871 | 0.906 | 0.703 | 0.820 | 0.756 | 0.894 | 60/60 |
| 0.001 | detector | 0.976 | -17.489 | -23.339 | -12.150 | 0.882 | 0.902 | 0.728 | 0.814 | 0.783 | 0.876 | 57/60 |
| 0.001 | detector+partial reset | 0.972 | -19.529 | -25.211 | -14.128 | 0.882 | 0.909 | 0.728 | 0.825 | 0.783 | 0.899 | 60/60 |
| 0.002 | detector | 0.988 | -8.710 | -11.728 | -5.924 | 0.893 | 0.906 | 0.765 | 0.829 | 0.824 | 0.886 | 50/60 |
| 0.002 | detector+partial reset | 0.988 | -8.800 | -11.775 | -6.003 | 0.893 | 0.910 | 0.765 | 0.834 | 0.824 | 0.902 | 50/60 |

Paired over (scenario, seed): DA-ACP with λ0 = λ, λ_fast = 0.02 vs exponentially weighted conformal with the same λ and no detector. rel_IS < 1 means the detector helps. Full per-scenario rows in the CSV.
