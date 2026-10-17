### Table 7 – Ablation study (averaged over 6 scenarios × seeds)

| Variant | Mean coverage | Mean worst-window cov. | Rel. interval score (vs full) | Seeds×scen. better than full | Mean #alarms | Post-shift cov. | IS abrupt | IS noise | IS conditional |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.793 | 0.602 | 1.172 | 0/60 | 0.000 | 0.661 | 626.996 | 573.688 | 1231.447 |
| da_acp_full | 0.909 | 0.857 | 1.000 | 0/60 | 10.333 | 0.903 | 542.066 | 514.986 | 851.301 |
| A_no_drift_detection | 0.906 | 0.848 | 0.999 | 40/60 | 0.000 | 0.893 | 542.085 | 514.519 | 852.347 |
| B_no_exp_weighting | 0.901 | 0.834 | 1.013 | 1/60 | 10.333 | 0.896 | 551.982 | 516.170 | 876.892 |
| C_no_rolling_updates | 0.934 | 0.874 | 1.180 | 0/60 | 10.333 | 0.929 | 629.422 | 600.076 | 941.254 |
| no_weight_boost | 0.906 | 0.848 | 0.999 | 40/60 | 10.333 | 0.893 | 542.085 | 514.519 | 852.347 |
| partial_reset | 0.911 | 0.859 | 1.003 | 7/60 | 10.333 | 0.910 | 544.623 | 515.453 | 855.095 |
| no_partial_reset | 0.909 | 0.857 | 1.000 | 0/60 | 10.333 | 0.903 | 542.066 | 514.986 | 851.301 |
| step_anneal | 0.909 | 0.858 | 0.999 | 42/60 | 10.333 | 0.906 | 540.475 | 515.401 | 848.223 |
| signed_residual_stream | 0.910 | 0.866 | 0.997 | 54/60 | 27.383 | 0.905 | 539.666 | 514.884 | 848.694 |
| detect_both_directions | 0.908 | 0.857 | 0.999 | 45/60 | 15.750 | 0.903 | 541.169 | 515.112 | 848.972 |
| D_decay_small | 0.875 | 0.717 | 1.057 | 0/60 | 10.333 | 0.769 | 582.365 | 531.901 | 935.260 |
| D_decay_medium | 0.897 | 0.790 | 1.019 | 0/60 | 10.333 | 0.847 | 557.454 | 518.819 | 875.000 |
| D_decay_large | 0.909 | 0.857 | 1.000 | 0/60 | 10.333 | 0.903 | 542.066 | 514.986 | 851.301 |
| E_buffer_100 | 0.909 | 0.871 | 1.001 | 28/60 | 10.333 | 0.909 | 539.991 | 516.917 | 854.674 |
| E_buffer_250 | 0.908 | 0.858 | 1.000 | 36/60 | 10.333 | 0.905 | 542.046 | 515.054 | 850.858 |
| E_buffer_500 | 0.909 | 0.858 | 1.000 | 23/60 | 10.333 | 0.904 | 542.150 | 515.036 | 851.289 |
| E_buffer_1000 | 0.909 | 0.857 | 1.000 | 36/60 | 10.333 | 0.903 | 542.065 | 514.986 | 851.302 |
| E_buffer_3000 | 0.909 | 0.857 | 1.000 | 0/60 | 10.333 | 0.903 | 542.066 | 514.986 | 851.301 |

Relative interval score = variant / full DA-ACP on the same (scenario, seed), then averaged; < 1 means the variant is better. A: no drift detector; B: box kernel (hard window) instead of exponential weights; C: frozen buffer refreshed only after alarms; D: decay λ0 small/medium/large; E: buffer length.
