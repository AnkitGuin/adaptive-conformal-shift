### Development-split hyperparameter selection (top 3 per family)

| family | method | rel_interval_score | coverage | worst_window_coverage | width |
|---|---|---|---|---|---|
| aci | aci|gamma=0.05 | 0.8544 | 0.8991 | 0.8600 | 364.5891 |
| aci | aci|gamma=0.01 | 0.8590 | 0.8960 | 0.8240 | 349.2941 |
| aci | aci|gamma=0.005 | 0.8670 | 0.8931 | 0.8098 | 346.3119 |
| da_acp | da_acp|decay=0.01,decay_fast=0.02,ph_threshold=30.0,pre_drift_weight=1.0 | 0.8553 | 0.9050 | 0.8578 | 363.3479 |
| da_acp | da_acp|decay=0.01,decay_fast=0.02,ph_threshold=15.0,pre_drift_weight=1.0 | 0.8565 | 0.9084 | 0.8644 | 368.7312 |
| da_acp | da_acp|decay=0.01,decay_fast=0.01,ph_threshold=30.0,pre_drift_weight=0.1 | 0.8577 | 0.9084 | 0.8587 | 373.2193 |
| rolling | rolling|window=50 | 0.8527 | 0.8964 | 0.8487 | 358.6622 |
| rolling | rolling|window=100 | 0.8687 | 0.8882 | 0.8131 | 346.0060 |
| rolling | rolling|window=250 | 0.8748 | 0.8820 | 0.7829 | 333.5668 |
| static | static | 1.0000 | 0.8206 | 0.6536 | 269.3709 |
| weighted | weighted|decay=0.02 | 0.8536 | 0.9077 | 0.8522 | 369.8055 |
| weighted | weighted|decay=0.01 | 0.8581 | 0.8971 | 0.8353 | 350.2375 |
| weighted | weighted|decay=0.005 | 0.8683 | 0.8877 | 0.8078 | 333.3973 |

Selected on the nested development split inside the training period; never on test data.
