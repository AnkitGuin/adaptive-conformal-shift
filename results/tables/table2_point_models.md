### Table 2 – Point-model performance (chronological split, mean ± sd over seeds)

| Model | Features | Split part | MAE | RMSE | R² |
|---|---|---|---|---|---|
| ridge | full | calibration | 106.0 ± 0.0 | 138.5 ± 0.0 | 0.587 ± 0.000 |
| ridge | full | test | 105.1 ± 0.0 | 144.7 ± 0.0 | 0.569 ± 0.000 |
| ridge | no_time_id | calibration | 113.5 ± 0.0 | 161.4 ± 0.0 | 0.439 ± 0.000 |
| ridge | no_time_id | test | 121.5 ± 0.0 | 175.6 ± 0.0 | 0.366 ± 0.000 |
| random_forest | full | calibration | 79.7 ± 0.3 | 109.5 ± 0.4 | 0.742 ± 0.002 |
| random_forest | full | test | 87.2 ± 0.2 | 121.5 ± 0.4 | 0.697 ± 0.002 |
| random_forest | no_time_id | calibration | 87.9 ± 0.1 | 119.8 ± 0.1 | 0.691 ± 0.001 |
| random_forest | no_time_id | test | 98.0 ± 0.1 | 136.6 ± 0.1 | 0.616 ± 0.001 |
| gbr | full | calibration | 63.2 ± 1.4 | 86.5 ± 1.9 | 0.839 ± 0.007 |
| gbr | full | test | 76.0 ± 1.0 | 105.3 ± 1.4 | 0.772 ± 0.006 |
| gbr | no_time_id | calibration | 88.6 ± 0.2 | 120.3 ± 0.3 | 0.688 ± 0.001 |
| gbr | no_time_id | test | 99.3 ± 0.1 | 138.3 ± 0.2 | 0.607 ± 0.001 |
| hgb | full | calibration | 65.3 ± 1.8 | 91.3 ± 2.2 | 0.820 ± 0.009 |
| hgb | full | test | 78.5 ± 0.9 | 109.7 ± 1.1 | 0.752 ± 0.005 |
| hgb | no_time_id | calibration | 88.1 ± 0.3 | 119.9 ± 0.4 | 0.691 ± 0.002 |
| hgb | no_time_id | test | 98.6 ± 0.3 | 137.1 ± 0.4 | 0.613 ± 0.003 |

Models trained on the first 60% only. `no_time_id` removes yr, season and mnth.
