### Table 1 – Dataset statistics (UCI Bike Sharing, hourly; chronological split)

| split | n | start | end | target_mean | target_std | target_median | target_min | target_max | temp_mean |
|---|---|---|---|---|---|---|---|---|---|
| train | 10427 | 2011-01-01 00:00:00 | 2012-03-15 13:00:00 | 144.797 | 134.621 | 110.000 | 1 | 782 | 0.460 |
| calibration | 3476 | 2012-03-15 14:00:00 | 2012-08-07 11:00:00 | 264.157 | 215.525 | 233.000 | 1 | 957 | 0.618 |
| test | 3476 | 2012-08-07 12:00:00 | 2012-12-31 23:00:00 | 248.755 | 220.517 | 205.000 | 1 | 977 | 0.488 |
| all | 17379 | 2011-01-01 00:00:00 | 2012-12-31 23:00:00 | 189.463 | 181.388 | 142.000 | 1 | 977 | 0.497 |

Target = hourly rental count `cnt`. The test period (Aug–Dec 2012) has a higher mean and variance than the training period (2011 – Mar 2012): the real data already contain a trend/seasonal shift.
