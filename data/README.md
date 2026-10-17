# Data

## Primary dataset: UCI Bike Sharing (hourly)

* Source: UCI Machine Learning Repository, "Bike Sharing Dataset" (Fanaee-T & Gama, 2013),
  <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>, licence CC BY 4.0.
* File used: `data/raw/hour.csv` (17,379 hourly records, 2011-01-01 to 2012-12-31).
* SHA-256 of the canonical `hour.csv`:
  `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`

### Acquisition

```bash
python -m experiments.download_data      # downloads + verifies the checksum
```

The loader (`src/data/load.py`) also downloads the file automatically if it is
missing. Manual alternative: download the zip from the URL above and extract
`hour.csv` into `data/raw/`.

### Columns

| column | use |
|---|---|
| `dteday`, `hr` | timestamp = date + hour; defines chronological order |
| `cnt` | **target** (total hourly rentals) |
| `season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed` | predictors |
| `casual`, `registered` | **never used** (they sum to `cnt` → target leakage); enforced by `check_feature_spec` |
| `instant` | row id; never used (encodes time position) |

Known data properties: 165 hours are missing (gaps in the hourly sequence);
no missing values in the remaining rows; `weathersit = 4` occurs 3 times.

## Using another dataset

Set in a config file (see `configs/default.yaml`):

```yaml
data:
  source: csv
  path: data/raw/my_data.csv
  timestamp_col: time         # or date_col + hour_col
  target: y
  forbidden_features: []
features:
  numeric: [...]
  categorical: [...]
  cyclic: {hour: 24}
```

Paths are always relative to the repository root. The synthetic-shift
scenarios in `configs/shifts.yaml` reference Bike Sharing column names
(`temp`, `hum`, `workingday`, `hr`) and must be adapted for other data.

`data/processed/` is intentionally empty: preprocessing is part of each
model pipeline and fitted on training data at run time, so no pre-processed
(and potentially leaky) data files are stored.
