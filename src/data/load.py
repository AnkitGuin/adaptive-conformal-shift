"""Dataset loading.

Two sources are supported:

* ``bike_sharing`` – the UCI Bike Sharing hourly data (``hour.csv``).  If the
  file is missing it can be downloaded with :func:`download_bike_sharing`
  (or ``python -m experiments.download_data``).
* ``csv`` – any CSV with a timestamp column, predictors and a continuous
  target, described entirely in the configuration file.

In both cases the loader returns a DataFrame sorted by a ``timestamp`` column
(stable sort, so ties keep file order) and never shuffles rows.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path

BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
# SHA-256 of the canonical hour.csv from the UCI archive (used to detect a corrupted file)
BIKE_HOUR_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_bike_sharing(dest_dir: str | Path = "data/raw", url: str = BIKE_URL) -> Path:
    """Download and extract hour.csv / day.csv into ``dest_dir``; return the hour.csv path."""
    dest = resolve_path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in ("hour.csv", "day.csv", "Readme.txt"):
            if name in zf.namelist():
                (dest / name).write_bytes(zf.read(name))
    return dest / "hour.csv"


def load_dataset(data_cfg: dict, allow_download: bool = True) -> pd.DataFrame:
    """Load the dataset described by ``data_cfg`` and return it in chronological order.

    Required keys: ``source`` ('bike_sharing' | 'csv'), ``path``, ``target``.
    For 'csv' either ``timestamp_col`` or (``date_col`` and ``hour_col``) is used
    to build the ``timestamp`` column.
    """
    path = resolve_path(data_cfg["path"])
    source = data_cfg.get("source", "csv")
    if not path.exists():
        if source == "bike_sharing" and allow_download:
            download_bike_sharing(path.parent, data_cfg.get("download_url", BIKE_URL))
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. See data/README.md for acquisition instructions."
            )
    if source == "bike_sharing" and data_cfg.get("verify_checksum", True):
        digest = sha256(path)
        if digest != BIKE_HOUR_SHA256:
            raise ValueError(
                f"Bike Sharing hour.csv checksum mismatch for {path}: {digest}; "
                f"expected canonical UCI SHA-256 {BIKE_HOUR_SHA256}"
            )
    df = pd.read_csv(path)

    if data_cfg.get("timestamp_col"):
        ts = pd.to_datetime(df[data_cfg["timestamp_col"]])
    else:
        ts = pd.to_datetime(df[data_cfg["date_col"]])
        if data_cfg.get("hour_col"):
            ts = ts + pd.to_timedelta(df[data_cfg["hour_col"]], unit="h")
    df = df.assign(timestamp=ts)
    target = data_cfg["target"]
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not in dataset")
    df = df.dropna(subset=[target])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df
