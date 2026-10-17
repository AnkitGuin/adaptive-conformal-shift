"""Download the UCI Bike Sharing dataset into data/raw/ (idempotent) and verify its checksum.

    python -m experiments.download_data
"""
from __future__ import annotations

from src.data.load import BIKE_HOUR_SHA256, download_bike_sharing, sha256
from src.utils.config import load_config, resolve_path


def main() -> None:
    cfg = load_config("configs/default.yaml")
    path = resolve_path(cfg["data"]["path"])
    if cfg["data"].get("source") != "bike_sharing":
        print(f"data.source is not bike_sharing; expecting user-provided file at {path}")
        return
    if not path.exists():
        print(f"downloading Bike Sharing dataset to {path.parent} ...")
        download_bike_sharing(path.parent)
    digest = sha256(path)
    status = "OK" if digest == BIKE_HOUR_SHA256 else "MISMATCH (file differs from the canonical UCI hour.csv)"
    print(f"{path.relative_to(resolve_path('.'))}: sha256={digest} [{status}]")
    if digest != BIKE_HOUR_SHA256:
        raise RuntimeError("downloaded Bike Sharing hour.csv failed the canonical checksum")


if __name__ == "__main__":
    main()
