"""Download and load the Open Power System Data hourly load dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from gridcast.config import get_settings

OPSD_BASE = "https://data.open-power-system-data.org/time_series/2020-10-06"
OPSD_URL = f"{OPSD_BASE}/time_series_60min_singleindex.csv"
RAW_FILENAME = "opsd_time_series_60min.csv"

LOAD_SUFFIX = "_load_actual_entsoe_transparency"
DEFAULT_COUNTRIES = ("GR", "DE", "FR", "IT", "ES")


def download_raw(force: bool = False) -> Path:
    """Download the raw CSV into data/raw/ unless it is already there."""
    raw_dir = get_settings().data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILENAME
    if target.exists() and not force:
        return target
    urlretrieve(OPSD_URL, target)
    return target


def load_hourly_load(
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    path: Path | None = None,
) -> pd.DataFrame:
    """Return hourly actual load in long format: item_id, timestamp, target (MW)."""
    path = path or download_raw()
    columns = ["utc_timestamp"] + [f"{c}{LOAD_SUFFIX}" for c in countries]
    wide = pd.read_csv(path, usecols=columns, parse_dates=["utc_timestamp"])

    long = wide.melt(id_vars="utc_timestamp", var_name="item_id", value_name="target")
    long["item_id"] = long["item_id"].str.removesuffix(LOAD_SUFFIX)
    long["timestamp"] = long.pop("utc_timestamp").dt.tz_convert(None)

    return long[["item_id", "timestamp", "target"]].sort_values(
        ["item_id", "timestamp"], ignore_index=True
    )
