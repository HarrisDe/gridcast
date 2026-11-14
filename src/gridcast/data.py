"""Download and load the Open Power System Data hourly load dataset."""

from pathlib import Path
from urllib.request import urlretrieve

from gridcast.config import get_settings

OPSD_BASE = "https://data.open-power-system-data.org/time_series/2020-10-06"
OPSD_URL = f"{OPSD_BASE}/time_series_60min_singleindex.csv"
RAW_FILENAME = "opsd_time_series_60min.csv"


def download_raw(force: bool = False) -> Path:
    """Download the raw CSV into data/raw/ unless it is already there."""
    raw_dir = get_settings().data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILENAME
    if target.exists() and not force:
        return target
    urlretrieve(OPSD_URL, target)
    return target
