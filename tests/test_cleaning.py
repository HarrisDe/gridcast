import numpy as np
import pandas as pd
import pytest

from gridcast.cleaning import clean_load


@pytest.fixture
def hourly_frame():
    """Twenty days of clean hourly load with a daily cycle, for one series."""
    timestamps = pd.date_range("2020-01-01", periods=24 * 20, freq="h")
    load = 50_000 + 15_000 * np.sin(2 * np.pi * timestamps.hour / 24)
    return pd.DataFrame({"item_id": "FR", "timestamp": timestamps, "target": load})


def test_spike_is_replaced_by_plausible_value(hourly_frame):
    hourly_frame.loc[100, "target"] = 158_000

    cleaned = clean_load(hourly_frame)

    assert cleaned["target"].max() <= 65_000
    assert cleaned["target"].notna().all()


def test_short_gap_is_filled_and_long_gap_is_kept(hourly_frame):
    hourly_frame.loc[200:202, "target"] = np.nan  # 3 hours
    hourly_frame.loc[300:319, "target"] = np.nan  # 20 hours

    cleaned = clean_load(hourly_frame, max_gap=6)

    assert cleaned["target"].isna().sum() == 20


def test_missing_edges_are_trimmed(hourly_frame):
    hourly_frame.loc[0, "target"] = np.nan
    hourly_frame.loc[len(hourly_frame) - 1, "target"] = np.nan

    cleaned = clean_load(hourly_frame)

    assert len(cleaned) == len(hourly_frame) - 2
    assert cleaned["target"].notna().all()


def test_clean_data_is_left_unchanged(hourly_frame):
    cleaned = clean_load(hourly_frame)

    pd.testing.assert_frame_equal(cleaned, hourly_frame)
