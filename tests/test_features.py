import numpy as np
import pandas as pd
import pytest

from gridcast.features import build_features


@pytest.fixture
def history():
    """Two weeks of hourly values 0, 1, 2, ... ending on 2020-01-14 23:00."""
    timestamps = pd.date_range("2020-01-01", periods=336, freq="h")
    return pd.Series(np.arange(336.0), index=timestamps)


def test_features_describe_the_next_24_hours(history):
    features, scale = build_features(history, horizon=24)

    assert features.shape == (24, 9)
    assert scale == 251.5  # mean of the last week: 168 ... 335
    assert list(features["hour"]) == list(range(24))
    assert features["dayofweek"].iloc[0] == 2  # 2020-01-15 is a Wednesday
    assert features["lag_24"].iloc[0] == 312 / scale
    assert features["lag_168"].iloc[0] == 168 / scale


def test_features_do_not_depend_on_the_size_of_the_series(history):
    small, small_scale = build_features(history, horizon=24)
    large, large_scale = build_features(history * 10, horizon=24)

    pd.testing.assert_frame_equal(small, large)
    assert large_scale == pytest.approx(10 * small_scale)


def test_horizon_longer_than_a_day_is_rejected(history):
    with pytest.raises(ValueError):
        build_features(history, horizon=25)
