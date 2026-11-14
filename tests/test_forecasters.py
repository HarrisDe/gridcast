import numpy as np
import pandas as pd

from gridcast.forecasters import SeasonalNaive


def test_seasonal_naive_repeats_last_week():
    history = pd.Series(np.arange(336.0))

    forecast = SeasonalNaive(season=168).predict(history, horizon=24)

    np.testing.assert_array_equal(forecast, np.arange(168.0, 192.0))


def test_seasonal_naive_handles_horizon_longer_than_season():
    history = pd.Series([1.0, 2.0, 3.0])

    forecast = SeasonalNaive(season=3).predict(history, horizon=7)

    np.testing.assert_array_equal(forecast, [1, 2, 3, 1, 2, 3, 1])
