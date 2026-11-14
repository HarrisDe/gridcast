import numpy as np
import pandas as pd
import pytest

from gridcast.backtest import backtest
from gridcast.forecasters import SeasonalNaive


def test_backtest_scores_seasonal_naive_on_a_ramp():
    timestamps = pd.date_range("2020-01-01", periods=24 * 28, freq="h")
    ramp = np.arange(len(timestamps), dtype=float)
    df = pd.concat(
        [
            pd.DataFrame({"item_id": "A", "timestamp": timestamps, "target": ramp}),
            pd.DataFrame({"item_id": "B", "timestamp": timestamps, "target": 10 * ramp}),
        ]
    )

    results = backtest(df, SeasonalNaive(), test_start="2020-01-22", test_end="2020-01-29")

    assert list(results["item_id"]) == ["A", "B"]
    assert list(results["mae"]) == [168.0, 1680.0]
    assert list(results["mase"]) == pytest.approx([1.0, 1.0])
