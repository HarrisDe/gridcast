import numpy as np
import pandas as pd

from gridcast.lgbm import LightGBMForecaster
from gridcast.metrics import mae


def test_lightgbm_forecaster_learns_a_daily_pattern():
    timestamps = pd.date_range("2020-01-01", periods=24 * 60, freq="h")
    load = 1000 + 300 * np.sin(2 * np.pi * timestamps.hour / 24)
    df = pd.DataFrame({"item_id": "A", "timestamp": timestamps, "target": load})

    model = LightGBMForecaster(n_estimators=50, verbose=-1).fit(df, train_end="2020-02-20")
    history = df.set_index("timestamp")["target"][:"2020-02-24 23:00"]
    forecast = model.predict(history, horizon=24)

    assert forecast.shape == (24,)
    assert mae(load[:24], forecast) < 30
