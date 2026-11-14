"""Walk-forward evaluation of a forecaster on held-out data."""

import pandas as pd

from gridcast.forecasters import Forecaster
from gridcast.metrics import mae, mase


def backtest(
    df: pd.DataFrame,
    model: Forecaster,
    test_start: str,
    test_end: str,
    horizon: int = 24,
    season: int = 168,
) -> pd.DataFrame:
    """Forecast each window in [test_start, test_end) and score every series."""
    origins = pd.date_range(test_start, test_end, freq=f"{horizon}h", inclusive="left")
    last_step = pd.Timedelta(hours=horizon - 1)

    rows = []
    for item_id, group in df.groupby("item_id"):
        series = group.set_index("timestamp")["target"]
        actuals, forecasts = [], []
        for origin in origins:
            history = series[series.index < origin]
            actuals.extend(series[origin : origin + last_step])
            forecasts.extend(model.predict(history, horizon))

        train = series[series.index < origins[0]]
        rows.append(
            {
                "item_id": item_id,
                "mae": mae(actuals, forecasts),
                "mase": mase(actuals, forecasts, train, season),
            }
        )
    return pd.DataFrame(rows)
