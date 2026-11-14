"""Feature engineering shared by training and prediction."""

import numpy as np
import pandas as pd

MAX_HORIZON = 24


def build_features(history: pd.Series, horizon: int) -> tuple[pd.DataFrame, float]:
    """Build one feature row per forecast step, using only values in `history`.

    Returns the features and the scale (mean of the last week) that the
    load-based features were divided by.
    """
    if horizon > MAX_HORIZON:
        raise ValueError(f"horizon must be at most {MAX_HORIZON}, got {horizon}")

    values = history.to_numpy()
    steps = np.arange(horizon)
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    scale = float(np.nanmean(values[-168:]))

    def lag(hours: int) -> np.ndarray:
        return values[len(values) - hours + steps] / scale

    last_day = values[-24:] / scale
    features = pd.DataFrame(
        {
            "hour": future.hour,
            "dayofweek": future.dayofweek,
            "dayofyear": future.dayofyear,
            "lag_24": lag(24),
            "lag_48": lag(48),
            "lag_168": lag(168),
            "last_day_mean": np.nanmean(last_day),
            "last_day_min": np.nanmin(last_day),
            "last_day_max": np.nanmax(last_day),
        }
    )
    return features, scale
