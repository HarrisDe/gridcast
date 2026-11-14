"""LightGBM forecaster: one global model trained across all series."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from gridcast.features import build_features

MIN_HISTORY = pd.Timedelta(days=14)


class LightGBMForecaster:
    def __init__(self, **params) -> None:
        self.model = LGBMRegressor(**params)

    def fit(self, df: pd.DataFrame, train_end: str, horizon: int = 24) -> "LightGBMForecaster":
        """Train on every daily forecast window before `train_end`."""
        last_step = pd.Timedelta(hours=horizon - 1)
        feature_frames, targets = [], []
        for _, group in df.groupby("item_id"):
            series = group.set_index("timestamp")["target"]
            series = series[series.index < train_end]
            first_origin = series.index[0].normalize() + MIN_HISTORY
            origins = pd.date_range(first_origin, train_end, freq="24h", inclusive="left")
            for origin in origins:
                features, scale = build_features(series[series.index < origin], horizon)
                feature_frames.append(features)
                targets.append(series[origin : origin + last_step].to_numpy() / scale)

        X = pd.concat(feature_frames, ignore_index=True)
        y = np.concatenate(targets)
        known = ~np.isnan(y)
        self.model.fit(X[known], y[known])
        return self

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        features, scale = build_features(history, horizon)
        return self.model.predict(features) * scale
