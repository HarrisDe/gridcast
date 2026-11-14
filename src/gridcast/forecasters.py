"""Forecasting models behind one common interface."""

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    """Anything that can turn a history into a forecast."""

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray: ...


class SeasonalNaive:
    """Repeat the values observed one season ago."""

    def __init__(self, season: int = 168) -> None:
        self.season = season

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        last_season = history.to_numpy()[-self.season :]
        repeats = -(-horizon // self.season)
        return np.tile(last_season, repeats)[:horizon]
