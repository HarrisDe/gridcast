"""Forecast accuracy metrics."""

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean absolute error, ignoring missing values."""
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(error)))


def mase(y_true, y_pred, y_train, season: int = 168) -> float:
    """MAE scaled by the in-sample error of a seasonal-naive forecast.

    Below 1 means better than repeating the value from one season ago.
    """
    y_train = np.asarray(y_train, dtype=float)
    scale = np.nanmean(np.abs(y_train[season:] - y_train[:-season]))
    return mae(y_true, y_pred) / float(scale)
