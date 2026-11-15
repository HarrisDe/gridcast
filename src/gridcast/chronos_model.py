"""Chronos-2 forecaster: a pretrained foundation model behind the Forecaster interface."""

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from gridcast.config import get_settings


class ChronosForecaster:
    def __init__(
        self,
        model_id: str | None = None,
        context_length: int | None = None,
        pipeline=None,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.model_id
        self.context_length = context_length or settings.context_length
        self.pipeline = pipeline or Chronos2Pipeline.from_pretrained(
            self.model_id, device_map=settings.device
        )

    def predict_quantiles(self, history, horizon: int, quantile_levels) -> np.ndarray:
        """Return forecasts with shape (horizon, number of quantile levels)."""
        values = np.asarray(history, dtype=np.float32)[-self.context_length :]
        quantiles, _ = self.pipeline.predict_quantiles(
            [torch.tensor(values)],
            prediction_length=horizon,
            quantile_levels=list(quantile_levels),
        )
        return quantiles[0][0].numpy()

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Return the median forecast, as the Forecaster interface requires."""
        return self.predict_quantiles(history.to_numpy(), horizon, [0.5])[:, 0]
