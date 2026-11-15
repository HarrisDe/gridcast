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
        self.context_length = context_length or settings.context_length
        self.pipeline = pipeline or Chronos2Pipeline.from_pretrained(
            model_id or settings.model_id, device_map=settings.device
        )

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        context = torch.tensor(history.to_numpy()[-self.context_length :], dtype=torch.float32)
        quantiles, _ = self.pipeline.predict_quantiles(
            [context], prediction_length=horizon, quantile_levels=[0.5]
        )
        return quantiles[0][0, :, 0].numpy()
