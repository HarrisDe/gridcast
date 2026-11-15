import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")

from gridcast.chronos_model import ChronosForecaster  # noqa: E402


class FakePipeline:
    """Stands in for the real model: records its input, returns a constant."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.context = inputs[0]
        forecast = torch.full((1, prediction_length, len(quantile_levels)), 42.0)
        return [forecast], [forecast[..., 0]]


def test_chronos_forecaster_truncates_context_and_returns_median():
    fake = FakePipeline()
    model = ChronosForecaster(pipeline=fake, context_length=48)

    forecast = model.predict(pd.Series(np.arange(100.0)), horizon=24)

    assert forecast.shape == (24,)
    assert (forecast == 42.0).all()
    assert len(fake.context) == 48
    assert fake.context[-1] == 99.0
