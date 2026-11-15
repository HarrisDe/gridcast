import numpy as np
from fastapi.testclient import TestClient

from gridcast import __version__
from gridcast.api import app, get_forecaster


class FakeForecaster:
    """Stands in for the real model: each quantile forecast is 100 x its level."""

    model_id = "fake-model"

    def predict_quantiles(self, history, horizon, quantile_levels):
        return np.tile(np.asarray(quantile_levels) * 100, (horizon, 1))


app.dependency_overrides[get_forecaster] = FakeForecaster
client = TestClient(app)


def test_health_reports_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_forecast_returns_one_series_per_quantile():
    payload = {"history": [1000.0] * 168, "horizon": 3, "quantile_levels": [0.1, 0.5]}

    response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "model": "fake-model",
        "horizon": 3,
        "quantiles": {"0.1": [10.0, 10.0, 10.0], "0.5": [50.0, 50.0, 50.0]},
    }


def test_forecast_accepts_missing_values_and_uses_defaults():
    payload = {"history": [1000.0, None] * 84}

    response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    assert response.json()["horizon"] == 24
    assert list(response.json()["quantiles"]) == ["0.1", "0.5", "0.9"]


def test_history_shorter_than_a_week_is_rejected():
    response = client.post("/v1/forecast", json={"history": [1000.0] * 100})

    assert response.status_code == 422


def test_quantile_level_outside_range_is_rejected():
    payload = {"history": [1000.0] * 168, "quantile_levels": [1.5]}

    response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 422
