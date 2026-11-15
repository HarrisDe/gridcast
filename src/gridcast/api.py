"""HTTP API for the forecasting service."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from gridcast import __version__
from gridcast.schemas import ForecastRequest, ForecastResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts."""
    from gridcast.chronos_model import ChronosForecaster

    app.state.forecaster = ChronosForecaster()
    yield


app = FastAPI(
    title="gridcast",
    description="Day-ahead electricity load forecasting with a fine-tuned Chronos-2 model.",
    version=__version__,
    lifespan=lifespan,
)


def get_forecaster(request: Request):
    return request.app.state.forecaster


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok", "version": __version__}


@app.post("/v1/forecast")
def forecast(
    body: ForecastRequest,
    forecaster: Annotated[object, Depends(get_forecaster)],
) -> ForecastResponse:
    """Forecast the next hours of a series as quantiles."""
    values = forecaster.predict_quantiles(body.history, body.horizon, body.quantile_levels)
    return ForecastResponse(
        model=forecaster.model_id,
        horizon=body.horizon,
        quantiles={str(q): values[:, i].tolist() for i, q in enumerate(body.quantile_levels)},
    )
