"""Request and response models for the API."""

from typing import Annotated

from pydantic import BaseModel, Field

MIN_HISTORY = 168

QuantileLevel = Annotated[float, Field(ge=0.01, le=0.99)]


class ForecastRequest(BaseModel):
    history: list[float | None] = Field(
        min_length=MIN_HISTORY,
        description="Hourly load values, oldest first. Use null for missing hours.",
    )
    horizon: int = Field(default=24, ge=1, le=168, description="Hours to forecast.")
    quantile_levels: list[QuantileLevel] = Field(default=[0.1, 0.5, 0.9], min_length=1)


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    quantiles: dict[str, list[float]] = Field(
        description="One list of hourly forecasts per requested quantile level."
    )
