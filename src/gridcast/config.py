"""Central configuration, overridable with GRIDCAST_* environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRIDCAST_",
        env_file=".env",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    # Model
    model_id: str = "amazon/chronos-2"
    device: Literal["cpu", "cuda"] = "cpu"

    # Forecasting task (hourly data: 24 steps = one day ahead)
    prediction_length: int = Field(default=24, ge=1, le=168)
    context_length: int = Field(default=336, ge=24)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance, so the environment is read only once."""
    return Settings()