"""HTTP API for the forecasting service."""

from fastapi import FastAPI

from gridcast import __version__

app = FastAPI(
    title="gridcast",
    description="Day-ahead electricity load forecasting with a fine-tuned Chronos-2 model.",
    version=__version__,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok", "version": __version__}
