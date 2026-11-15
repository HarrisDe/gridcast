"""Choose the Chronos-2 context length on the 2018 validation year."""

import time

from chronos import Chronos2Pipeline

from gridcast.backtest import backtest
from gridcast.chronos_model import ChronosForecaster
from gridcast.cleaning import clean_load
from gridcast.config import get_settings
from gridcast.data import load_hourly_load

VALIDATION_START = "2018-01-01"
VALIDATION_END = "2019-01-01"
CONTEXT_LENGTHS = (168, 336, 672, 1344, 2016)


def main() -> None:
    settings = get_settings()
    df = clean_load(load_hourly_load())
    pipeline = Chronos2Pipeline.from_pretrained(settings.model_id, device_map=settings.device)

    for context_length in CONTEXT_LENGTHS:
        model = ChronosForecaster(pipeline=pipeline, context_length=context_length)
        start = time.perf_counter()
        results = backtest(df, model, test_start=VALIDATION_START, test_end=VALIDATION_END)
        minutes = (time.perf_counter() - start) / 60
        score = results["mase"].mean()
        print(
            f"context {context_length:5d} h: mean MASE {score:.3f} ({minutes:.1f} min)", flush=True
        )


if __name__ == "__main__":
    main()
