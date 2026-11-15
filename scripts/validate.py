"""Score a Chronos-2 model (base or fine-tuned) on the 2018 validation year."""

import argparse

from gridcast.backtest import backtest
from gridcast.chronos_model import ChronosForecaster
from gridcast.cleaning import clean_load
from gridcast.data import load_hourly_load

VALIDATION_START = "2018-01-01"
VALIDATION_END = "2019-01-01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None, help="Hub name or local folder")
    args = parser.parse_args()

    df = clean_load(load_hourly_load())
    model = ChronosForecaster(model_id=args.model_id)
    results = backtest(df, model, test_start=VALIDATION_START, test_end=VALIDATION_END)
    print(results.round(3).to_string(index=False))
    print(f"Mean validation MASE: {results['mase'].mean():.3f}")


if __name__ == "__main__":
    main()
