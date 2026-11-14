"""Evaluate forecasters on the 2019 test period."""

from gridcast.backtest import backtest
from gridcast.cleaning import clean_load
from gridcast.data import load_hourly_load
from gridcast.forecasters import SeasonalNaive


def main() -> None:
    df = clean_load(load_hourly_load())
    results = backtest(df, SeasonalNaive(), test_start="2019-01-01", test_end="2020-01-01")
    print(results.round(3).to_string(index=False))
    print(f"Mean MASE: {results['mase'].mean():.3f}")


if __name__ == "__main__":
    main()
