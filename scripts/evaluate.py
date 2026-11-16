"""Evaluate forecasters on the 2019 test period."""

from gridcast.backtest import backtest
from gridcast.chronos_model import ChronosForecaster
from gridcast.cleaning import clean_load
from gridcast.data import load_hourly_load
from gridcast.forecasters import SeasonalNaive
from gridcast.lgbm import LightGBMForecaster

TRAIN_END = "2019-01-01"
TEST_END = "2020-01-01"
FINETUNED_MODEL = "HarrisDe/chronos2-lora-eu-load"
BASE_MODEL = "amazon/chronos-2"


def main() -> None:
    df = clean_load(load_hourly_load())
    models = {
        "seasonal_naive": SeasonalNaive(),
        "lightgbm": LightGBMForecaster(
            n_estimators=500, learning_rate=0.05, random_state=0, verbose=-1
        ).fit(df, train_end=TRAIN_END),
        "chronos2_zero_shot": ChronosForecaster(model_id=BASE_MODEL),
        "chronos2_lora": ChronosForecaster(model_id=FINETUNED_MODEL),
    }

    for name, model in models.items():
        results = backtest(df, model, test_start=TRAIN_END, test_end=TEST_END)
        print(f"\n{name}")
        print(results.round(3).to_string(index=False))
        print(f"Mean MASE: {results['mase'].mean():.3f}")


if __name__ == "__main__":
    main()
