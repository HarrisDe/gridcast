"""Call the running API with real Greek load data and compare with what happened."""

import httpx
import numpy as np

from gridcast.cleaning import clean_load
from gridcast.data import load_hourly_load

API_URL = "http://127.0.0.1:8000/v1/forecast"


def main() -> None:
    df = clean_load(load_hourly_load(countries=("GR",)))
    series = df.set_index("timestamp")["target"]
    history = series[:"2019-06-30 23:00"].tail(1344)
    actual = series["2019-07-01 00:00":"2019-07-01 23:00"].to_numpy()

    payload = {
        "history": [None if np.isnan(v) else float(v) for v in history],
        "horizon": 24,
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    response = httpx.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    quantiles = result["quantiles"]

    print(f"Status {response.status_code}, model: {result['model']}")
    print(f"Round trip: {response.elapsed.total_seconds():.2f} s")
    print("hour  actual     p10     p50     p90")
    for hour in range(0, 24, 4):
        low, median, high = (quantiles[level][hour] for level in ("0.1", "0.5", "0.9"))
        print(f"{hour:4d} {actual[hour]:7.0f} {low:7.0f} {median:7.0f} {high:7.0f}")
    print(f"MAE of the median: {np.mean(np.abs(actual - np.array(quantiles['0.5']))):.0f} MW")


if __name__ == "__main__":
    main()
