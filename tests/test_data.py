import pandas as pd

from gridcast.data import load_hourly_load


def test_load_hourly_load_returns_long_format(tmp_path):
    raw = pd.DataFrame(
        {
            "utc_timestamp": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z",
            ],
            "GR_load_actual_entsoe_transparency": [5000, 4800, None],
            "DE_load_actual_entsoe_transparency": [40000, 39000, 38000],
            "GR_solar_generation_actual": [0, 0, 0],
        }
    )
    csv_path = tmp_path / "raw.csv"
    raw.to_csv(csv_path, index=False)

    df = load_hourly_load(countries=("GR", "DE"), path=csv_path)

    assert list(df.columns) == ["item_id", "timestamp", "target"]
    assert len(df) == 6
    assert set(df["item_id"]) == {"GR", "DE"}
    assert df["timestamp"].dt.tz is None
    assert df["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00")
    assert df["target"].isna().sum() == 1
