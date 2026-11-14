"""Clean hourly load series: remove spikes, trim edges, fill short gaps."""

import pandas as pd


def _clean_series(s: pd.Series, spike_factor: float, max_gap: int) -> pd.Series:
    """Clean one hourly series indexed by timestamp."""
    median = s.rolling(25, center=True, min_periods=5).median()
    is_spike = (s > spike_factor * median) | (s < median / spike_factor)
    s = s.mask(is_spike)

    s = s.loc[s.first_valid_index() : s.last_valid_index()]

    is_gap = s.isna()
    gap_id = (is_gap != is_gap.shift()).cumsum()
    gap_length = is_gap.groupby(gap_id).transform("sum")
    filled = s.interpolate(method="time")
    return filled.where(~is_gap | (gap_length <= max_gap))


def clean_load(df: pd.DataFrame, spike_factor: float = 2.0, max_gap: int = 6) -> pd.DataFrame:
    """Clean every series in a long-format frame (item_id, timestamp, target)."""
    cleaned = []
    for item_id, group in df.groupby("item_id"):
        s = group.set_index("timestamp")["target"].asfreq("h")
        s = _clean_series(s, spike_factor, max_gap)
        cleaned.append(s.reset_index().assign(item_id=item_id))
    out = pd.concat(cleaned, ignore_index=True)
    return out[["item_id", "timestamp", "target"]]
