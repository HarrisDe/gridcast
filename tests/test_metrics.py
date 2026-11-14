import pytest

from gridcast.metrics import mae, mase


def test_mae():
    assert mae([14, 13], [12, 13]) == 1.0


def test_mae_ignores_missing_values():
    assert mae([14, float("nan")], [12, 13]) == 2.0


def test_mase_scales_by_naive_error():
    score = mase([14, 13], [12, 13], y_train=[10, 12, 11, 13, 12], season=1)
    assert score == pytest.approx(1 / 1.5)
