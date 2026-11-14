import pytest
from pydantic import ValidationError

from gridcast.config import Settings


def test_defaults_target_cpu():
    settings = Settings(_env_file=None)
    assert settings.device == "cpu"
    assert settings.prediction_length == 24


def test_env_variable_overrides_default(monkeypatch):
    monkeypatch.setenv("GRIDCAST_PREDICTION_LENGTH", "48")
    assert Settings(_env_file=None).prediction_length == 48


def test_invalid_value_is_rejected(monkeypatch):
    monkeypatch.setenv("GRIDCAST_PREDICTION_LENGTH", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
