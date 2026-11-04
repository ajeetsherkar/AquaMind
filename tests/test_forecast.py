import pandas as pd
import pytest

from core.forecast import ForecastConfig, forecast_health_threshold


def make_health_data(values, interval_minutes=60):
    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=len(values),
        freq=f"{interval_minutes}min",
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "health_index": values,
    })


def test_forecast_detects_declining_health():
    dataframe = make_health_data([100, 95, 90, 85, 80, 75])

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(health_threshold=70, min_samples=5),
    )

    assert result["forecast_valid"] is True
    assert result["trend_per_hour"] == pytest.approx(-5.0)
    assert result["time_to_threshold_hours"] == pytest.approx(1.0)


def test_forecast_stable_health_has_no_threshold_estimate():
    dataframe = make_health_data([90, 90, 90, 90, 90])

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(min_samples=5),
    )

    assert result["forecast_valid"] is True
    assert result["time_to_threshold_hours"] is None


def test_forecast_already_at_threshold():
    dataframe = make_health_data([80, 75, 70])

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(health_threshold=70, min_samples=2),
    )

    assert result["forecast_valid"] is True
    assert result["time_to_threshold_hours"] == 0.0


def test_forecast_rejects_insufficient_samples():
    dataframe = make_health_data([100, 95, 90])

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(min_samples=10),
    )

    assert result["forecast_valid"] is False
    assert result["time_to_threshold_hours"] is None


def test_forecast_suppresses_unreliable_data():
    dataframe = make_health_data([100, 95, 90, 85, 80])
    dataframe["status"] = "Healthy"
    dataframe.loc[2, "status"] = "Unreliable"

    result = forecast_health_threshold(dataframe)

    assert result["forecast_valid"] is False
    assert result["time_to_threshold_hours"] is None


def test_forecast_requires_timestamp_and_health_index():
    dataframe = pd.DataFrame({"health_index": [100, 90, 80]})

    with pytest.raises(ValueError):
        forecast_health_threshold(dataframe)


def test_forecast_ignores_invalid_health_values():
    dataframe = make_health_data([100, 95, float("nan"), 85, 80, 75])

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(health_threshold=70, min_samples=5),
    )

    assert result["forecast_valid"] is True
    assert result["samples_used"] == 5


def test_forecast_rejects_health_values_outside_valid_range():
    dataframe = make_health_data([110, 100, 95, 90, 85, 80, 75])

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(health_threshold=70, min_samples=5),
    )

    assert result["forecast_valid"] is True
    assert result["samples_used"] == 6


def test_forecast_rejects_duplicate_timestamps():
    dataframe = make_health_data([100, 95, 90, 85, 80])
    dataframe["timestamp"] = pd.Timestamp("2026-01-01 00:00:00")

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(min_samples=5),
    )

    assert result["forecast_valid"] is False
    assert result["reason"] == "Timestamps must span at least two distinct times."


def test_forecast_handles_missing_health_index_values():
    dataframe = make_health_data([100, 95, 90, 85, 80])
    dataframe.loc[2, "health_index"] = float("nan")

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(min_samples=5),
    )

    assert result["forecast_valid"] is False
    assert result["samples_used"] == 4


def test_forecast_handles_invalid_timestamps():
    dataframe = make_health_data([100, 95, 90, 85, 80])
    dataframe["timestamp"] = dataframe["timestamp"].astype(object)
    dataframe.loc[2, "timestamp"] = "invalid"

    result = forecast_health_threshold(
        dataframe,
        ForecastConfig(min_samples=5),
    )

    assert result["forecast_valid"] is False
    assert result["samples_used"] == 4
