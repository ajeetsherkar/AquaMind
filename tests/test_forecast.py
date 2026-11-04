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
