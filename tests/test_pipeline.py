import math

import pytest

from core.pipeline import run_pipeline


@pytest.fixture(scope="module")
def healthy_result():
    return run_pipeline(scenario="healthy")


def test_healthy_pipeline_returns_expected_outputs(healthy_result):
    result = healthy_result

    assert result["status"] == "Healthy"
    assert "health_index" in result["data"].columns
    assert "water_quality_status" in result["data"].columns
    assert "water_quality_alerts" in result["data"].columns
    assert result["data"]["water_quality_status"].iloc[-1] == "Normal"


def test_feed_tds_spike_generates_feed_water_alert():
    result = run_pipeline(scenario="feed_tds_spike")
    data = result["data"]

    assert data["water_quality_status"].iloc[-1] == "Feed Water Alert"
    assert "Elevated feed TDS" in data["water_quality_alerts"].iloc[-1]


def test_membrane_performance_loss_generates_warning():
    result = run_pipeline(scenario="membrane_performance_loss")
    data = result["data"]

    assert data["health_index"].mean() < 50
    assert data["water_quality_status"].iloc[-1] == "Warning"
    assert "Elevated permeate TDS" in data["water_quality_alerts"].iloc[-1]
    assert "Low salt rejection" in data["water_quality_alerts"].iloc[-1]


def test_unreliable_sensor_suppresses_health_index():
    result = run_pipeline(scenario="stuck_sensor")
    data = result["data"]

    assert result["status"] == "Unreliable"
    assert data["health_index"].isna().all()
    assert result["forecast"]["reason"] == "Forecast suppressed because data is Unreliable."


def test_feed_tds_spike_does_not_automatically_reduce_health_index():
    healthy = run_pipeline(scenario="healthy")
    feed_spike = run_pipeline(scenario="feed_tds_spike")

    healthy_mean = healthy["data"]["health_index"].mean()
    feed_spike_mean = feed_spike["data"]["health_index"].mean()

    assert math.isfinite(healthy_mean)
    assert math.isfinite(feed_spike_mean)
    assert feed_spike_mean > 80
