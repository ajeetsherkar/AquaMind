import pandas as pd

from core.water_quality import (
    WaterQualityConfig,
    assess_water_quality,
)


def test_normal_water_quality():
    row = {
        "feed_tds": 700.0,
        "permeate_tds": 25.0,
        "salt_rejection_percent": 96.4,
    }

    result = assess_water_quality(row)

    assert result["water_quality_status"] == "Normal"
    assert result["water_quality_alerts"] == []


def test_elevated_feed_tds():
    row = {
        "feed_tds": 1400.0,
        "permeate_tds": 40.0,
        "salt_rejection_percent": 97.0,
    }

    result = assess_water_quality(row)

    assert result["water_quality_status"] == "Feed Water Alert"
    assert "Elevated feed TDS" in result["water_quality_alerts"]


def test_elevated_permeate_tds():
    row = {
        "feed_tds": 700.0,
        "permeate_tds": 150.0,
        "salt_rejection_percent": 78.0,
    }

    result = assess_water_quality(row)

    assert result["water_quality_status"] == "Warning"
    assert "Elevated permeate TDS" in result["water_quality_alerts"]
    assert "Low salt rejection" in result["water_quality_alerts"]


def test_custom_thresholds():
    config = WaterQualityConfig(
        feed_tds_warning=500.0,
        permeate_tds_warning=50.0,
        minimum_salt_rejection_percent=95.0,
    )

    row = {
        "feed_tds": 700.0,
        "permeate_tds": 25.0,
        "salt_rejection_percent": 96.0,
    }

    result = assess_water_quality(row, config=config)

    assert result["water_quality_status"] == "Feed Water Alert"


def test_missing_tds_measurements():
    row = {
        "feed_tds": float("nan"),
        "permeate_tds": 25.0,
        "salt_rejection_percent": 96.0,
    }

    result = assess_water_quality(row)

    assert result["water_quality_status"] == "Uncertain"
    assert "Insufficient TDS measurements" in result["water_quality_alerts"]


def test_pandas_series_input():
    row = pd.Series({
        "feed_tds": 1400.0,
        "permeate_tds": 40.0,
        "salt_rejection_percent": 97.0,
    })

    result = assess_water_quality(row)

    assert result["water_quality_status"] == "Feed Water Alert"
