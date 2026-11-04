import numpy as np
import pandas as pd
import pytest

from core.physics import (
    PhysicsConfig,
    calculate_indicators,
    establish_healthy_baseline,
)


def test_physics_indicators_match_hand_calculation():
    dataframe = pd.DataFrame(
        {
            "feed_tds": [1000.0],
            "permeate_tds": [50.0],
            "temperature": [30.0],
            "p1": [6.0],
            "p2": [5.0],
            "p3": [4.0],
            "permeate_flow": [2.0],
            "concentrate_flow": [8.0],
        }
    )

    result = calculate_indicators(dataframe)

    # Prefilter pressure drop = p1 - p2 = 6 - 5 = 1
    assert result.loc[0, "prefilter_pressure_drop"] == pytest.approx(1.0)

    # System pressure = p2 = 5
    assert result.loc[0, "system_pressure"] == pytest.approx(5.0)

    # Salt rejection = (1 - 50/1000) * 100 = 95%
    assert result.loc[0, "salt_rejection_percent"] == pytest.approx(95.0)

    # Salt passage = (50/1000) * 100 = 5%
    assert result.loc[0, "salt_passage_percent"] == pytest.approx(5.0)

    # Temperature correction factor = 1 + 0.02 * (30 - 25) = 1.1
    # Normalized flow = 2 / 1.1
    assert result.loc[0, "normalized_permeate_flow"] == pytest.approx(
        2.0 / 1.1
    )

    # Recovery = 2 / (2 + 8) * 100 = 20%
    assert result.loc[0, "recovery_percent"] == pytest.approx(20.0)


def test_custom_pressure_sensor_mapping():
    dataframe = pd.DataFrame(
        {
            "feed_tds": [1000.0],
            "permeate_tds": [50.0],
            "temperature": [25.0],
            "p1": [6.0],
            "p2": [5.0],
            "p3": [4.0],
            "permeate_flow": [2.0],
            "concentrate_flow": [8.0],
        }
    )

    config = PhysicsConfig(
        prefilter_upstream="p2",
        prefilter_downstream="p3",
        system_pressure_sensor="p1",
    )

    result = calculate_indicators(dataframe, config)

    assert result.loc[0, "prefilter_pressure_drop"] == pytest.approx(1.0)
    assert result.loc[0, "system_pressure"] == pytest.approx(6.0)


def test_zero_feed_tds_produces_nan_salt_indicators():
    dataframe = pd.DataFrame(
        {
            "feed_tds": [0.0],
            "permeate_tds": [0.0],
            "temperature": [25.0],
            "p1": [6.0],
            "p2": [5.0],
            "p3": [4.0],
            "permeate_flow": [2.0],
            "concentrate_flow": [8.0],
        }
    )

    result = calculate_indicators(dataframe)

    assert np.isnan(result.loc[0, "salt_rejection_percent"])
    assert np.isnan(result.loc[0, "salt_passage_percent"])


def test_missing_required_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "feed_tds": [1000.0],
            "permeate_tds": [50.0],
            "temperature": [25.0],
            "p1": [6.0],
            "p2": [5.0],
            "p3": [4.0],
            "concentrate_flow": [8.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_indicators(dataframe)


def test_healthy_baseline_uses_first_healthy_samples():
    dataframe = pd.DataFrame(
        {
            "status": ["Healthy"] * 10 + ["Degraded"],
            "prefilter_pressure_drop": [1.0] * 10 + [99.0],
            "system_pressure": [5.0] * 10 + [99.0],
            "salt_rejection_percent": [95.0] * 10 + [1.0],
            "normalized_permeate_flow": [2.0] * 10 + [99.0],
            "recovery_percent": [20.0] * 10 + [99.0],
        }
    )

    baseline = establish_healthy_baseline(dataframe, min_samples=10)

    assert baseline["prefilter_pressure_drop"] == pytest.approx(1.0)
    assert baseline["system_pressure"] == pytest.approx(5.0)
    assert baseline["salt_rejection_percent"] == pytest.approx(95.0)
    assert baseline["normalized_permeate_flow"] == pytest.approx(2.0)
    assert baseline["recovery_percent"] == pytest.approx(20.0)


def test_baseline_requires_minimum_healthy_samples():
    dataframe = pd.DataFrame(
        {
            "status": ["Healthy"] * 3,
            "prefilter_pressure_drop": [1.0] * 3,
            "system_pressure": [5.0] * 3,
            "salt_rejection_percent": [95.0] * 3,
            "normalized_permeate_flow": [2.0] * 3,
            "recovery_percent": [20.0] * 3,
        }
    )

    with pytest.raises(ValueError, match="Need at least"):
        establish_healthy_baseline(dataframe, min_samples=10)
