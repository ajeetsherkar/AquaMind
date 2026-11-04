import math

import pandas as pd
import pytest

from core.physics import PhysicsConfig, calculate_indicators


def test_physics_indicators_match_hand_calculated_values():
    dataframe = pd.DataFrame(
        {
            "p1": [8.0],
            "p2": [6.0],
            "feed_tds": [500.0],
            "permeate_tds": [25.0],
            "temperature": [25.0],
            "permeate_flow": [2.0],
            "concentrate_flow": [3.0],
        }
    )

    result = calculate_indicators(dataframe)
    row = result.iloc[0]

    assert row["prefilter_pressure_drop"] == pytest.approx(2.0)
    assert row["system_pressure"] == pytest.approx(6.0)
    assert row["salt_rejection_percent"] == pytest.approx(95.0)
    assert row["salt_passage_percent"] == pytest.approx(5.0)
    assert row["normalized_permeate_flow"] == pytest.approx(2.0)
    assert row["recovery_percent"] == pytest.approx(40.0)


def test_temperature_normalization_uses_configured_coefficient():
    dataframe = pd.DataFrame(
        {
            "p1": [8.0],
            "p2": [6.0],
            "feed_tds": [500.0],
            "permeate_tds": [25.0],
            "temperature": [30.0],
            "permeate_flow": [2.0],
            "concentrate_flow": [3.0],
        }
    )

    config = PhysicsConfig(
        reference_temperature_c=25.0,
        temperature_correction_per_degree=0.02,
    )

    result = calculate_indicators(dataframe, config=config)

    expected = 2.0 / (1.0 + 0.02 * (30.0 - 25.0))

    assert result["normalized_permeate_flow"].iloc[0] == pytest.approx(
        expected
    )


def test_nonpositive_feed_tds_produces_nan_salt_indicators():
    dataframe = pd.DataFrame(
        {
            "p1": [8.0],
            "p2": [6.0],
            "feed_tds": [0.0],
            "permeate_tds": [25.0],
            "temperature": [25.0],
            "permeate_flow": [2.0],
            "concentrate_flow": [3.0],
        }
    )

    result = calculate_indicators(dataframe)

    assert math.isnan(result["salt_rejection_percent"].iloc[0])
    assert math.isnan(result["salt_passage_percent"].iloc[0])
