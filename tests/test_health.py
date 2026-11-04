import numpy as np
import pytest

from core.health import HealthConfig, calculate_health_index


BASELINE = {
    "prefilter_pressure_drop": 1.0,
    "system_pressure": 5.0,
    "salt_rejection_percent": 95.0,
    "normalized_permeate_flow": 2.0,
    "recovery_percent": 20.0,
}


def healthy_row():
    return BASELINE.copy()


def test_baseline_like_readings_produce_health_100():
    result = calculate_health_index(healthy_row(), BASELINE)

    assert result["health_index"] == pytest.approx(100.0)


def test_prefilter_threshold_reached_produces_health_zero():
    row = healthy_row()
    row["prefilter_pressure_drop"] = (
        BASELINE["prefilter_pressure_drop"] + 1.0
    )

    result = calculate_health_index(row, BASELINE)

    assert result["prefilter_pressure_drop_degradation"] == pytest.approx(1.0)
    assert result["health_index"] == pytest.approx(0.0)


def test_salt_rejection_threshold_reached():
    row = healthy_row()
    row["salt_rejection_percent"] = 90.0

    result = calculate_health_index(row, BASELINE)

    assert result["salt_rejection_percent_degradation"] == pytest.approx(1.0)
    assert result["health_index"] == pytest.approx(0.0)


def test_flow_degradation_is_baseline_relative():
    row = healthy_row()
    row["normalized_permeate_flow"] = 1.7

    result = calculate_health_index(row, BASELINE)

    assert result["normalized_permeate_flow_degradation"] == pytest.approx(0.5)
    assert result["health_index"] == pytest.approx(50.0)


def test_degradation_is_clipped_at_threshold():
    row = healthy_row()
    row["prefilter_pressure_drop"] = 10.0

    result = calculate_health_index(row, BASELINE)

    assert result["prefilter_pressure_drop_degradation"] == pytest.approx(1.0)


def test_missing_indicator_raises_error():
    row = healthy_row()
    del row["system_pressure"]

    with pytest.raises(ValueError, match="Missing indicator values"):
        calculate_health_index(row, BASELINE)


def test_missing_baseline_indicator_raises_error():
    baseline = BASELINE.copy()
    del baseline["recovery_percent"]

    with pytest.raises(ValueError, match="Missing baseline values"):
        calculate_health_index(healthy_row(), baseline)


def test_invalid_weight_configuration_raises_error():
    config = HealthConfig(
        weights={
            "prefilter_pressure_drop": 1.0,
        }
    )

    with pytest.raises(ValueError, match="Weights must be provided"):
        calculate_health_index(healthy_row(), BASELINE, config)


def test_non_finite_indicator_is_ignored_in_weighted_score():
    row = healthy_row()
    row["system_pressure"] = float("nan")

    result = calculate_health_index(row, BASELINE)

    assert np.isfinite(result["health_index"])
    assert result["health_index"] == pytest.approx(100.0)


def test_system_pressure_above_configured_range_produces_progressive_degradation():
    row = healthy_row()
    row["system_pressure"] = 11.0

    result = calculate_health_index(row, BASELINE)

    assert result["system_pressure_degradation"] == pytest.approx(0.2)
    assert result["health_index"] == pytest.approx(80.0)
def test_recovery_above_configured_range_produces_progressive_degradation():
    row = healthy_row()
    row["recovery_percent"] = 60.0

    result = calculate_health_index(row, BASELINE)

    assert result["recovery_percent_degradation"] == pytest.approx(1 / 3)
    assert result["health_index"] == pytest.approx(100 * (1 - 1 / 3))
def test_health_index_stays_between_zero_and_100():
    row = healthy_row()
    row["prefilter_pressure_drop"] = 100.0
    row["salt_rejection_percent"] = -100.0
    row["normalized_permeate_flow"] = 0.0

    result = calculate_health_index(row, BASELINE)

    assert 0.0 <= result["health_index"] <= 100.0
