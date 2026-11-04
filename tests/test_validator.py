import numpy as np
import pandas as pd

from core.simulator import ROSimulator
from core.validator import validate_window


def healthy_window():
    return ROSimulator(
        duration_seconds=60,
        sampling_interval=1,
        seed=42,
    ).generate_run("healthy")


def test_healthy_window_is_healthy():
    result = validate_window(healthy_window())

    assert result.status == "Healthy"
    assert result.valid_fraction > 0.90
    assert result.stuck_sensors == []


def test_stuck_sensor_is_detected():
    dataframe = ROSimulator(
        duration_seconds=60,
        sampling_interval=1,
        seed=42,
    ).generate_run("stuck_sensor")

    result = validate_window(dataframe)

    assert "feed_tds" in result.stuck_sensors
    assert result.status == "Unreliable"


def test_missing_required_column_is_unreliable():
    dataframe = healthy_window().drop(columns=["feed_tds"])

    result = validate_window(dataframe)

    assert result.status == "Unreliable"
    assert any("Missing required columns" in issue for issue in result.issues)


def test_out_of_range_value_is_unreliable():
    dataframe = healthy_window()
    dataframe.loc[0, "feed_tds"] = -100

    result = validate_window(dataframe)

    assert result.status == "Unreliable"
    assert result.out_of_range_counts["feed_tds"] == 1


def test_infinite_values_are_out_of_range():
    dataframe = healthy_window()
    dataframe.loc[0, "permeate_flow"] = np.inf

    result = validate_window(dataframe)

    assert result.status == "Unreliable"
    assert result.out_of_range_counts["permeate_flow"] == 1


def test_empty_window_is_unreliable():
    dataframe = pd.DataFrame()

    result = validate_window(dataframe)

    assert result.status == "Unreliable"
    assert result.valid_fraction == 0.0
