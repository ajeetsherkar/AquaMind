import numpy as np
import pytest

from core.simulator import ROSimulator, VALID_SCENARIOS


EXPECTED_COLUMNS = {
    "timestamp",
    "experiment_id",
    "scenario",
    "feed_tds",
    "permeate_tds",
    "temperature",
    "p1",
    "p2",
    "p3",
    "permeate_flow",
    "concentrate_flow",
    "data_source",
}


@pytest.mark.parametrize("scenario", VALID_SCENARIOS)
def test_all_scenarios_generate_expected_data(scenario):
    simulator = ROSimulator(
        duration_seconds=60,
        sampling_interval=1,
        seed=42,
    )

    df = simulator.generate_run(scenario)

    assert len(df) == 60
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert df["scenario"].eq(scenario).all()
    assert df["data_source"].eq("simulated").all()
    assert df["experiment_id"].nunique() == 1
    assert df["timestamp"].notna().all()


def test_unsupported_scenario_raises_error():
    simulator = ROSimulator(seed=42)

    with pytest.raises(ValueError, match="Unsupported scenario"):
        simulator.generate_run("not_a_real_scenario")


@pytest.mark.parametrize(
    ("duration", "interval"),
    [(0, 1), (-1, 1), (10, 0), (10, -1)],
)
def test_invalid_simulator_configuration_raises_error(duration, interval):
    with pytest.raises(ValueError):
        ROSimulator(
            duration_seconds=duration,
            sampling_interval=interval,
        )


def test_prefilter_restriction_trend():
    df = ROSimulator(duration_seconds=300, seed=42).generate_run(
        "prefilter_restriction"
    )

    # Compare scenario trends using the intended endpoints.
    assert df["p1"].iloc[-1] > df["p1"].iloc[0]
    assert df["permeate_flow"].iloc[-1] < df["permeate_flow"].iloc[0]
    assert df["concentrate_flow"].iloc[-1] < df["concentrate_flow"].iloc[0]


def test_membrane_performance_loss_reduces_flow_and_rejection():
    df = ROSimulator(duration_seconds=300, seed=42).generate_run(
        "membrane_performance_loss"
    )

    assert df["permeate_flow"].iloc[-1] < df["permeate_flow"].iloc[0]
    assert df["permeate_tds"].iloc[-1] > df["permeate_tds"].iloc[0]


def test_feed_tds_spike_has_expected_increase():
    df = ROSimulator(duration_seconds=300, seed=42).generate_run(
        "feed_tds_spike"
    )

    before_spike = df["feed_tds"].iloc[:100].median()
    after_spike = df["feed_tds"].iloc[150:].median()

    assert after_spike > before_spike * 1.5


def test_stuck_sensor_has_constant_observed_readings():
    df = ROSimulator(duration_seconds=300, seed=42).generate_run(
        "stuck_sensor"
    )

    observed = df["feed_tds"].dropna()

    assert len(observed) > 0
    assert observed.nunique() == 1


def test_sensor_dropouts_are_missing_not_infinite():
    df = ROSimulator(duration_seconds=300, seed=42).generate_run("healthy")

    numeric = df.select_dtypes(include="number").to_numpy()

    assert not np.isinf(numeric).any()
    # Dropouts are allowed, but every sensor should retain observed readings.
    for column in (
        "feed_tds",
        "permeate_tds",
        "temperature",
        "p1",
        "p2",
        "p3",
        "permeate_flow",
        "concentrate_flow",
    ):
        assert df[column].notna().any()
