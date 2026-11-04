from core.simulator import ROSimulator
from core.logger import save_run


SCENARIOS = [
    "healthy",
    "prefilter_restriction",
    "membrane_performance_loss",
    "feed_tds_spike",
    "stuck_sensor",
]


def main() -> None:
    simulator = ROSimulator(
        duration_seconds=300,
        sampling_interval=1.0,
        seed=42,
    )

    for scenario in SCENARIOS:
        dataframe = simulator.generate_run(scenario)

        experiment_id = dataframe[
            "experiment_id"
        ].iloc[0]

        output_path = save_run(
            dataframe=dataframe,
            experiment_id=experiment_id,
            scenario=scenario,
        )

        print(
            f"Generated {scenario}: "
            f"{len(dataframe)} rows -> {output_path}"
        )


if __name__ == "__main__":
    main()
