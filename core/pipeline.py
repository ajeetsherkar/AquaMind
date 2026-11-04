from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from core.forecast import ForecastConfig, forecast_health_threshold
from core.health import HealthConfig, calculate_health_index
from core.physics import (
    PhysicsConfig,
    calculate_indicators,
    establish_healthy_baseline,
)
from core.simulator import ROSimulator
from core.validator import validate_window
from core.water_quality import assess_water_quality


def run_pipeline(
    scenario: str = "healthy",
    duration_seconds: int = 300,
    sampling_interval: float = 1.0,
    seed: int = 42,
    baseline_min_samples: int = 10,
    physics_config: PhysicsConfig = PhysicsConfig(),
    health_config: HealthConfig = HealthConfig(),
    forecast_config: ForecastConfig = ForecastConfig(),
) -> dict[str, Any]:
    """
    Run the complete synthetic RO monitoring pipeline.

    The baseline is generated separately using a healthy simulation.
    All model thresholds and physics settings are provisional.
    """

    # 1. Generate the requested scenario.
    simulator = ROSimulator(
        duration_seconds=duration_seconds,
        sampling_interval=sampling_interval,
        seed=seed,
    )
    raw_data = simulator.generate_run(scenario)

    # 2. Validate the requested scenario.
    validation = validate_window(raw_data)

    data = raw_data.copy()
    data["status"] = validation.status

    # 3. Calculate RO physics indicators.
    data = calculate_indicators(data, config=physics_config)

    # 4. Generate a separate healthy baseline run.
    baseline_simulator = ROSimulator(
        duration_seconds=duration_seconds,
        sampling_interval=sampling_interval,
        seed=42,
    )
    baseline_raw = baseline_simulator.generate_run("healthy")
    baseline_validation = validate_window(baseline_raw)

    if baseline_validation.status != "Healthy":
        raise ValueError(
            "The reference healthy simulation was not validated as Healthy. "
            f"Status: {baseline_validation.status}"
        )

    baseline_data = baseline_raw.copy()
    baseline_data["status"] = baseline_validation.status
    baseline_data = calculate_indicators(
        baseline_data,
        config=physics_config,
    )

    baseline = establish_healthy_baseline(
        baseline_data,
        min_samples=baseline_min_samples,
    )

    # 5. Calculate health index and indicator degradation values.
    health_results = data.apply(
        lambda row: calculate_health_index(
            row=row,
            baseline=baseline,
            config=health_config,
        ),
        axis=1,
    )

    health_frame = pd.DataFrame(
        health_results.tolist(),
        index=data.index,
    )
    data = pd.concat([data, health_frame], axis=1)

    # Suppress health scores when sensor data is unreliable.
    if validation.status == "Unreliable":
        data["health_index"] = float("nan")

    # 6. Assess water quality independently of component health.
    water_quality_results = data.apply(
        lambda row: assess_water_quality(row),
        axis=1,
    )
    water_quality_frame = pd.DataFrame(
        water_quality_results.tolist(),
        index=data.index,
    )

    # Keep original sensor measurements; append only water-quality
    # assessment outputs that are not already present in the data.
    water_quality_columns = [
        column
        for column in water_quality_frame.columns
        if column not in data.columns
    ]
    data = pd.concat(
        [data, water_quality_frame[water_quality_columns]],
        axis=1,
    )


    # 7. Forecast the health threshold.
    forecast = forecast_health_threshold(
        data[["timestamp", "health_index", "status"]],
        config=forecast_config,
    )

    # 8. Return all outputs.
    return {
        "scenario": scenario,
        "status": validation.status,
        "validation": asdict(validation),
        "baseline": baseline,
        "data": data,
        "forecast": forecast,
        "metadata": {
            "data_source": "simulated",
            "duration_seconds": duration_seconds,
            "sampling_interval": sampling_interval,
            "seed": seed,
            "baseline_seed": 42,
            "baseline_min_samples": baseline_min_samples,
        },
    }
