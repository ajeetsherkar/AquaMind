from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HealthConfig:
    """
    Provisional baseline-relative health thresholds.

    These are development defaults, not manufacturer specifications.
    Confirm them against the SDD and validated hardware data.
    """

    # Maximum allowed increase from the healthy baseline.
    prefilter_pressure_drop_increase: float = 1.0

    # Maximum allowed decrease from the healthy baseline.
    salt_rejection_decrease: float = 5.0
    normalized_flow_decrease_fraction: float = 0.30

    # Provisional acceptable operating ranges.
    system_pressure_range: tuple[float, float] = (0.0, 10.0)
    recovery_range: tuple[float, float] = (0.0, 50.0)

    # Each indicator contributes equally to the overall score.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "prefilter_pressure_drop": 1.0,
            "system_pressure": 1.0,
            "salt_rejection_percent": 1.0,
            "normalized_permeate_flow": 1.0,
            "recovery_percent": 1.0,
        }
    )


INDICATOR_COLUMNS = (
    "prefilter_pressure_drop",
    "system_pressure",
    "salt_rejection_percent",
    "normalized_permeate_flow",
    "recovery_percent",
)


def _degradation_fraction(
    value: float,
    baseline: float,
    threshold: float,
    direction: str,
) -> float:
    """
    Calculate degradation from 0 to 1.

    direction:
      - "increase": degradation occurs as value rises above baseline.
      - "decrease": degradation occurs as value falls below baseline.
    """

    if not np.isfinite(value) or not np.isfinite(baseline):
        return float("nan")

    if direction == "increase":
        if threshold <= 0:
            raise ValueError("Increase threshold must be positive.")
        degradation = (value - baseline) / threshold

    elif direction == "decrease":
        if threshold <= 0:
            raise ValueError("Decrease threshold must be positive.")
        degradation = (baseline - value) / threshold

    else:
        raise ValueError("direction must be 'increase' or 'decrease'.")

    return float(np.clip(degradation, 0.0, 1.0))


def calculate_health_index(
    row: pd.Series | dict[str, float],
    baseline: dict[str, float],
    config: HealthConfig = HealthConfig(),
) -> dict[str, float]:
    """
    Calculate a transparent health index for one sample.

    Returns the overall health index and per-indicator degradation
    fractions. A score of 100 indicates baseline-like performance.
    A score of 0 indicates at least one configured threshold is reached.
    """

    row_keys = set(row.index) if hasattr(row, "index") else set(row.keys())
    missing_row = sorted(set(INDICATOR_COLUMNS) - row_keys)
    missing_baseline = sorted(set(INDICATOR_COLUMNS) - set(baseline))

    if missing_row:
        raise ValueError(f"Missing indicator values: {missing_row}")

    if missing_baseline:
        raise ValueError(f"Missing baseline values: {missing_baseline}")

    degradations: dict[str, float] = {}

    degradations["prefilter_pressure_drop"] = _degradation_fraction(
        float(row["prefilter_pressure_drop"]),
        float(baseline["prefilter_pressure_drop"]),
        config.prefilter_pressure_drop_increase,
        "increase",
    )

    # System pressure and recovery are range-based. Their degradation
    # is zero while inside the configured range and increases outside it.
    pressure = float(row["system_pressure"])
    pressure_low, pressure_high = config.system_pressure_range

    if pressure < pressure_low:
        pressure_threshold = max(
            abs(pressure_low - baseline["system_pressure"]),
            1e-9,
        )
        pressure_degradation = (
            pressure_low - pressure
        ) / pressure_threshold
    elif pressure > pressure_high:
        pressure_threshold = max(
            abs(pressure_high - baseline["system_pressure"]),
            1e-9,
        )
        pressure_degradation = (
            pressure - pressure_high
        ) / pressure_threshold
    else:
        pressure_degradation = 0.0

    degradations["system_pressure"] = float(
        np.clip(pressure_degradation, 0.0, 1.0)
    )

    degradations["salt_rejection_percent"] = _degradation_fraction(
        float(row["salt_rejection_percent"]),
        float(baseline["salt_rejection_percent"]),
        config.salt_rejection_decrease,
        "decrease",
    )

    baseline_flow = float(baseline["normalized_permeate_flow"])
    flow_threshold = abs(baseline_flow) * (
        config.normalized_flow_decrease_fraction
    )

    if flow_threshold <= 0:
        raise ValueError(
            "Baseline normalized flow must be positive "
            "for relative health scoring."
        )

    degradations["normalized_permeate_flow"] = _degradation_fraction(
        float(row["normalized_permeate_flow"]),
        baseline_flow,
        flow_threshold,
        "decrease",
    )

    recovery = float(row["recovery_percent"])
    recovery_low, recovery_high = config.recovery_range

    if recovery < recovery_low:
        recovery_threshold = max(
            abs(recovery_low - baseline["recovery_percent"]),
            1e-9,
        )
        recovery_degradation = (
            recovery_low - recovery
        ) / recovery_threshold
    elif recovery > recovery_high:
        recovery_threshold = max(
            abs(recovery_high - baseline["recovery_percent"]),
            1e-9,
        )
        recovery_degradation = (
            recovery - recovery_high
        ) / recovery_threshold
    else:
        recovery_degradation = 0.0

    degradations["recovery_percent"] = float(
        np.clip(recovery_degradation, 0.0, 1.0)
    )

    weights = config.weights
    if set(weights) != set(INDICATOR_COLUMNS):
        raise ValueError(
            "Weights must be provided for all health indicators."
        )

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Health weights cannot be negative.")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("At least one health weight must be positive.")

    valid_degradations = [
        value
        for value in degradations.values()
        if np.isfinite(value)
    ]

    if not valid_degradations:
        health_index = float("nan")
    else:
        maximum_degradation = max(valid_degradations)

        health_index = float(
            np.clip(
                100.0 * (1.0 - maximum_degradation),
                0.0,
                100.0,
            )
        )

    result = {
        "health_index": health_index,
    }

    result.update(
        {
            f"{name}_degradation": value
            for name, value in degradations.items()
        }
    )

    return result
