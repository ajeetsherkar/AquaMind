from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhysicsConfig:
    """
    Provisional RO calculation settings.

    Pressure sensor positions and temperature correction coefficient
    must be confirmed against the SDD or hardware documentation.
    """

    prefilter_upstream: str = "p1"
    prefilter_downstream: str = "p2"
    system_pressure_sensor: str = "p2"

    reference_temperature_c: float = 25.0
    temperature_correction_per_degree: float = 0.02


@dataclass
class PhysicsResult:
    prefilter_pressure_drop: float
    system_pressure: float
    salt_rejection_percent: float
    salt_passage_percent: float
    normalized_permeate_flow: float
    recovery_percent: float


def calculate_indicators(
    dataframe: pd.DataFrame,
    config: PhysicsConfig = PhysicsConfig(),
) -> pd.DataFrame:
    """
    Calculate RO operating indicators for each sample.

    Inputs are expected to contain:
      feed_tds, permeate_tds, temperature,
      configured pressure sensors, permeate_flow, concentrate_flow.

    Invalid or missing values result in NaN indicators rather than
    fabricated measurements.
    """
    required = {
        "feed_tds",
        "permeate_tds",
        "temperature",
        "permeate_flow",
        "concentrate_flow",
        config.prefilter_upstream,
        config.prefilter_downstream,
        config.system_pressure_sensor,
    }

    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = dataframe.copy()

    result["prefilter_pressure_drop"] = (
        pd.to_numeric(result[config.prefilter_upstream], errors="coerce")
        - pd.to_numeric(result[config.prefilter_downstream], errors="coerce")
    )

    result["system_pressure"] = pd.to_numeric(
        result[config.system_pressure_sensor],
        errors="coerce",
    )

    feed_tds = pd.to_numeric(result["feed_tds"], errors="coerce")
    permeate_tds = pd.to_numeric(result["permeate_tds"], errors="coerce")

    valid_feed = feed_tds > 0
    result["salt_rejection_percent"] = np.where(
        valid_feed,
        (1.0 - permeate_tds / feed_tds) * 100.0,
        np.nan,
    )

    result["salt_passage_percent"] = np.where(
        valid_feed,
        (permeate_tds / feed_tds) * 100.0,
        np.nan,
    )

    temperature = pd.to_numeric(result["temperature"], errors="coerce")
    permeate_flow = pd.to_numeric(
        result["permeate_flow"],
        errors="coerce",
    )

    correction_factor = 1.0 + (
        config.temperature_correction_per_degree
        * (temperature - config.reference_temperature_c)
    )

    result["normalized_permeate_flow"] = np.where(
        correction_factor > 0,
        permeate_flow / correction_factor,
        np.nan,
    )

    concentrate_flow = pd.to_numeric(
        result["concentrate_flow"],
        errors="coerce",
    )

    total_feed_flow = permeate_flow + concentrate_flow

    result["recovery_percent"] = np.where(
        total_feed_flow > 0,
        permeate_flow / total_feed_flow * 100.0,
        np.nan,
    )

    return result


def establish_healthy_baseline(
    dataframe: pd.DataFrame,
    min_samples: int = 10,
) -> dict[str, float]:
    """
    Establish a baseline from the first stable healthy period.

    The input must already contain calculated indicators and a status
    column identifying samples as Healthy.
    """
    required = {
        "status",
        "prefilter_pressure_drop",
        "system_pressure",
        "salt_rejection_percent",
        "normalized_permeate_flow",
        "recovery_percent",
    }

    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    healthy = dataframe.loc[dataframe["status"] == "Healthy"].copy()

    if len(healthy) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} healthy samples "
            f"to establish a baseline; found {len(healthy)}"
        )

    baseline_columns = [
        "prefilter_pressure_drop",
        "system_pressure",
        "salt_rejection_percent",
        "normalized_permeate_flow",
        "recovery_percent",
    ]

    baseline: dict[str, float] = {}

    for column in baseline_columns:
        values = pd.to_numeric(healthy[column], errors="coerce").dropna()

        if len(values) < min_samples:
            raise ValueError(
                f"Insufficient valid values for baseline: {column}"
            )

        baseline[column] = float(values.iloc[:min_samples].median())

    return baseline
