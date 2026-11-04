from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


DataStatus = Literal["Healthy", "Degraded", "Unreliable"]


@dataclass(frozen=True)
class ValidationConfig:
    """
    Provisional validation limits for AquaMind's simulated RO data.

    These are software-development defaults, not manufacturer specifications.
    Update them when the SDD or hardware limits are confirmed.
    """

    ranges: dict[str, tuple[float, float]]
    max_missing_fraction: float = 0.10
    stuck_min_samples: int = 10
    stuck_min_unique: int = 2
    max_relative_jump: float = 0.50


DEFAULT_CONFIG = ValidationConfig(
    ranges={
        "feed_tds": (0.0, 5000.0),
        "permeate_tds": (0.0, 2000.0),
        "temperature": (0.0, 60.0),
        "p1": (0.0, 20.0),
        "p2": (0.0, 20.0),
        "p3": (0.0, 20.0),
        "permeate_flow": (0.0, 10.0),
        "concentrate_flow": (0.0, 20.0),
    }
)

SENSOR_COLUMNS = tuple(DEFAULT_CONFIG.ranges)


@dataclass
class ValidationResult:
    status: DataStatus
    issues: list[str]
    valid_fraction: float
    stuck_sensors: list[str]
    out_of_range_counts: dict[str, int]
    missing_counts: dict[str, int]


def _detect_stuck_sensor(
    series: pd.Series,
    min_samples: int,
    min_unique: int,
) -> bool:
    """
    Detect a sensor that has enough observed samples but too few unique values.

    This is a simple heuristic. A naturally steady sensor may also trigger it,
    so thresholds should be tuned using real sensor data.
    """
    observed = series.dropna()

    if len(observed) < min_samples:
        return False

    return observed.nunique() < min_unique


def validate_window(
    dataframe: pd.DataFrame,
    config: ValidationConfig = DEFAULT_CONFIG,
) -> ValidationResult:
    """
    Validate one sample window and assign Healthy, Degraded, or Unreliable.

    Unreliable:
      - required sensor columns are missing
      - all samples are missing
      - too many missing or out-of-range values
      - a stuck sensor is detected

    Degraded:
      - some missing values, range violations, or implausible jumps exist,
        but the window remains usable

    Healthy:
      - no validation issues were detected
    """
    if dataframe.empty:
        return ValidationResult(
            status="Unreliable",
            issues=["Window contains no samples"],
            valid_fraction=0.0,
            stuck_sensors=[],
            out_of_range_counts={},
            missing_counts={},
        )

    required_columns = set(SENSOR_COLUMNS)
    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        return ValidationResult(
            status="Unreliable",
            issues=[f"Missing required columns: {missing_columns}"],
            valid_fraction=0.0,
            stuck_sensors=[],
            out_of_range_counts={},
            missing_counts={},
        )

    issues: list[str] = []
    missing_counts: dict[str, int] = {}
    out_of_range_counts: dict[str, int] = {}
    stuck_sensors: list[str] = []

    total_values = len(dataframe) * len(SENSOR_COLUMNS)
    valid_values = 0

    for column in SENSOR_COLUMNS:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        missing_count = int(values.isna().sum())
        missing_counts[column] = missing_count

        lower, upper = config.ranges[column]
        out_of_range_mask = values.notna() & (
            (values < lower) | (values > upper) | ~np.isfinite(values)
        )
        out_of_range_count = int(out_of_range_mask.sum())
        out_of_range_counts[column] = out_of_range_count

        valid_mask = values.notna() & ~out_of_range_mask
        valid_values += int(valid_mask.sum())

        if missing_count:
            issues.append(f"{column}: {missing_count} missing values")

        if out_of_range_count:
            issues.append(
                f"{column}: {out_of_range_count} out-of-range values"
            )

        if _detect_stuck_sensor(
            values,
            min_samples=config.stuck_min_samples,
            min_unique=config.stuck_min_unique,
        ):
            stuck_sensors.append(column)
            issues.append(f"{column}: possible stuck sensor")

        observed = values.where(valid_mask).dropna()

        if len(observed) >= 2:
            previous = observed.shift(1).dropna()
            current = observed.iloc[1:]

            denominator = previous.abs().clip(lower=1.0)
            relative_jumps = (current - previous).abs() / denominator

            if (relative_jumps > config.max_relative_jump).any():
                issues.append(f"{column}: implausible relative jump")

    valid_fraction = valid_values / total_values if total_values else 0.0

    missing_fraction = 1.0 - valid_fraction
    out_of_range_total = sum(out_of_range_counts.values())

    jump_detected = any(
        "implausible relative jump" in issue
        for issue in issues
    )

    if valid_fraction == 0:
        status: DataStatus = "Unreliable"
    elif (
        missing_fraction > config.max_missing_fraction
        or out_of_range_total > 0
        or stuck_sensors
    ):
        status = "Unreliable"
    elif jump_detected:
        status = "Degraded"
    else:
        status = "Healthy"

    return ValidationResult(
        status=status,
        issues=issues,
        valid_fraction=valid_fraction,
        stuck_sensors=stuck_sensors,
        out_of_range_counts=out_of_range_counts,
        missing_counts=missing_counts,
    )
