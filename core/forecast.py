from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastConfig:
    """
    Provisional health-index forecasting settings.

    The health threshold and sample requirements are software defaults,
    not manufacturer service-life specifications.
    """

    health_threshold: float = 70.0
    min_samples: int = 10
    min_decline_per_hour: float = 0.1


def forecast_health_threshold(
    dataframe: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> dict[str, float | int | str | None | bool]:
    """
    Estimate time to a health-index threshold using a linear trend.

    Required columns:
      - timestamp
      - health_index

    Optional column:
      - status

    Forecasting is suppressed when status is Unreliable.
    """

    required = {"timestamp", "health_index"}
    missing = sorted(required - set(dataframe.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not 0 <= config.health_threshold <= 100:
        raise ValueError("health_threshold must be between 0 and 100.")

    if config.min_samples < 2:
        raise ValueError("min_samples must be at least 2.")

    if config.min_decline_per_hour < 0:
        raise ValueError("min_decline_per_hour cannot be negative.")

    if dataframe.empty:
        return {
            "forecast_valid": False,
            "reason": "No samples available.",
            "samples_used": 0,
            "health_index_now": None,
            "trend_per_hour": None,
            "time_to_threshold_hours": None,
            "threshold": config.health_threshold,
        }

    if (
        "status" in dataframe.columns
        and (dataframe["status"] == "Unreliable").any()
    ):
        return {
            "forecast_valid": False,
            "reason": "Forecast suppressed because data is Unreliable.",
            "samples_used": 0,
            "health_index_now": None,
            "trend_per_hour": None,
            "time_to_threshold_hours": None,
            "threshold": config.health_threshold,
        }

    working = dataframe[["timestamp", "health_index"]].copy()

    working["timestamp"] = pd.to_datetime(
        working["timestamp"],
        errors="coerce",
        utc=True,
    )

    working["health_index"] = pd.to_numeric(
        working["health_index"],
        errors="coerce",
    )

    working = working.dropna(
        subset=["timestamp", "health_index"]
    ).sort_values("timestamp")

    working = working.loc[
        working["health_index"].between(0, 100)
    ]

    if len(working) < config.min_samples:
        return {
            "forecast_valid": False,
            "reason": (
                f"Insufficient valid samples: need {config.min_samples}, "
                f"found {len(working)}."
            ),
            "samples_used": len(working),
            "health_index_now": (
                float(working["health_index"].iloc[-1])
                if not working.empty
                else None
            ),
            "trend_per_hour": None,
            "time_to_threshold_hours": None,
            "threshold": config.health_threshold,
        }

    elapsed_hours = (
        working["timestamp"] - working["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy() / 3600.0

    health_values = working["health_index"].to_numpy(dtype=float)

    if not np.isfinite(elapsed_hours).all():
        return {
            "forecast_valid": False,
            "reason": "Invalid timestamps prevent forecasting.",
            "samples_used": len(working),
            "health_index_now": float(health_values[-1]),
            "trend_per_hour": None,
            "time_to_threshold_hours": None,
            "threshold": config.health_threshold,
        }

    if np.unique(elapsed_hours).size < 2:
        return {
            "forecast_valid": False,
            "reason": "Timestamps must span at least two distinct times.",
            "samples_used": len(working),
            "health_index_now": float(health_values[-1]),
            "trend_per_hour": None,
            "time_to_threshold_hours": None,
            "threshold": config.health_threshold,
        }

    slope, intercept = np.polyfit(elapsed_hours, health_values, 1)

    health_now = float(health_values[-1])
    trend_per_hour = float(slope)

    if health_now <= config.health_threshold:
        reason = "Health index is already at or below the threshold."
        time_to_threshold = 0.0

    elif trend_per_hour >= -config.min_decline_per_hour:
        reason = "No sufficiently declining health trend detected."
        time_to_threshold = None

    else:
        time_to_threshold = float(
            max(
                0.0,
                (config.health_threshold - health_now)
                / trend_per_hour,
            )
        )
        reason = "Threshold time estimated from the recent linear trend."

    return {
        "forecast_valid": True,
        "reason": reason,
        "samples_used": len(working),
        "health_index_now": health_now,
        "trend_per_hour": trend_per_hour,
        "time_to_threshold_hours": time_to_threshold,
        "threshold": config.health_threshold,
    }
