from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WaterQualityConfig:
    """
    Provisional water-quality thresholds.

    These are software-development defaults, not manufacturer
    specifications or drinking-water certification limits.
    """

    feed_tds_warning: float = 1000.0
    permeate_tds_warning: float = 100.0
    minimum_salt_rejection_percent: float = 90.0


def assess_water_quality(
    row: pd.Series | dict[str, Any],
    config: WaterQualityConfig = WaterQualityConfig(),
) -> dict[str, Any]:
    """
    Assess feed and permeate water quality for one sample.

    The result is independent of the component Health Index.
    """

    feed_tds = pd.to_numeric(row.get("feed_tds"), errors="coerce")
    permeate_tds = pd.to_numeric(row.get("permeate_tds"), errors="coerce")
    salt_rejection = pd.to_numeric(
        row.get("salt_rejection_percent"), errors="coerce"
    )

    alerts: list[str] = []

    if pd.isna(feed_tds) or pd.isna(permeate_tds):
        return {
            "water_quality_status": "Uncertain",
            "water_quality_alerts": ["Insufficient TDS measurements"],
            "feed_tds": None if pd.isna(feed_tds) else float(feed_tds),
            "permeate_tds": (
                None if pd.isna(permeate_tds) else float(permeate_tds)
            ),
            "salt_rejection_percent": (
                None if pd.isna(salt_rejection) else float(salt_rejection)
            ),
        }

    if feed_tds > config.feed_tds_warning:
        alerts.append("Elevated feed TDS")

    if permeate_tds > config.permeate_tds_warning:
        alerts.append("Elevated permeate TDS")

    if (
        not pd.isna(salt_rejection)
        and salt_rejection < config.minimum_salt_rejection_percent
    ):
        alerts.append("Low salt rejection")

    if "Elevated permeate TDS" in alerts or "Low salt rejection" in alerts:
        status = "Warning"
    elif "Elevated feed TDS" in alerts:
        status = "Feed Water Alert"
    else:
        status = "Normal"

    return {
        "water_quality_status": status,
        "water_quality_alerts": alerts,
        "feed_tds": float(feed_tds),
        "permeate_tds": float(permeate_tds),
        "salt_rejection_percent": (
            None if pd.isna(salt_rejection) else float(salt_rejection)
        ),
    }
