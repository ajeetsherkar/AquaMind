from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from core.schema import RODataPoint


Scenario = Literal[
    "healthy",
    "prefilter_restriction",
    "membrane_performance_loss",
    "feed_tds_spike",
    "stuck_sensor",
]


class ROSimulator:
    """
    Physics-informed RO operating data simulator.

    All generated measurements are synthetic.
    This is a software-development simulator, not a calibrated
    digital twin or physical validation of an RO purifier.
    """

    def __init__(
        self,
        duration_seconds: int = 300,
        sampling_interval: float = 1.0,
        seed: int | None = 42,
    ) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be positive")

        self.duration_seconds = duration_seconds
        self.sampling_interval = sampling_interval
        self.rng = np.random.default_rng(seed)

    def generate_run(self, scenario: Scenario) -> pd.DataFrame:
        valid_scenarios = (
            "healthy",
            "prefilter_restriction",
            "membrane_performance_loss",
            "feed_tds_spike",
            "stuck_sensor",
        )

        if scenario not in valid_scenarios:
            raise ValueError(f"Unsupported scenario: {scenario}")

        experiment_id = str(uuid.uuid4())[:8]

        elapsed = np.arange(
            0,
            self.duration_seconds,
            self.sampling_interval,
        )

        n = len(elapsed)

        # Generate compressed experiment timestamps.
        start_time = datetime.now(timezone.utc)

        timestamps = [
            (
                start_time
                + timedelta(seconds=float(seconds))
            ).isoformat()
            for seconds in elapsed
        ]

        # Healthy baseline.
        feed_tds = np.full(n, 700.0)
        temperature = np.full(n, 25.0)

        p1 = np.full(n, 5.5)
        p2 = np.full(n, 5.0)
        p3 = np.full(n, 4.5)

        permeate_flow = np.full(n, 0.25)
        concentrate_flow = np.full(n, 0.75)

        rejection = np.full(n, 0.97)

        # Prefilter restriction.
        if scenario == "prefilter_restriction":
            severity = np.linspace(0.0, 1.0, n)

            # Illustrative increase in prefilter pressure drop.
            p1 += 0.8 * severity

            # Restriction reduces available flow.
            permeate_flow *= 1.0 - 0.25 * severity
            concentrate_flow *= 1.0 - 0.15 * severity

        # Membrane performance loss.
        elif scenario == "membrane_performance_loss":
            severity = np.linspace(0.0, 1.0, n)

            permeate_flow *= 1.0 - 0.30 * severity

            # Lower rejection means more dissolved solids
            # pass into the permeate.
            rejection = 0.97 - 0.12 * severity

        # Feed-water TDS spike.
        elif scenario == "feed_tds_spike":
            spike_start = n // 3
            feed_tds[spike_start:] = 1400.0

        # Simplified salt-rejection model.
        permeate_tds = feed_tds * (1.0 - rejection)

        # Add measurement noise.
        feed_tds += self.rng.normal(0, 8.0, n)
        permeate_tds += self.rng.normal(0, 2.0, n)

        temperature += self.rng.normal(0, 0.2, n)

        p1 += self.rng.normal(0, 0.03, n)
        p2 += self.rng.normal(0, 0.03, n)
        p3 += self.rng.normal(0, 0.03, n)

        permeate_flow += self.rng.normal(0, 0.005, n)
        concentrate_flow += self.rng.normal(0, 0.01, n)

        # Stuck feed-TDS sensor.
        # Freeze the reading after measurement noise is applied.
        if scenario == "stuck_sensor" and n > 1:
            feed_tds[1:] = feed_tds[0]

        # Prevent impossible negative readings.
        feed_tds = np.maximum(feed_tds, 0)
        permeate_tds = np.maximum(permeate_tds, 0)

        permeate_flow = np.maximum(permeate_flow, 0)
        concentrate_flow = np.maximum(concentrate_flow, 0)

        # Apply occasional dropouts to sensor channels.
        sensor_arrays = {
            "feed_tds": feed_tds,
            "permeate_tds": permeate_tds,
            "temperature": temperature,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "permeate_flow": permeate_flow,
            "concentrate_flow": concentrate_flow,
        }

        for values in sensor_arrays.values():
            dropout_mask = self.rng.random(n) < 0.015
            values[dropout_mask] = np.nan

        records = []

        for i in range(n):
            point = RODataPoint(
                timestamp=timestamps[i],
                experiment_id=experiment_id,
                scenario=scenario,
                feed_tds=float(feed_tds[i]),
                permeate_tds=float(permeate_tds[i]),
                temperature=float(temperature[i]),
                p1=float(p1[i]),
                p2=float(p2[i]),
                p3=float(p3[i]),
                permeate_flow=float(permeate_flow[i]),
                concentrate_flow=float(concentrate_flow[i]),
                data_source="simulated",
            )

            records.append(point.to_dict())

        return pd.DataFrame(records)
