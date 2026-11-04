from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class RODataPoint:
    timestamp: str
    experiment_id: str
    scenario: str

    feed_tds: float
    permeate_tds: float
    temperature: float

    p1: float
    p2: float
    p3: float

    permeate_flow: float
    concentrate_flow: float

    data_source: str = "simulated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
