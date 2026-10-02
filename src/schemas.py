from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SecurityEvent(BaseModel):
    event_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    source: Literal[
        "network-anomaly-detector",
        "cicids2017",
        "cse-cic-ids2018",
        "other",
    ]

    event_type: str = "network_anomaly"

    source_ip: str | None = None
    destination_ip: str | None = None

    source_port: int | None = Field(default=None, ge=0, le=65535)
    destination_port: int | None = Field(default=None, ge=0, le=65535)

    protocol: str | None = None

    prediction: Literal["BENIGN", "ATTACK", "UNKNOWN"]

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    evidence: dict = Field(default_factory=dict)
