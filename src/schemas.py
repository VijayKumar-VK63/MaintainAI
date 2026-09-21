"""Pydantic schemas: telemetry validation + SLM structured-output validation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]


class Telemetry(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    cycle: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensors: dict[str, float]  # s1..s21 for CMAPSS FD001


class Prediction(BaseModel):
    failure_probability: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0)
    health_state: Literal["HEALTHY", "WARNING", "HIGH_RISK", "CRITICAL"]
    rul_cycles: Optional[int] = Field(default=None, ge=0)
    model_version: str
    latency_ms: float = Field(ge=0.0)


class SLMAnalysis(BaseModel):
    """Structured maintenance interpretation. UNKNOWN allowed on weak evidence."""

    risk_level: RiskLevel
    likely_condition: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=10)
    recommended_action: str = Field(min_length=1, max_length=512)
    urgency: Literal["ROUTINE", "PRIORITY", "IMMEDIATE", "UNKNOWN"]


class MachineContext(BaseModel):
    machine_id: str
    current: Telemetry
    trends: dict[str, float]
    prediction: Prediction
    known_conditions: list[str]
