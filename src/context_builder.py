"""ContextBuilder: compact structured machine context for SLMService.

UI must never build prompts directly; it calls ContextBuilder + SLMService.
"""
from __future__ import annotations

from src.schemas import MachineContext, Prediction, Telemetry


def _trend(previous: float | None, current: float, eps: float = 1e-9) -> float:
    if previous is None:
        return 0.0
    denom = abs(previous) if abs(previous) > eps else eps
    return (current - previous) / denom


def build_context(
    current: Telemetry,
    previous: Telemetry | None,
    prediction: Prediction,
    known_conditions: list[str],
    watch_sensors: tuple[str, ...] = ("s3", "s4", "s9", "s11", "s12"),
) -> MachineContext:
    trends: dict[str, float] = {}
    prev_sensors = previous.sensors if previous else {}
    for name in watch_sensors:
        if name in current.sensors:
            trends[f"{name}_rel_change"] = _trend(prev_sensors.get(name), current.sensors[name])
    return MachineContext(
        machine_id=current.machine_id,
        current=current,
        trends=trends,
        prediction=prediction,
        known_conditions=known_conditions,
    )


def render_prompt(context: MachineContext) -> str:
    """Single place where structured context becomes SLM input text."""
    lines = [
        "You are a predictive maintenance assistant.",
        "Analyze machine condition ONLY from the provided evidence.",
        "Do not invent sensor measurements, maintenance history, inspections,",
        "or unsupported failure modes. Respond with structured JSON only.",
        f"Machine: {context.machine_id}",
        f"Cycle: {context.current.cycle}",
        f"Current sensors: {context.current.sensors}",
        f"Sensor trends: {context.trends}",
        "Predictive model: "
        f"failure_probability={context.prediction.failure_probability:.3f} "
        f"anomaly_score={context.prediction.anomaly_score:.3f} "
        f"health_state={context.prediction.health_state} "
        f"rul_cycles={context.prediction.rul_cycles}",
        f"Known possible conditions: {context.known_conditions}",
        "Task: provide risk_level, likely_condition, confidence, evidence,",
        "recommended_action, urgency as JSON.",
    ]
    return "\n".join(lines)
