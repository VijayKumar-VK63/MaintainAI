"""ContextBuilder: compact structured machine context for SLMService.

UI must never build prompts directly; it calls ContextBuilder + SLMService.
"""
from __future__ import annotations

from src.prompts import render_prompt
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


__all__ = ["build_context", "render_prompt"]
