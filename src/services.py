"""Centralized service factory for the MaintainAI application.

This module provides a single entry point for creating all services,
ensuring consistent configuration and easy testing.
"""
from __future__ import annotations

from typing import Any

from src.config import load_config
from src.context_builder import build_context
from src.knowledge import KnowledgeService
from src.predictive import PredictionService
from src.simulation import SimulationService
from src.slm_dataset import KNOWN_CONDITIONS
from src.slm_service import SLMService


class ServiceFactory:
    """Creates and manages all application services."""

    def __init__(self, config_path: str = "configs/development.yaml"):
        self.config = load_config(config_path)
        self._services: dict[str, Any] = {}

    def get_simulator(self) -> SimulationService:
        if "simulator" not in self._services:
            sim_cfg = self.config.get("simulation", {})
            machines = tuple(sim_cfg.get("machines", ["M001"]))
            seed = sim_cfg.get("seed", 42)
            self._services["simulator"] = SimulationService(machines, seed)
        return self._services["simulator"]

    def get_predictor(self) -> PredictionService | None:
        if "predictor" not in self._services:
            try:
                artifact_dir = self.config.get("predictive", {}).get(
                    "model_path", "models/predictive"
                )
                # Extract directory from model_path
                import os
                artifact_dir = os.path.dirname(artifact_dir) or "models/predictive"
                self._services["predictor"] = PredictionService(artifact_dir)
            except Exception as exc:
                import logging
                logging.getLogger("maintainai.services").error(
                    "predictive model unavailable: %s", exc
                )
                self._services["predictor"] = None
        return self._services["predictor"]

    def get_slm(self) -> SLMService:
        if "slm" not in self._services:
            self._services["slm"] = SLMService(config=self.config)
        return self._services["slm"]

    def get_knowledge(self) -> KnowledgeService:
        if "knowledge" not in self._services:
            self._services["knowledge"] = KnowledgeService()
        return self._services["knowledge"]

    def get_all(self) -> dict[str, Any]:
        return {
            "sim": self.get_simulator(),
            "predictor": self.get_predictor(),
            "slm": self.get_slm(),
            "knowledge": self.get_knowledge(),
            "config": self.config,
        }


def create_services(config_path: str = "configs/development.yaml") -> dict[str, Any]:
    """Convenience function for backward compatibility."""
    factory = ServiceFactory(config_path)
    return factory.get_all()


def machine_state(services: dict[str, Any], machine_id: str) -> dict[str, Any]:
    """Get full machine state: telemetry, prediction, analysis."""
    from src.schemas import MachineContext, Prediction, Telemetry

    sim = services["sim"]
    predictor = services["predictor"]
    slm = services["slm"]

    if predictor is None:
        raise RuntimeError("predictive model unavailable")

    buf = list(sim.machines[machine_id].buffer)
    if not buf:
        raise RuntimeError("telemetry unavailable/stale")

    def _telemetry(mid: str, reading: dict) -> Telemetry:
        return Telemetry(
            machine_id=mid,
            cycle=reading["cycle"],
            op_setting_1=reading["op_setting_1"],
            op_setting_2=reading["op_setting_2"],
            op_setting_3=reading["op_setting_3"],
            sensors=reading["sensors"],
        )

    current = _telemetry(machine_id, buf[-1])
    previous = _telemetry(machine_id, buf[-2]) if len(buf) > 1 else None

    r = predictor.predict(sim.machines[machine_id].history_df())
    pred = Prediction(
        failure_probability=r["failure_probability"],
        anomaly_score=r["anomaly_score"],
        health_state=r["health_state"],
        rul_cycles=r["rul_cycles"],
        model_version=r["model_version"],
        latency_ms=r["latency_ms"],
    )
    ctx = build_context(current, previous, pred, KNOWN_CONDITIONS)
    analysis = slm.analyze(ctx)

    return {
        "telemetry": buf[-1],
        "history": buf,
        "prediction": r,
        "analysis": analysis,
        "trends": ctx.trends,
        "backend": slm.version,
    }


def chat(services: dict[str, Any], machine_id: str, question: str) -> dict[str, Any]:
    """Chat with the maintenance assistant for a specific machine."""
    state = machine_state(services, machine_id)
    from src.schemas import MachineContext, Prediction, Telemetry

    current = Telemetry(
        machine_id=machine_id,
        cycle=state["telemetry"]["cycle"],
        op_setting_1=state["telemetry"]["op_setting_1"],
        op_setting_2=state["telemetry"]["op_setting_2"],
        op_setting_3=state["telemetry"]["op_setting_3"],
        sensors=state["telemetry"]["sensors"],
    )
    pred = Prediction(**state["prediction"])
    ctx = MachineContext(
        machine_id=machine_id,
        current=current,
        trends=state["trends"],
        prediction=pred,
        known_conditions=KNOWN_CONDITIONS,
    )
    return services["slm"].chat(question, ctx)