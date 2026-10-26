"""Single application engine used by the API, dashboard, and evaluations.

Telemetry -> PredictionService -> ContextBuilder -> SLMService is implemented
once here. Transport layers (FastAPI, Streamlit) only call this engine.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import load_config
from src.context_builder import build_context
from src.prompts import build_messages, render_prompt
from src.schemas import MachineContext, Prediction, Telemetry
from src.simulation import SimulationService
from src.slm_dataset import KNOWN_CONDITIONS
from src.slm_service import SLMService

logger = logging.getLogger("maintainai.engine")


class EngineError(Exception):
    """Base class for engine failures mapped to HTTP/API errors."""


class UnknownMachineError(EngineError):
    pass


class PredictiveUnavailableError(EngineError):
    pass


class TelemetryStaleError(EngineError):
    pass


class MaintainAIEngine:
    def __init__(self, config_path: str = "configs/development.yaml"):
        self.config = load_config(config_path)
        sim_cfg = self.config.get("simulation", {})
        self.simulator = SimulationService(
            tuple(sim_cfg.get("machines", ["M001"])),
            seed=int(sim_cfg.get("seed", 42)),
        )
        self.predictor = self._load_predictor()
        self.slm = SLMService(config=self.config)
        self.knowledge = self.slm.knowledge

    def _load_predictor(self):
        try:
            from src.predictive import PredictionService

            model_path = self.config.get("predictive", {}).get(
                "model_path", "models/predictive"
            )
            return PredictionService(model_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("predictive model unavailable: %s", exc)
            return None

    @property
    def sim(self) -> SimulationService:
        return self.simulator

    @property
    def services(self) -> dict[str, Any]:
        return {
            "engine": self,
            "sim": self.simulator,
            "predictor": self.predictor,
            "slm": self.slm,
            "knowledge": self.knowledge,
            "config": self.config,
        }

    @staticmethod
    def _telemetry(machine_id: str, reading: dict) -> Telemetry:
        return Telemetry(
            machine_id=machine_id,
            cycle=reading["cycle"],
            op_setting_1=reading["op_setting_1"],
            op_setting_2=reading["op_setting_2"],
            op_setting_3=reading["op_setting_3"],
            sensors=reading["sensors"],
        )

    def _build(self, machine_id: str) -> tuple[list[dict], Prediction, MachineContext]:
        if machine_id not in self.simulator.machines:
            raise UnknownMachineError(f"unknown machine {machine_id}")
        if self.predictor is None:
            raise PredictiveUnavailableError(
                "predictive model unavailable; no prediction fabricated"
            )
        buf = list(self.simulator.machines[machine_id].buffer)
        if not buf:
            raise TelemetryStaleError("telemetry unavailable/stale")

        current = self._telemetry(machine_id, buf[-1])
        previous = self._telemetry(machine_id, buf[-2]) if len(buf) > 1 else None
        try:
            r = self.predictor.predict(self.simulator.machines[machine_id].history_df())
        except Exception as exc:  # noqa: BLE001
            logger.exception("prediction failed")
            raise PredictiveUnavailableError(
                "prediction failed; no fallback value used"
            ) from exc
        pred = Prediction(
            failure_probability=r["failure_probability"],
            anomaly_score=r["anomaly_score"],
            health_state=r["health_state"],  # type: ignore[arg-type]
            rul_cycles=r["rul_cycles"],
            model_version=r["model_version"],
            latency_ms=r["latency_ms"],
        )
        ctx = build_context(current, previous, pred, KNOWN_CONDITIONS)
        return buf, pred, ctx

    def snapshot(self, machine_id: str) -> dict[str, Any]:
        buf, pred, ctx = self._build(machine_id)
        analysis = self.slm.analyze(ctx)
        return {
            "telemetry": buf[-1],
            "history": buf,
            "prediction": pred.model_dump(),
            "analysis": analysis,
            "trends": ctx.trends,
            "backend": self.slm.version,
        }

    def chat(self, machine_id: str, question: str) -> dict[str, Any]:
        _, _, ctx = self._build(machine_id)
        return self.slm.chat(question, ctx)

    def context(self, machine_id: str) -> MachineContext:
        _, _, ctx = self._build(machine_id)
        return ctx

    def messages(self, machine_id: str) -> list[dict[str, str]]:
        return build_messages(self.context(machine_id))

    def legacy_prompt(self, machine_id: str) -> str:
        return render_prompt(self.context(machine_id))


def create_engine(config_path: str = "configs/development.yaml") -> MaintainAIEngine:
    return MaintainAIEngine(config_path)


__all__ = [
    "MaintainAIEngine",
    "EngineError",
    "UnknownMachineError",
    "PredictiveUnavailableError",
    "TelemetryStaleError",
    "create_engine",
]
