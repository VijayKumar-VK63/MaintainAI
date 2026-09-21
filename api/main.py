"""MaintainAI FastAPI: simulator -> prediction -> context -> SLM -> validation.

Error contract (spec section 32):
  - SLM unavailable -> analysis field reports demo/unavailable; monitoring +
    numerical prediction still returned (HTTP 200, never a crash).
  - Predictive model unavailable -> HTTP 503, no fabricated prediction.
  - Stale telemetry (no readings yet) -> HTTP 409 TELEMETRY_STALE.
  - SLM validation failure -> controlled UNKNOWN fallback, logged.
No stack traces leak to clients.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.context_builder import build_context
from src.data_cmapss import SENSOR_COLS
from src.schemas import MachineContext, Prediction, Telemetry
from src.simulation import VALID_SCENARIOS, SimulationService
from src.slm_dataset import KNOWN_CONDITIONS
from src.slm_service import SLMService

logger = logging.getLogger("maintainai.api")

STALE_AFTER_STEPS = 1  # telemetry exists only after at least one tick


class ScenarioRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    scenario: str = Field(pattern="^(NORMAL|DEGRADATION|CRITICAL|FAILURE)$")


class ChatRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=1000)


def create_app(config_path: str = "configs/development.yaml") -> FastAPI:
    cfg = load_config(config_path)
    sim = SimulationService(tuple(cfg.get("simulation", {}).get("machines", ["M001"])),
                            seed=cfg.get("simulation", {}).get("seed", 42))
    try:
        from src.predictive import PredictionService
        predictor = PredictionService("models/predictive")
    except Exception as exc:  # noqa: BLE001
        logger.error("predictive model unavailable: %s", exc)
        predictor = None
    slm = SLMService(config=cfg)  # demo-fallback until HF weights configured
    last_tick = {"ts": 0.0}

    app = FastAPI(title="MaintainAI API")

    def _telemetry(machine_id: str, reading: dict) -> Telemetry:
        return Telemetry(machine_id=machine_id, cycle=reading["cycle"],
                         op_setting_1=reading["op_setting_1"],
                         op_setting_2=reading["op_setting_2"],
                         op_setting_3=reading["op_setting_3"], sensors=reading["sensors"])

    def _machine_state(machine_id: str) -> dict[str, Any]:
        if machine_id not in sim.machines:
            raise HTTPException(404, f"unknown machine {machine_id}")
        if predictor is None:
            raise HTTPException(503, "predictive model unavailable; no prediction fabricated")
        buf = sim.machines[machine_id].buffer
        if len(buf) < STALE_AFTER_STEPS:
            raise HTTPException(409, "telemetry unavailable/stale")
        readings = list(buf)
        current, previous = _telemetry(machine_id, readings[-1]), (
            _telemetry(machine_id, readings[-2]) if len(readings) > 1 else None)
        try:
            r = predictor.predict(sim.machines[machine_id].history_df())
        except Exception as exc:  # noqa: BLE001
            logger.exception("prediction failed")
            raise HTTPException(503, "prediction failed; no fallback value used") from exc
        pred = Prediction(failure_probability=r["failure_probability"],
                          anomaly_score=r["anomaly_score"], health_state=r["health_state"],  # type: ignore[arg-type]
                          rul_cycles=r["rul_cycles"], model_version=r["model_version"],
                          latency_ms=r["latency_ms"])
        ctx = build_context(current, previous, pred, KNOWN_CONDITIONS)
        analysis = slm.analyze(ctx)
        return {"telemetry": readings[-1], "prediction": r, "analysis": analysis,
                "context_trends": ctx.trends, "slm_backend": slm.version}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "simulator_running": sim.running,
                "predictive": predictor is not None, "slm_backend": slm.version,
                "last_tick_ts": last_tick["ts"]}

    @app.get("/machines")
    def machines() -> dict:
        return {"machines": sorted(sim.machines), "scenarios": list(VALID_SCENARIOS)}

    @app.get("/machines/{machine_id}")
    def machine_state(machine_id: str) -> dict:
        return _machine_state(machine_id)

    @app.get("/machines/{machine_id}/telemetry")
    def telemetry(machine_id: str, limit: int = 60) -> dict:
        if machine_id not in sim.machines:
            raise HTTPException(404, f"unknown machine {machine_id}")
        buf = list(sim.machines[machine_id].buffer)[-max(1, min(limit, 200)):]
        if not buf:
            raise HTTPException(409, "telemetry unavailable/stale")
        return {"machine_id": machine_id, "readings": buf}

    @app.post("/simulation/start")
    def start() -> dict:
        sim.start()
        return {"running": True}

    @app.post("/simulation/stop")
    def stop() -> dict:
        sim.stop()
        return {"running": False}

    @app.post("/simulation/reset")
    def reset() -> dict:
        sim.reset()
        return {"running": False, "reset": True}

    @app.post("/simulation/tick")
    def tick() -> dict:
        try:
            out = sim.tick()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        last_tick["ts"] = time.time()
        return {"readings": out}

    @app.post("/simulation/scenario")
    def scenario(req: ScenarioRequest) -> dict:
        if req.machine_id not in sim.machines:
            raise HTTPException(404, f"unknown machine {req.machine_id}")
        sim.set_scenario(req.machine_id, req.scenario)
        return {"machine_id": req.machine_id, "scenario": req.scenario}

    @app.post("/assistant/chat")
    def chat(req: ChatRequest) -> dict:
        state = _machine_state(req.machine_id)
        tel = state["telemetry"]
        current = _telemetry(req.machine_id, tel)
        pred = Prediction(**state["prediction"])
        ctx = MachineContext(machine_id=req.machine_id, current=current,
                             trends=state["context_trends"], prediction=pred,
                             known_conditions=KNOWN_CONDITIONS)
        return slm.chat(req.question, ctx)

    return app


app = create_app()
