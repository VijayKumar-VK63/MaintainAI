"""MaintainAI FastAPI: simulator -> prediction -> context -> SLM -> validation.

Error contract:
  - No SLM configured -> startup fails with a clear error (never demo fallback).
  - SLM generation/validation failure -> controlled UNKNOWN analysis, logged.
  - Predictive model unavailable -> HTTP 503, no fabricated prediction.
  - Stale telemetry (no readings yet) -> HTTP 409 TELEMETRY_STALE.
No stack traces leak to clients.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

try:  # Preload before pandas/xgboost/FastAPI on Windows (DLL conflict).
    import llama_cpp  # noqa: F401
except ImportError:
    pass

# Allow `python api/main.py` from the project root (not only `-m api.main`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.config import load_config  # noqa: E402
from src.engine import (  # noqa: E402
    MaintainAIEngine,
    PredictiveUnavailableError,
    TelemetryStaleError,
    UnknownMachineError,
)
from src.simulation import VALID_SCENARIOS  # noqa: E402


class ScenarioRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    scenario: str = Field(pattern="^(NORMAL|DEGRADATION|CRITICAL|FAILURE)$")


class ChatRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=1000)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownMachineError):
        return HTTPException(404, str(exc))
    if isinstance(exc, PredictiveUnavailableError):
        return HTTPException(503, str(exc))
    if isinstance(exc, TelemetryStaleError):
        return HTTPException(409, str(exc))
    return HTTPException(500, "internal error")


def create_app(config_path: str = "configs/development.yaml") -> FastAPI:
    # Config is loaded by the engine; kept here for backwards-compatible imports.
    load_config(config_path)
    engine = MaintainAIEngine(config_path)
    sim = engine.sim
    last_tick = {"ts": 0.0}

    app = FastAPI(title="MaintainAI API")

    def _machine_state(machine_id: str) -> dict[str, Any]:
        try:
            state = engine.snapshot(machine_id)
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc
        return {
            "telemetry": state["telemetry"],
            "prediction": state["prediction"],
            "analysis": state["analysis"],
            "context_trends": state["trends"],
            "slm_backend": state["backend"],
        }

    @app.get("/")
    def root() -> dict:
        return {
            "service": "MaintainAI API",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "machines": "/machines",
            "chat": "POST /assistant/chat",
        }

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "simulator_running": sim.running,
            "predictive": engine.predictor is not None,
            "slm_backend": engine.slm.version,
            "last_tick_ts": last_tick["ts"],
        }

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
        last_tick["ts"] = 0.0
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
        try:
            return engine.chat(req.machine_id, req.question)
        except Exception as exc:  # noqa: BLE001
            raise _http_error(exc) from exc

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )
