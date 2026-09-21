"""Importable local pipeline for the Streamlit app (no Streamlit dependency).

Lets tests exercise exactly what the dashboard runs: tick simulator ->
PredictionService -> ContextBuilder -> SLMService -> validated analysis.
"""
from __future__ import annotations

from typing import Any

from src.config import load_config
from src.context_builder import build_context
from src.data_cmapss import SENSOR_COLS
from src.schemas import MachineContext, Prediction, Telemetry
from src.simulation import SimulationService
from src.slm_dataset import KNOWN_CONDITIONS
from src.slm_service import SLMService


def create_services(config_path: str = "configs/development.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    sim_cfg = cfg.get("simulation", {})
    sim = SimulationService(tuple(sim_cfg.get("machines", ["M001"])), seed=sim_cfg.get("seed", 42))
    try:
        from src.predictive import PredictionService
        predictor = PredictionService("models/predictive")
    except Exception:
        predictor = None
    return {"config": cfg, "sim": sim, "predictor": predictor, "slm": SLMService(config=cfg)}


def _telemetry(machine_id: str, reading: dict) -> Telemetry:
    return Telemetry(machine_id=machine_id, cycle=reading["cycle"],
                     op_setting_1=reading["op_setting_1"], op_setting_2=reading["op_setting_2"],
                     op_setting_3=reading["op_setting_3"], sensors=reading["sensors"])


def machine_state(svc: dict[str, Any], machine_id: str) -> dict[str, Any]:
    sim, predictor, slm = svc["sim"], svc["predictor"], svc["slm"]
    if predictor is None:
        raise RuntimeError("predictive model unavailable")
    buf = list(sim.machines[machine_id].buffer)
    if not buf:
        raise RuntimeError("telemetry unavailable/stale")
    current = _telemetry(machine_id, buf[-1])
    previous = _telemetry(machine_id, buf[-2]) if len(buf) > 1 else None
    r = predictor.predict(sim.machines[machine_id].history_df())
    pred = Prediction(failure_probability=r["failure_probability"], anomaly_score=r["anomaly_score"],
                      health_state=r["health_state"],  # type: ignore[arg-type]
                      rul_cycles=r["rul_cycles"], model_version=r["model_version"], latency_ms=r["latency_ms"])
    ctx = build_context(current, previous, pred, KNOWN_CONDITIONS)
    analysis = slm.analyze(ctx)
    return {"telemetry": buf[-1], "history": buf, "prediction": r,
            "analysis": analysis, "trends": ctx.trends, "backend": slm.version}


def chat(svc: dict[str, Any], machine_id: str, question: str) -> dict[str, Any]:
    state = machine_state(svc, machine_id)
    current = _telemetry(machine_id, state["telemetry"])
    pred = Prediction(**state["prediction"])
    ctx = MachineContext(machine_id=machine_id, current=current, trends=state["trends"],
                         prediction=pred, known_conditions=KNOWN_CONDITIONS)
    return svc["slm"].chat(question, ctx)
