"""Phase 3 tests: service interface, bounds, determinism, health bands."""
import pandas as pd

from src.data_cmapss import SENSOR_COLS, load_fd001
from src.predictive import PredictionService, health_state

RAW_COLS = ["unit", "cycle", "op1", "op2", "op3"] + SENSOR_COLS


def _history(n: int = 40):
    _, te, _ = load_fd001("CMAPSSData")
    return te[te["unit"] == 1].head(n)[RAW_COLS]


def test_predict_returns_structured_bounds():
    svc = PredictionService("models/predictive")
    r = svc.predict(_history())
    assert 0.0 <= r["failure_probability"] <= 1.0
    assert r["anomaly_score"] >= 0.0
    assert r["health_state"] in ("HEALTHY", "WARNING", "HIGH_RISK", "CRITICAL")
    assert r["rul_cycles"] >= 0
    assert r["model_version"] == "rul-xgb-clf-v1"
    assert r["latency_ms"] >= 0.0


def test_predict_deterministic():
    svc = PredictionService("models/predictive")
    h = _history()
    assert svc.predict(h)["rul_cycles"] == svc.predict(h)["rul_cycles"]


def test_health_bands():
    assert health_state(0.0) == "HEALTHY"
    assert health_state(0.5) == "WARNING"
    assert health_state(0.8) == "HIGH_RISK"
    assert health_state(0.95) == "CRITICAL"


def test_degraded_engine_scores_worse_than_fresh():
    svc = PredictionService("models/predictive")
    _, te, _ = load_fd001("CMAPSSData")
    fresh = te[te["unit"] == 1].head(10)[RAW_COLS]
    aged = te[te["unit"] == 1].tail(30)[RAW_COLS]
    assert svc.predict(aged)["failure_probability"] >= svc.predict(fresh)["failure_probability"]
