"""Phase 8 tests: simulator realism/determinism + API + end-to-end integration."""
from fastapi.testclient import TestClient

from api.main import create_app
from src.schemas import SLMAnalysis
from src.simulation import MachineSimulator


def test_simulator_deterministic_per_seed():
    a, b = MachineSimulator("M001", seed=11), MachineSimulator("M001", seed=11)
    for _ in range(10):
        assert a.step()["sensors"]["s3"] == b.step()["sensors"]["s3"]


def test_simulator_degradation_drifts_up():
    s = MachineSimulator("M001", seed=5)
    early = [s.step()["sensors"]["s3"] for _ in range(10)]
    s.set_scenario("DEGRADATION")
    late = [s.step()["sensors"]["s3"] for _ in range(150)][-10:]
    assert sum(late) / 10 > sum(early) / 10 + 5.0  # pronounced, FD001-like drift


def test_simulator_rejects_bad_scenario():
    import pytest
    s = MachineSimulator("M001")
    with pytest.raises(ValueError):
        s.set_scenario("EXPLODE")


def _client():
    return TestClient(create_app("configs/development.yaml"))


def test_api_health_and_machines():
    c = _client()
    assert c.get("/health").json()["status"] == "ok"
    assert "M001" in c.get("/machines").json()["machines"]


def test_api_stale_before_tick():
    c = _client()
    r = c.get("/machines/M001")
    assert r.status_code == 409  # stale, not fabricated


def test_full_pipeline_telemetry_to_validated_analysis():
    c = _client()
    c.post("/simulation/start")
    c.post("/simulation/scenario", json={"machine_id": "M001", "scenario": "DEGRADATION"})
    for _ in range(40):
        c.post("/simulation/tick")
    state = c.get("/machines/M001").json()
    assert 0.0 <= state["prediction"]["failure_probability"] <= 1.0
    analysis = state["analysis"]
    SLMAnalysis.model_validate({k: analysis[k] for k in
                                ("risk_level", "likely_condition", "confidence",
                                 "evidence", "recommended_action", "urgency")})
    assert analysis["meta"]["valid"] is True
    chat = c.post("/assistant/chat", json={"machine_id": "M001",
                                           "question": "Why is this machine at risk?"}).json()
    assert "M001" in chat["text"]
    hist = c.get("/machines/M001/telemetry?limit=10").json()
    assert len(hist["readings"]) == 10
