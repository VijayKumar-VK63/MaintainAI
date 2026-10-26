"""Tests for canonical prompts, backend selection, and the shared engine."""
import pytest

from src.engine import (
    MaintainAIEngine,
    TelemetryStaleError,
    UnknownMachineError,
)
from src.prompts import SYSTEM_PROMPT, build_messages
from src.schemas import MachineContext, Prediction, Telemetry
from src.slm_eval import run_service_eval
from src.slm_service import (
    DemoBackend,
    OpenAICompatibleBackend,
    SLMService,
    extract_json,
)


def _context():
    tel = Telemetry(
        machine_id="M001", cycle=10, op_setting_1=0.0, op_setting_2=0.0,
        op_setting_3=100.0, sensors={"s3": 1590.0, "s4": 1400.0, "s9": 9060.0},
    )
    pred = Prediction(
        failure_probability=0.8, anomaly_score=2.0, health_state="HIGH_RISK",
        rul_cycles=25, model_version="test", latency_ms=1.0,
    )
    return MachineContext(
        machine_id="M001", current=tel, trends={"s3_rel_change": 0.01},
        prediction=pred, known_conditions=["hpc_degradation"],
    )


def test_canonical_messages_have_system_and_evidence():
    messages = build_messages(_context())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Machine: M001" in messages[1]["content"]
    assert "failure_probability=0.800" in messages[1]["content"]
    assert SYSTEM_PROMPT not in messages[1]["content"]


def test_extract_json_import_is_available_for_notebooks():
    assert extract_json('{"risk_level": "LOW"}') == {"risk_level": "LOW"}


def test_factory_defaults_to_demo():
    svc = SLMService(config={"slm": {"backend": "demo"}})
    assert isinstance(svc.backend, DemoBackend)
    assert svc.version == "demo-fallback"


def test_factory_selects_openai_from_endpoint():
    svc = SLMService(config={"slm": {"backend": "openai",
                                    "inference_endpoint": "http://localhost:8001"}})
    assert isinstance(svc.backend, OpenAICompatibleBackend)
    assert svc.backend.url == "http://localhost:8001/v1/chat/completions"


def test_openai_backend_sends_messages_and_auth(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return FakeResponse()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    backend = OpenAICompatibleBackend(
        "http://localhost:8001", api_key="secret", model_name="qwen"
    )
    out = backend.generate([{"role": "user", "content": "hello"}])
    assert out == "ok"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["messages"][0]["content"] == "hello"


def test_engine_errors_before_telemetry():
    engine = MaintainAIEngine("configs/development.yaml")
    with pytest.raises(TelemetryStaleError):
        engine.snapshot("M001")
    with pytest.raises(UnknownMachineError):
        engine.snapshot("NOPE")


def test_eval_service_path_uses_production_pipeline():
    svc = SLMService(config={"slm": {"backend": "demo"}})
    metrics = run_service_eval(svc, n_engines=3)
    assert metrics["n"] == 3
    assert metrics["validity_rate"] == 1.0
    assert metrics["n_failures"] == 0


def test_engine_snapshot_end_to_end():
    engine = MaintainAIEngine("configs/development.yaml")
    engine.sim.start()
    for _ in range(5):
        engine.sim.tick()
    state = engine.snapshot("M001")
    assert state["backend"] == engine.slm.version
    assert 0.0 <= state["prediction"]["failure_probability"] <= 1.0
    assert state["analysis"]["meta"]["valid"] is True
    assert engine.messages("M001")[0]["role"] == "system"
    assert engine.chat("M001", "What should we inspect?")["text"]
