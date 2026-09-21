"""Phase 5 tests: SLMService contract with mock backends (no GPU needed)."""
import pytest

from src.context_builder import build_context
from src.schemas import Prediction, Telemetry
from src.slm_eval import _evidence_f1, run_harness
from src.slm_service import SLMService
from src.validation import controlled_fallback, extract_json


class GoodBackend:
    name = "mock-good"

    def generate(self, prompt: str) -> str:
        assert "Machine:" in prompt  # grounded in context, not a bare question
        return ('{"risk_level": "HIGH", "likely_condition": "hpc_degradation", '
                '"confidence": 0.8, "evidence": ["s3_above_baseline"], '
                '"recommended_action": "schedule_hpc_inspection", "urgency": "PRIORITY"}')


class GarbageBackend:
    name = "mock-garbage"

    def generate(self, prompt: str) -> str:
        return "the machine seems kinda bad, trust me"


class ExplodingBackend:
    name = "mock-boom"

    def generate(self, prompt: str) -> str:
        raise RuntimeError("CUDA OOM (simulated)")


def _context():
    tel = Telemetry(machine_id="M001", cycle=200, op_setting_1=0.0,
                    op_setting_2=0.0, op_setting_3=100.0,
                    sensors={"s3": 1610.0, "s4": 1430.0, "s9": 9200.0})
    pred = Prediction(failure_probability=0.9, anomaly_score=2.0,
                      health_state="CRITICAL", rul_cycles=12,
                      model_version="test", latency_ms=1.0)
    return build_context(tel, None, pred, ["hpc_degradation"])


def test_valid_output_passes_through():
    svc = SLMService(backend=GoodBackend())
    out = svc.analyze(_context())
    assert out["meta"]["valid"] is True
    assert out["risk_level"] == "HIGH" and out["urgency"] == "PRIORITY"


def test_garbage_output_becomes_unknown_fallback():
    svc = SLMService(backend=GarbageBackend())
    out = svc.analyze(_context())
    assert out["risk_level"] == "UNKNOWN"
    assert out["meta"]["valid"] is False


def test_backend_crash_never_raises():
    svc = SLMService(backend=ExplodingBackend())
    out = svc.analyze(_context())  # must not raise
    assert out["meta"]["valid"] is False
    chat = svc.chat("Why high risk?", _context())
    assert "temporarily unavailable" in chat["text"]


def test_demo_backend_tagged_and_valid():
    svc = SLMService()  # default demo
    out = svc.analyze(_context())
    assert out["meta"]["demo"] is True and out["meta"]["valid"] is True
    chat = svc.chat("What changed?", _context())
    assert "M001" in chat["text"]


def test_extract_json_none_without_object():
    assert extract_json("no json here") is None
    assert controlled_fallback("x")["risk_level"] == "UNKNOWN"


def test_harness_scores_known_backend():
    exs = [{"target": ('{"risk_level": "HIGH", "likely_condition": "hpc_degradation",'
                       ' "confidence": 0.8, "evidence": ["s3_above_baseline"],'
                       ' "recommended_action": "schedule_hpc_inspection", "urgency": "PRIORITY"}')}]
    svc = SLMService(backend=GoodBackend())
    m = run_harness(exs, lambda e: svc.analyze(_context()))
    assert m["risk_accuracy"] == 1.0 and m["validity_rate"] == 1.0
    assert _evidence_f1(["a", "b"], ["a"]) == pytest.approx(2 / 3)
