"""Phase 5 tests: SLMService contract with mock backends (no GPU needed)."""
import pytest

from src.context_builder import build_context
from src.schemas import Prediction, Telemetry
from src.slm_eval import _evidence_f1, run_harness
from src.slm_service import DemoBackend, LlamaCppBackend, SLMService
from src.validation import controlled_fallback, extract_json


class GoodBackend:
    name = "mock-good"

    def generate(self, messages):
        assert messages[0]["role"] == "system"
        assert "Machine:" in messages[1]["content"]  # grounded in context
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


CRITICAL_ANALYSIS = {
    "risk_level": "CRITICAL",
    "likely_condition": "unknown_anomaly",
    "confidence": 0.45,
    "evidence": ["rul_low", "failure_probability_high"],
    "recommended_action": "escalate_immediate_review",
    "urgency": "IMMEDIATE",
    "meta": {"valid": True},
}


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
    chat = svc.chat(
        "Compare this machine to a car.",
        _context(),
        analysis=CRITICAL_ANALYSIS,
    )
    assert "temporarily unavailable" in chat["text"]


def test_demo_backend_tagged_and_valid():
    svc = SLMService(config={"slm": {"backend": "demo"}})
    out = svc.analyze(_context())
    assert out["meta"]["demo"] is True and out["meta"]["valid"] is True
    chat = svc.chat("What changed?", _context())
    assert "M001" in chat["text"]


class FreeTextBackend:
    name = "mock-free"

    def generate(self, messages):
        raise AssertionError("constrained JSON path must not be used for chat")

    def generate_free(self, messages, max_tokens=None, temperature=None):
        return "Inspect the HPC outlet temperature sensor."


def test_chat_uses_free_text_not_json_grammar():
    svc = SLMService(backend=FreeTextBackend())
    out = svc.chat(
        "Compare this machine to a car.",
        _context(),
        analysis=CRITICAL_ANALYSIS,
    )
    assert out["text"] == "Inspect the HPC outlet temperature sensor."
    assert out["meta"]["source"] == "slm"


def test_explicit_demo_backend_wins_over_endpoint():
    svc = SLMService(config={"slm": {
        "backend": "demo", "inference_endpoint": "http://example.invalid",
    }})
    assert isinstance(svc.backend, DemoBackend)


def test_llamacpp_backend_selected_from_config():
    svc = SLMService(config={"slm": {
        "backend": "llamacpp", "model_path": "base.gguf",
        "adapter_path": "lora.gguf", "n_threads": 4, "lora_scale": 0.5,
        "constrained": True,
    }})
    assert isinstance(svc.backend, LlamaCppBackend)
    assert svc.backend.adapter_path == "lora.gguf"
    assert svc.backend.n_threads == 4
    assert svc.backend.lora_scale == 0.5
    assert svc.backend.constrained is True
    assert svc.version == "llamacpp:base.gguf+lora.gguf"


def test_llamacpp_close_before_load_is_noop():
    LlamaCppBackend("base.gguf").close()


def test_slm_json_grammar_compiles():
    pytest.importorskip("llama_cpp")
    from llama_cpp import LlamaGrammar

    from src.slm_grammar import SLM_JSON_GRAMMAR

    assert "CRITICAL" in SLM_JSON_GRAMMAR
    assert "IMMEDIATE" in SLM_JSON_GRAMMAR
    LlamaGrammar.from_string(SLM_JSON_GRAMMAR)


def test_slm_json_grammar_compiles_after_full_context_build():
    pytest.importorskip("llama_cpp")
    from llama_cpp import LlamaGrammar

    from src.slm_eval import build_live_contexts
    from src.slm_grammar import SLM_JSON_GRAMMAR

    build_live_contexts(1)
    LlamaGrammar.from_string(SLM_JSON_GRAMMAR, verbose=False)


def test_llamacpp_without_path_raises(monkeypatch):
    monkeypatch.delenv("SLM_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="model_path"):
        SLMService(config={"slm": {"backend": "llamacpp"}})


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
