"""Phase 1 foundation tests: config, logging, schemas, context builder."""
from src.config import load_config
from src.context_builder import build_context, render_prompt
from src.logging_utils import get_logger, log_event
from src.schemas import MachineContext, Prediction, SLMAnalysis, Telemetry


def _telemetry(cycle=10, s3=1590.0):
    return Telemetry(
        machine_id="M001",
        cycle=cycle,
        op_setting_1=0.0,
        op_setting_2=0.0,
        op_setting_3=100.0,
        sensors={"s3": s3, "s4": 1400.0, "s9": 9060.0, "s11": 47.5, "s12": 521.0},
    )


def test_config_loads():
    cfg = load_config("configs/development.yaml")
    assert cfg["slm"]["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg["predictive"]["rul_cap"] == 125


def test_logging_never_leaks_secrets(caplog):
    logger = get_logger("test")
    with caplog.at_level("INFO", logger="test"):
        log_event(logger, "auth", hf_token="secret-xyz")
    assert "secret-xyz" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_slm_schema_rejects_freeform():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SLMAnalysis.model_validate({"risk_level": "HIGH"})  # missing fields


def test_slm_schema_accepts_unknown():
    a = SLMAnalysis(
        risk_level="UNKNOWN",
        likely_condition="unknown_anomaly",
        confidence=0.2,
        evidence=["insufficient_evidence"],
        recommended_action="Continue monitoring; review at next check.",
        urgency="UNKNOWN",
    )
    assert a.risk_level == "UNKNOWN"


def test_context_builder_trends():
    pred = Prediction(
        failure_probability=0.8,
        anomaly_score=2.5,
        health_state="HIGH_RISK",
        rul_cycles=25,
        model_version="test",
        latency_ms=1.0,
    )
    ctx = build_context(_telemetry(10, 1590.0), _telemetry(9, 1580.0), pred, ["hpc_degradation"])
    assert isinstance(ctx, MachineContext)
    assert ctx.trends["s3_rel_change"] > 0
    prompt = render_prompt(ctx)
    assert "M001" in prompt and "hpc_degradation" in prompt
