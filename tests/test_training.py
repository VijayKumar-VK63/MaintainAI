"""Phase 6 tests: training format, resource audit, smoke-test artifacts."""
import json
from pathlib import Path

from src.train_slm import audit_resources, format_text


def test_format_contains_all_parts():
    t = format_text("SYS", "USER", '{"a": 1}', tokenizer=None)
    assert "SYS" in t and "USER" in t and '{"a": 1}' in t


def test_audit_reports_no_gpu_honestly():
    res = audit_resources()
    assert res["cuda"] is False and res["gpu"] is None
    assert "torch" in res


def test_smoke_adapter_artifacts_exist():
    d = Path("experiments/exp_001-smoke/adapter")
    assert (d / "adapter_model.safetensors").exists()
    cfg = json.loads((d / "adapter_config.json").read_text())
    assert cfg["r"] == 8 and cfg["task_type"] == "CAUSAL_LM"
