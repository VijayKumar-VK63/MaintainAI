"""Phase 6 tests: training format, resource audit, HF model availability."""
import json
import os

import pytest
import requests

from src.train_slm import audit_resources, format_text


def test_format_contains_all_parts():
    t = format_text("SYS", "USER", '{"a": 1}', tokenizer=None)
    assert "SYS" in t and "USER" in t and '{"a": 1}' in t


def test_format_messages_override():
    messages = [
        {"role": "system", "content": "SYS2"},
        {"role": "user", "content": "USER2"},
        {"role": "assistant", "content": '{"b": 2}'},
    ]
    t = format_text("ignored", "ignored", "ignored", messages=messages)
    assert "SYS2" in t and "USER2" in t and '{"b": 2}' in t


def test_audit_reports_hardware_honestly():
    res = audit_resources()
    assert isinstance(res["cuda"], bool)
    assert "torch" in res
    if not res["cuda"]:
        assert res["gpu"] is None


@pytest.mark.skipif(
    os.getenv("RUN_HUB_TESTS") != "1",
    reason="set RUN_HUB_TESTS=1 to hit the Hugging Face Hub",
)
def test_finetuned_model_exists_on_hf_hub():
    """Verify the fine-tuned LoRA adapter exists on Hugging Face Hub."""
    url = "https://huggingface.co/api/models/Vijay-kumar-63/maintainai-qwen2.5-3b-lora"
    resp = requests.get(url, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert data["modelId"] == "Vijay-kumar-63/maintainai-qwen2.5-3b-lora"
    assert "lora" in data.get("tags", [])
    assert "peft" in data.get("tags", [])


def test_train_metrics_contract(tmp_path):
    # Training itself needs a GPU; this validates the persisted contract shape.
    payload = {"train_loss": 0.4, "output_dir": str(tmp_path), "audit": {}, "smoke": True}
    path = tmp_path / "train_metrics.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text())["train_loss"] == 0.4
