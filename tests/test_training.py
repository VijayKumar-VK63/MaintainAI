"""Phase 6 tests: training format, resource audit, HF model availability."""
import json
from pathlib import Path

import requests

from src.train_slm import audit_resources, format_text


def test_format_contains_all_parts():
    t = format_text("SYS", "USER", '{"a": 1}', tokenizer=None)
    assert "SYS" in t and "USER" in t and '{"a": 1}' in t


def test_audit_reports_no_gpu_honestly():
    res = audit_resources()
    assert res["cuda"] is False and res["gpu"] is None
    assert "torch" in res


def test_finetuned_model_exists_on_hf_hub():
    """Verify the fine-tuned LoRA adapter exists on Hugging Face Hub."""
    url = "https://huggingface.co/api/models/Vijay-kumar-63/maintainai-qwen2.5-3b-lora"
    resp = requests.get(url, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert data["modelId"] == "Vijay-kumar-63/maintainai-qwen2.5-3b-lora"
    assert "lora" in data.get("tags", [])
    assert "peft" in data.get("tags", [])
