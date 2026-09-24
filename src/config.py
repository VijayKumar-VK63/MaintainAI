"""YAML config loader with environment-variable overrides (no secrets in git)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    # Env overrides for deployment flexibility (config files stay secret-free).
    cfg.setdefault("slm", {})
    if os.getenv("SLM_MODEL_NAME"):
        cfg["slm"]["model_name"] = os.environ["SLM_MODEL_NAME"]
    if os.getenv("SLM_ADAPTER_PATH"):
        cfg["slm"]["adapter_path"] = os.environ["SLM_ADAPTER_PATH"]
    if os.getenv("SLM_INFERENCE_ENDPOINT"):
        cfg["slm"]["inference_endpoint"] = os.environ["SLM_INFERENCE_ENDPOINT"]
    if os.getenv("API_BASE_URL"):
        cfg.setdefault("api", {})["base_url"] = os.environ["API_BASE_URL"]
    return cfg
