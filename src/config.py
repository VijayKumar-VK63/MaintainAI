"""YAML config loader with environment-variable overrides (no secrets in git)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:  # optional dependency; listed in requirements.txt
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv()


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    # Env overrides for deployment flexibility (config files stay secret-free).
    cfg.setdefault("slm", {})
    slm = cfg["slm"]
    if os.getenv("SLM_MODEL_NAME"):
        slm["model_name"] = os.environ["SLM_MODEL_NAME"]
    if os.getenv("SLM_ADAPTER_PATH"):
        slm["adapter_path"] = os.environ["SLM_ADAPTER_PATH"]
    if os.getenv("SLM_MODEL_PATH"):
        slm["model_path"] = os.environ["SLM_MODEL_PATH"]
    if os.getenv("SLM_INFERENCE_ENDPOINT"):
        slm["inference_endpoint"] = os.environ["SLM_INFERENCE_ENDPOINT"]
    if os.getenv("SLM_BACKEND"):
        slm["backend"] = os.environ["SLM_BACKEND"]
    if os.getenv("SLM_INFERENCE_PATH"):
        slm["inference_path"] = os.environ["SLM_INFERENCE_PATH"]
    if os.getenv("SLM_API_KEY"):
        slm["api_key"] = os.environ["SLM_API_KEY"]
    if os.getenv("SLM_QUANTIZATION"):
        slm["quantization"] = os.environ["SLM_QUANTIZATION"]
    constrained = _env_bool("SLM_CONSTRAINED")
    if constrained is not None:
        slm["constrained"] = constrained
    for key, env_name in (
        ("n_ctx", "SLM_N_CTX"),
        ("n_threads", "SLM_N_THREADS"),
        ("n_batch", "SLM_N_BATCH"),
        ("max_new_tokens", "SLM_MAX_NEW_TOKENS"),
        ("seed", "SLM_SEED"),
    ):
        value = _env_int(env_name)
        if value is not None:
            slm[key] = value
    for key, env_name in (
        ("lora_scale", "SLM_LORA_SCALE"),
        ("temperature", "SLM_TEMPERATURE"),
    ):
        value = _env_float(env_name)
        if value is not None:
            slm[key] = value
    if os.getenv("API_BASE_URL"):
        cfg.setdefault("api", {})["base_url"] = os.environ["API_BASE_URL"]

    root = Path(__file__).resolve().parents[1]
    for key in ("model_path", "adapter_path"):
        value = slm.get(key)
        if value and not Path(str(value)).is_absolute():
            candidate = (root / str(value)).resolve()
            if candidate.exists():
                slm[key] = str(candidate)
    return cfg
