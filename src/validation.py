"""ValidationService: structured output validation and safe fallbacks."""
from __future__ import annotations

import json
import re
from typing import Any

from src.schemas import SLMAnalysis

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from text."""
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def controlled_fallback(reason: str) -> dict[str, Any]:
    """Return a safe UNKNOWN fallback when validation fails."""
    return {
        "risk_level": "UNKNOWN",
        "likely_condition": "unknown_anomaly",
        "confidence": 0.0,
        "evidence": ["insufficient_evidence"],
        "recommended_action": "continue_monitoring",
        "urgency": "UNKNOWN",
        "meta": {"backend": "fallback", "valid": False, "reason": reason},
    }


def validate_slm_output(raw: str, model_name: str) -> dict[str, Any]:
    """Validate SLM output against schema; return validated dict with meta."""
    obj = extract_json(raw)
    if obj is None:
        fb = controlled_fallback("no JSON object in output")
        fb["meta"]["model"] = model_name
        return fb
    try:
        SLMAnalysis.model_validate(obj)
        obj["meta"] = {"backend": "validated", "valid": True, "model": model_name}
        return obj
    except Exception as exc:
        fb = controlled_fallback(f"schema validation failed: {exc}")
        fb["meta"]["model"] = model_name
        return fb


def validate_slm_output_with_retry(
    generate_fn, prompt: str, model_name: str, max_retries: int = 1
) -> dict[str, Any]:
    """Call generate_fn(prompt) and validate; retry on validation failure."""
    last_err = "no output"
    for _ in range(max_retries + 1):
        try:
            raw = generate_fn(prompt)
        except Exception as exc:
            last_err = f"generation failed: {type(exc).__name__}"
            continue
        obj = extract_json(raw)
        if obj is None:
            last_err = "no JSON object in output"
            continue
        try:
            SLMAnalysis.model_validate(obj)
            obj["meta"] = {"backend": "validated", "valid": True, "model": model_name}
            return obj
        except Exception as exc:
            last_err = f"schema validation failed: {exc}"
    fb = controlled_fallback(last_err)
    fb["meta"]["model"] = model_name
    return fb