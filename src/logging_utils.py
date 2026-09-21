"""Structured logging (never log secrets/tokens)."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

_REDACTED = "[REDACTED]"
_SECRET_KEYS = {"hf_token", "api_key", "token", "password", "secret"}


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (_REDACTED if k.lower() in _SECRET_KEYS else sanitize(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": time.time(), **sanitize(fields)}
    logger.info(json.dumps(payload))
