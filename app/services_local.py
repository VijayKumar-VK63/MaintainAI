"""Importable local pipeline for the Streamlit app (no Streamlit dependency).

Thin wrapper around the single MaintainAIEngine so tests exercise exactly what
the dashboard runs.
"""
from __future__ import annotations

from typing import Any

from src.engine import MaintainAIEngine


def create_services(config_path: str = "configs/development.yaml") -> dict[str, Any]:
    engine = MaintainAIEngine(config_path)
    return engine.services


def machine_state(svc: dict[str, Any], machine_id: str) -> dict[str, Any]:
    return svc["engine"].snapshot(machine_id)


def chat(svc: dict[str, Any], machine_id: str, question: str) -> dict[str, Any]:
    return svc["engine"].chat(machine_id, question)


__all__ = ["create_services", "machine_state", "chat"]
