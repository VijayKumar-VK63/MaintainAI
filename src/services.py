"""Compatibility service factory backed by the single MaintainAIEngine.

New code should use `src.engine.MaintainAIEngine` directly. This module keeps
older imports (`ServiceFactory`, `create_services`, `machine_state`, `chat`)
working without duplicating the pipeline.
"""
from __future__ import annotations

from typing import Any

from src.engine import MaintainAIEngine


class ServiceFactory:
    """Creates and exposes the shared application engine."""

    def __init__(self, config_path: str = "configs/development.yaml"):
        self.engine = MaintainAIEngine(config_path)
        self.config = self.engine.config

    def get_simulator(self):
        return self.engine.sim

    def get_predictor(self):
        return self.engine.predictor

    def get_slm(self):
        return self.engine.slm

    def get_knowledge(self):
        return self.engine.knowledge

    def get_all(self) -> dict[str, Any]:
        return self.engine.services


def create_services(config_path: str = "configs/development.yaml") -> dict[str, Any]:
    return ServiceFactory(config_path).get_all()


def machine_state(services: dict[str, Any], machine_id: str) -> dict[str, Any]:
    engine = services.get("engine")
    if engine is None:
        raise RuntimeError("services dict has no engine; use create_services()")
    return engine.snapshot(machine_id)


def chat(services: dict[str, Any], machine_id: str, question: str) -> dict[str, Any]:
    engine = services.get("engine")
    if engine is None:
        raise RuntimeError("services dict has no engine; use create_services()")
    return engine.chat(machine_id, question)


__all__ = ["ServiceFactory", "create_services", "machine_state", "chat"]
