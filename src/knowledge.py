"""KnowledgeService: loads controlled maintenance knowledge from YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class KnowledgeService:
    """Loads and provides access to failure modes, actions, sensor relationships."""

    def __init__(self, knowledge_dir: str | Path = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self._failure_modes: dict[str, Any] = {}
        self._maintenance_actions: dict[str, Any] = {}
        self._sensor_relationships: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        self._failure_modes = self._load_yaml("failure_modes.yaml")
        self._maintenance_actions = self._load_yaml("maintenance_actions.yaml")
        self._sensor_relationships = self._load_yaml("sensor_relationships.yaml")

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self.knowledge_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def failure_modes(self) -> dict[str, Any]:
        return self._failure_modes

    @property
    def maintenance_actions(self) -> dict[str, Any]:
        return self._maintenance_actions

    @property
    def sensor_relationships(self) -> dict[str, Any]:
        return self._sensor_relationships

    def get_condition_info(self, condition_id: str) -> dict[str, Any] | None:
        for cond in self._failure_modes.get("conditions", []):
            if cond.get("id") == condition_id:
                return cond
        return None

    def get_action_info(self, action_id: str) -> dict[str, Any] | None:
        actions = self._maintenance_actions.get("actions", {})
        # Handle both list of dicts with 'id' and dict keyed by action name
        if isinstance(actions, list):
            for action in actions:
                if action.get("id") == action_id:
                    return action
        elif isinstance(actions, dict):
            return actions.get(action_id)
        return None

    def get_informative_sensors(self) -> list[str]:
        return self._sensor_relationships.get("informative_sensors", [])

    def get_sensor_relationships(self) -> list[dict[str, Any]]:
        return self._sensor_relationships.get("relationships", [])

    def _iter_conditions(self) -> list[dict[str, Any]]:
        conds = self._failure_modes.get("conditions", [])
        if isinstance(conds, list):
            return conds
        return []

    def _iter_actions(self) -> list[dict[str, Any]]:
        actions = self._maintenance_actions.get("actions", {})
        if isinstance(actions, list):
            return actions
        # Convert dict to list of dicts with id
        return [{"id": k, **v} for k, v in actions.items()]

    def format_for_slm(self) -> str:
        """Format knowledge for SLM context injection."""
        lines = ["=== MAINTENANCE KNOWLEDGE BASE ==="]
        lines.append("\nKNOWN CONDITIONS:")
        for cond in self._iter_conditions():
            lines.append(
                f"- {cond.get('id', 'unknown')}: {cond.get('description', '')} | "
                f"evidence: {', '.join(cond.get('typical_evidence', []))}"
            )
        lines.append("\nRECOMMENDED ACTIONS:")
        for action in self._iter_actions():
            lines.append(
                f"- {action.get('id', 'unknown')}: {action.get('description', '')} | when: {action.get('when', '')}"
            )
        lines.append("\nSENSOR RELATIONSHIPS:")
        for rel in self.get_sensor_relationships():
            lines.append(
                f"- {', '.join(rel.get('sensors', []))}: {rel.get('meaning', '')}"
            )
        return "\n".join(lines)