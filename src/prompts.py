"""Canonical prompt construction shared by training, evaluation, and inference.

Single source of truth: training examples, the evaluation harness, and the
serving backends all build the same chat messages here. Backends are
responsible for applying the model-specific chat template.
"""
from __future__ import annotations

from typing import Any

from src.schemas import MachineContext

SYSTEM_PROMPT = (
    "You are a predictive maintenance assistant. "
    "Analyze machine condition ONLY from the provided evidence. "
    "Do not invent sensor measurements, maintenance history, inspections, "
    "or unsupported failure modes. Respond with structured JSON only."
)


def build_user_message(context: MachineContext) -> str:
    """Render the evidence block (no system instructions)."""
    lines = [
        f"Machine: {context.machine_id}",
        f"Cycle: {context.current.cycle}",
        f"Current sensors: {context.current.sensors}",
        f"Sensor trends: {context.trends}",
        "Predictive model: "
        f"failure_probability={context.prediction.failure_probability:.3f} "
        f"anomaly_score={context.prediction.anomaly_score:.3f} "
        f"health_state={context.prediction.health_state} "
        f"rul_cycles={context.prediction.rul_cycles}",
        f"Known possible conditions: {context.known_conditions}",
        "Task: provide risk_level, likely_condition, confidence, evidence, "
        "recommended_action, urgency as JSON.",
    ]
    return "\n".join(lines)


def build_messages(context: MachineContext) -> list[dict[str, str]]:
    """Chat messages for serving/evaluation (system + user)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(context)},
    ]


def render_prompt(context: MachineContext) -> str:
    """Legacy flat-text view (system + user). Prefer build_messages()."""
    return f"{SYSTEM_PROMPT}\n{build_user_message(context)}"


def render_chat(
    tokenizer: Any | None,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Apply the tokenizer chat template, with a deterministic ChatML fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)
