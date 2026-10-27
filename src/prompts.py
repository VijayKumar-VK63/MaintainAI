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

CHAT_SYSTEM_PROMPT = (
    "You are a predictive-maintenance assistant. Answer in 1-3 plain-text "
    "sentences. Use ONLY the validated assessment and evidence below. The "
    "validated assessment is authoritative. Do not output JSON. Do not invent "
    "sensor values, probabilities, thresholds, conditions, or maintenance "
    "history. If the user's question assumes a risk level that conflicts with "
    "the assessment, correct the premise first. If evidence is insufficient, "
    "say so. Do not make shutdown decisions; refer to site procedures and "
    "qualified personnel."
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


def _format_assessment(analysis: dict[str, Any]) -> str:
    evidence = ", ".join(str(item) for item in analysis.get("evidence", [])) or "none"
    confidence = float(analysis.get("confidence", 0.0))
    return (
        f"Validated assessment (authoritative): risk_level={analysis.get('risk_level')} "
        f"likely_condition={analysis.get('likely_condition')} "
        f"confidence={confidence:.2f} evidence={evidence} "
        f"recommended_action={analysis.get('recommended_action')} "
        f"urgency={analysis.get('urgency')}"
    )


def build_chat_messages(
    context: MachineContext,
    analysis: dict[str, Any],
    question: str,
    *,
    decision_threshold: float = 0.65,
    rul_cap: int = 125,
) -> list[dict[str, str]]:
    """Prompt for free-form chat, grounded in the validated analysis."""
    prediction = context.prediction
    lines = [
        f"Machine: {context.machine_id}",
        f"Cycle: {context.current.cycle}",
        f"Current sensors: {context.current.sensors}",
        f"Sensor trends: {context.trends}",
        "Predictive model: "
        f"failure_probability={prediction.failure_probability:.3f} "
        f"anomaly_score={prediction.anomaly_score:.3f} "
        f"health_state={prediction.health_state} "
        f"rul_cycles={prediction.rul_cycles}",
        f"Decision threshold: {decision_threshold:.2f}",
        f"RUL cap: {rul_cap} cycles",
        _format_assessment(analysis),
        f"Question: {question}",
        "Answer the question directly. Do not repeat the structured assessment as JSON.",
    ]
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
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
