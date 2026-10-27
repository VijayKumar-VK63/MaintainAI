"""Deterministic, assessment-grounded answers for common maintenance questions.

The fine-tuned adapter is trained for structured JSON analysis, not free-form
chat. Common UI questions are answered here from the validated analysis so the
risk level, numbers, RUL, and recommended action cannot drift. Unrecognized
questions fall back to the SLM with a grounded prompt.
"""
from __future__ import annotations

import re
from typing import Any

from src.schemas import MachineContext

WATCH_SENSORS = ("s3", "s4", "s9", "s11", "s12")
TREND_THRESHOLD = 0.01


def _norm(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _matches(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def classify_intent(question: str) -> str | None:
    """Map a user question to a deterministic answer intent."""
    q = _norm(question)
    if not q:
        return "help"
    if _matches(
        q,
        r"\b(ignore|disregard|forget|override|jailbreak|system prompt)\b",
        r"\bpretend\b",
        r"\bsay\s+(critical|high risk)\b",
    ):
        return "injection"
    if _matches(q, r"\b(shut\s?down|stop the machine|turn off|emergency stop|power off)\b"):
        return "shutdown"
    if _matches(q, r"\b(guarantee|100\s?%|certain|will it fail|can it fail|never fail)\b"):
        return "guarantee"
    if _matches(q, r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b"):
        return "greeting"
    if _matches(q, r"\b(help|what can you do|how can you help|capabilities)\b"):
        return "help"
    if _matches(q, r"\b(weather|stock|sports|joke|politics|news|story|who are you)\b"):
        return "out_of_scope"
    if _matches(
        q,
        r"\b(next \d+ cycles|future cycles|what happens if|run for|continue running|forecast)\b",
    ):
        return "forecast"
    if _matches(
        q,
        r"\b(uncertain|uncertainty|confidence|how sure|limitations|what don't you know|missing)\b",
    ):
        return "uncertainty"
    if _matches(q, r"\b(changed|change since|difference|last cycle|previous cycle)\b"):
        return "changed"
    if _matches(q, r"\b(sensor|sensors|trend|trends|abnormal|reading|readings)\b"):
        return "sensors"
    if _matches(q, r"\b(rul|remaining useful life|remaining life|cycles left|life left)\b"):
        return "rul"
    if _matches(q, r"\b(evidence|why do you think|what supports|what indicates|based on)\b"):
        return "evidence"
    if _matches(q, r"^why\b", r"\bwhy (is|are|does|did)\b"):
        return "why"
    if _matches(
        q,
        r"\b(what should|what do|next step|next action|recommend|maintenance|inspect|"
        r"fix|resolve|do now|what to do)\b",
    ):
        return "action"
    if _matches(q, r"\b(urgent|urgency|priority|how soon|when should)\b"):
        return "urgency"
    if _matches(q, r"\b(condition|fault|wrong|diagnosis|failure mode|likely)\b"):
        return "condition"
    if _matches(q, r"\b(what is happening|what's happening|status|summary|overview|rationale|situation|how is|how are)\b"):
        return "summary"
    if _matches(q, r"\b(risk|risky|safe|safety|danger)\b"):
        return "risk"
    return None


def assessment_valid(analysis: dict[str, Any] | None) -> bool:
    if not analysis:
        return False
    if analysis.get("meta", {}).get("valid", True) is False:
        return False
    return analysis.get("risk_level") in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def _risk(analysis: dict[str, Any]) -> str:
    return str(analysis.get("risk_level", "UNKNOWN"))


def _confidence(analysis: dict[str, Any]) -> float:
    try:
        return float(analysis.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _condition(analysis: dict[str, Any]) -> str:
    condition = str(analysis.get("likely_condition") or "unknown")
    if _risk(analysis) == "LOW":
        return "The likely condition is healthy_operation; no fault pattern is indicated."
    return f"The likely condition is {condition}."


def _action(analysis: dict[str, Any]) -> str:
    return str(analysis.get("recommended_action") or "no action available").replace("_", " ")


def _urgency(analysis: dict[str, Any]) -> str:
    return str(analysis.get("urgency") or "UNKNOWN")


def _evidence(analysis: dict[str, Any]) -> str:
    evidence = analysis.get("evidence") or ["no specific evidence codes"]
    return ", ".join(str(item) for item in evidence)


def _reason(context: MachineContext, decision_threshold: float) -> str:
    prediction = context.prediction
    direction = "above" if prediction.failure_probability >= decision_threshold else "below"
    return (
        f"failure probability {prediction.failure_probability:.1%} is {direction} the "
        f"{decision_threshold:.0%} decision threshold and health state is "
        f"{prediction.health_state}"
    )


def _rul(context: MachineContext, rul_cap: int) -> str:
    rul = context.prediction.rul_cycles
    if rul is None:
        return "RUL is unavailable."
    if rul <= 30:
        band = "low"
    elif rul <= 60:
        band = "limited"
    else:
        band = "a healthy margin"
    return f"RUL is {rul} cycles (cap {rul_cap}); this is {band}."


def _safety(analysis: dict[str, Any]) -> str:
    if _risk(analysis) == "LOW":
        return "No immediate repair or shutdown is indicated."
    return "Involve a qualified operator and follow site procedures."


def _trends(context: MachineContext) -> str:
    parts: list[str] = []
    changed: list[str] = []
    for name in WATCH_SENSORS:
        value = context.trends.get(f"{name}_rel_change")
        if value is None:
            continue
        parts.append(f"{name} {value * 100:+.2f}%")
        if abs(value) >= TREND_THRESHOLD:
            changed.append(name)
    if not parts:
        return "No watched sensor trend data is available."
    text = f"Watched sensor changes since the previous cycle: {', '.join(parts)}."
    if changed:
        text += f" Above the 1% trend threshold: {', '.join(changed)}."
        values = []
        for name in changed:
            current = context.current.sensors.get(name)
            if current is not None:
                values.append(f"{name}={current:.1f}")
        if values:
            text += f" Current values: {', '.join(values)}."
    else:
        text += " No watched sensor moved more than 1%, so no trend is abnormal by that threshold."
    return text


def invalid_reply(context: MachineContext) -> str:
    prediction = context.prediction
    return (
        "The structured assessment is unavailable, so I can't give a reliable "
        "maintenance recommendation. Raw prediction: "
        f"failure_probability={prediction.failure_probability:.3f}, "
        f"health_state={prediction.health_state}, RUL={prediction.rul_cycles} cycles. "
        "Please retry or inspect the model logs."
    )


def summary_reply(
    context: MachineContext,
    analysis: dict[str, Any],
    decision_threshold: float,
    rul_cap: int,
) -> str:
    return (
        f"{context.machine_id} is at {_risk(analysis)} risk "
        f"({_confidence(analysis):.0%} confidence). "
        f"The predictive model shows {_reason(context, decision_threshold)}. "
        f"{_condition(analysis)} Recommended action: {_action(analysis)} "
        f"({_urgency(analysis)}). {_rul(context, rul_cap)}"
    )


def guard_chat_text(
    text: str,
    context: MachineContext,
    analysis: dict[str, Any],
    *,
    decision_threshold: float = 0.65,
    rul_cap: int = 125,
) -> str:
    """Replace a model answer that contradicts the validated assessment."""
    cleaned = text.strip()
    if not cleaned:
        return summary_reply(context, analysis, decision_threshold, rul_cap)

    if re.search(r'"(?:risk_level|likely_condition)"\s*:', cleaned, re.IGNORECASE):
        return summary_reply(context, analysis, decision_threshold, rul_cap)
    if re.search(r"\brisk_level\s*:", cleaned, re.IGNORECASE):
        return summary_reply(context, analysis, decision_threshold, rul_cap)

    expected = _risk(analysis)
    mentioned = re.findall(
        r"(?:risk(?:\s*level)?\s*(?:is|:|=)?\s*|at\s+)"
        r"(LOW|MEDIUM|HIGH|CRITICAL)\b",
        cleaned,
        re.IGNORECASE,
    )
    if mentioned and any(item.upper() != expected for item in mentioned):
        return summary_reply(context, analysis, decision_threshold, rul_cap)

    prediction = context.prediction
    probability = re.search(
        r"failure[_ ]?probability[^0-9]{0,25}([0-9]*\.?[0-9]+)\s*(%?)",
        cleaned,
        re.IGNORECASE,
    )
    if probability:
        value = float(probability.group(1))
        if probability.group(2) == "%":
            value /= 100.0
        if abs(value - prediction.failure_probability) > 0.02:
            return summary_reply(context, analysis, decision_threshold, rul_cap)

    rul = re.search(r"\bRUL\b[^0-9]{0,25}(\d+)", cleaned, re.IGNORECASE)
    if rul and prediction.rul_cycles is not None:
        value = int(rul.group(1))
        if value not in (prediction.rul_cycles, rul_cap):
            return summary_reply(context, analysis, decision_threshold, rul_cap)

    if not re.search(r"[.!?][\"')\]]*$", cleaned):
        end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
        if end < 0:
            return summary_reply(context, analysis, decision_threshold, rul_cap)
        cleaned = cleaned[: end + 1].strip()

    return cleaned


def answer_from_assessment(
    context: MachineContext,
    analysis: dict[str, Any],
    question: str,
    *,
    decision_threshold: float = 0.65,
    rul_cap: int = 125,
) -> str | None:
    """Return a deterministic answer, or None for the grounded SLM fallback."""
    intent = classify_intent(question)
    if intent is None:
        return None

    machine = context.machine_id
    if intent == "greeting":
        return (
            f"Hello. Ask me about {machine}'s current risk, evidence, sensor "
            "trends, RUL, urgency, or recommended action."
        )
    if intent == "help":
        return (
            f"I can explain {machine}'s risk level, evidence, sensor trends, "
            "RUL, urgency, recommended action, and what changed since the "
            "previous cycle. I cannot make shutdown decisions."
        )
    if intent == "out_of_scope":
        return (
            f"I can only answer questions about {machine}'s condition, evidence, "
            "sensor trends, RUL, and recommended maintenance."
        )
    if intent == "injection":
        if not assessment_valid(analysis):
            return (
                "I can't ignore the machine evidence or override the assessment. "
                "The structured assessment is currently unavailable."
            )
        return (
            "I can't ignore the machine evidence or override the validated "
            f"assessment. {summary_reply(context, analysis, decision_threshold, rul_cap)}"
        )
    if not assessment_valid(analysis):
        return invalid_reply(context)

    risk = _risk(analysis)
    confidence = _confidence(analysis)
    urgency = _urgency(analysis)
    action = _action(analysis)
    reason = _reason(context, decision_threshold)
    evidence = _evidence(analysis)

    if intent == "risk":
        return (
            f"{machine} is at {risk} risk ({confidence:.0%} confidence). "
            f"The predictive model shows {reason}. {_condition(analysis)} "
            f"Recommended action: {action} ({urgency}). {_rul(context, rul_cap)}"
        )
    if intent == "why":
        if risk == "LOW":
            return (
                f"{machine} is not at high risk. Risk is LOW because {reason}; "
                f"the validated evidence is {evidence}. {_rul(context, rul_cap)}"
            )
        return (
            f"{machine} is at {risk} risk because {reason}; the validated "
            f"evidence is {evidence}. {_rul(context, rul_cap)}"
        )
    if intent == "evidence":
        prediction = context.prediction
        return (
            f"The validated assessment is based on: {evidence}. Predictive "
            f"values: failure_probability={prediction.failure_probability:.3f}, "
            f"anomaly_score={prediction.anomaly_score:.3f}, "
            f"health_state={prediction.health_state}, "
            f"RUL={prediction.rul_cycles} cycles."
        )
    if intent == "sensors":
        return f"For {machine}: {_trends(context)} {_condition(analysis)}"
    if intent == "rul":
        prediction = context.prediction
        return (
            f"{_rul(context, rul_cap)} Failure probability is "
            f"{prediction.failure_probability:.1%} and health state is "
            f"{prediction.health_state}."
        )
    if intent == "action":
        return (
            f"For {machine}, the recommended action is {action} with {urgency} "
            f"urgency. {_safety(analysis)}"
        )
    if intent == "urgency":
        return f"Urgency is {urgency}. {_safety(analysis)} {_rul(context, rul_cap)}"
    if intent == "shutdown":
        if risk == "LOW":
            return (
                f"No immediate shutdown is indicated: {machine} is at LOW risk "
                f"with {urgency} urgency. Follow site safety procedures and "
                "escalate to a qualified operator if conditions change. The "
                "assistant does not make shutdown decisions."
            )
        return (
            f"The current assessment is {risk} with {urgency} urgency. A "
            "qualified operator should follow site emergency procedures; the "
            "assistant does not make shutdown decisions."
        )
    if intent == "guarantee":
        prediction = context.prediction
        return (
            f"No. The model estimates failure probability "
            f"{prediction.failure_probability:.1%}; this is a probability, not "
            f"a guarantee. {_safety(analysis)}"
        )
    if intent == "forecast":
        return (
            "I can't simulate future cycles or predict a specific failure time. "
            f"At the current assessment, {machine} is at {risk} risk with "
            f"{urgency} urgency. Continue monitoring and watch the listed sensor "
            "trends."
        )
    if intent == "uncertainty":
        return (
            f"Confidence in the current assessment is {confidence:.0%}. It uses "
            f"the predictive model and the validated evidence ({evidence}); it "
            "does not include inspection history, maintenance records, or future "
            "operating changes."
        )
    if intent == "changed":
        return f"For {machine}: {_trends(context)} {_rul(context, rul_cap)}"
    if intent == "condition":
        return (
            f"{_condition(analysis)} Confidence is {confidence:.0%}. Evidence: "
            f"{evidence}."
        )
    if intent == "summary":
        return summary_reply(context, analysis, decision_threshold, rul_cap)
    return None


__all__ = [
    "answer_from_assessment",
    "assessment_valid",
    "classify_intent",
    "guard_chat_text",
    "invalid_reply",
    "summary_reply",
]
