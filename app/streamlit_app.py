"""MaintainAI product console: telemetry, prediction, assessment, and assistant."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:  # Preload before pandas/plotly/streamlit on Windows (DLL conflict).
    import llama_cpp  # noqa: F401
except ImportError:
    pass

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services_local import chat, create_services, machine_state  # noqa: E402

st.set_page_config(
    page_title="MaintainAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud exposes secrets via st.secrets, not as environment variables.
# Bridge the SLM/API settings so src.config sees the same values locally/cloud.
try:
    for _key in (
        "SLM_BACKEND",
        "SLM_INFERENCE_ENDPOINT",
        "SLM_INFERENCE_PATH",
        "SLM_API_KEY",
        "HF_TOKEN",
        "SLM_MODEL_NAME",
        "SLM_MODEL_PATH",
        "SLM_ADAPTER_PATH",
        "SLM_LORA_SCALE",
        "SLM_CONSTRAINED",
        "SLM_N_CTX",
        "SLM_N_THREADS",
        "SLM_N_BATCH",
        "SLM_MAX_NEW_TOKENS",
        "SLM_TEMPERATURE",
        "SLM_CHAT_MAX_NEW_TOKENS",
        "SLM_CHAT_TEMPERATURE",
        "SLM_QUANTIZATION",
    ):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

st.title("MaintainAI")
st.markdown("##### Predict failures. Explain the evidence. Recommend the next action.")
st.caption(
    "Simulated turbofan telemetry | XGBoost risk and RUL | grounded Qwen2.5-3B "
    "maintenance assistant"
)


@st.cache_resource
def _services():
    return create_services()


try:
    svc = _services()
except Exception as exc:
    st.error("The maintenance assistant could not start.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Local: run `python scripts/prepare_local_slm.py` and set "
        "`SLM_MODEL_PATH` to the GGUF. Cloud: set `SLM_BACKEND=openai` and "
        "`SLM_INFERENCE_ENDPOINT` in Streamlit secrets."
    )
    st.stop()

for key, default in (
    ("running", False),
    ("prob_hist", []),
    ("chat_hist", []),
    ("last_prob_cycle", None),
):
    st.session_state.setdefault(key, default)

STATUS_COLOR = {
    "HEALTHY": "green",
    "WARNING": "orange",
    "HIGH_RISK": "red",
    "CRITICAL": "red",
}
QUICK_PROMPTS = (
    "What is the current risk level?",
    "Why is the machine at risk?",
    "What evidence supports your assessment?",
    "What is the RUL, and is it low?",
    "What should the operator do next?",
    "Should we shut down the machine immediately?",
)

DECISION_THRESHOLD = float(
    svc["engine"].config.get("predictive", {}).get("decision_threshold", 0.65)
)
RUL_CAP = int(svc["engine"].config.get("predictive", {}).get("rul_cap", 125))
FEATURE_WINDOW = int(
    svc["engine"].config.get("predictive", {}).get("feature_window", 30)
)


def _pretty(value: object) -> str:
    return str(value).replace("_", " ").strip().capitalize()


with st.sidebar:
    st.header("Operations")
    st.caption("Live simulation")
    machine = st.selectbox("Machine", sorted(svc["sim"].machines))
    if st.session_state.get("active_machine") != machine:
        st.session_state.active_machine = machine
        st.session_state.prob_hist = []
        st.session_state.chat_hist = []
        st.session_state.state_cache = {}
        st.session_state.last_prob_cycle = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", type="primary", use_container_width=True):
            svc["sim"].start()
            st.session_state.running = True
    with col2:
        if st.button("Pause", use_container_width=True):
            svc["sim"].stop()
            st.session_state.running = False
    if st.button("Reset", use_container_width=True):
        svc["sim"].reset()
        st.session_state.update(
            running=False,
            prob_hist=[],
            chat_hist=[],
            state_cache={},
            last_prob_cycle=None,
        )

    st.divider()
    st.subheader("Scenario")
    scenario = st.selectbox(
        "Operating condition",
        ["NORMAL", "DEGRADATION", "CRITICAL", "FAILURE"],
        label_visibility="collapsed",
    )
    if st.button("Apply scenario", use_container_width=True):
        svc["sim"].set_scenario(machine, scenario)
        st.success(f"{machine} set to {scenario.lower()}.")
    speed = st.slider("Cycles per refresh", 1, 10, 3)
    auto = st.checkbox("Auto-refresh", value=False)

    st.divider()
    st.caption(f"Assistant: {svc['slm'].version}")
    if svc["slm"].version.startswith("demo"):
        st.warning(
            "Demo rule mirror active. Configure `SLM_MODEL_PATH` or "
            "`SLM_INFERENCE_ENDPOINT` to use the fine-tuned model."
        )
    st.caption("Decision support only. Qualified personnel make maintenance decisions.")


if st.session_state.running:
    try:
        for _ in range(speed):
            svc["sim"].tick()
    except RuntimeError as exc:
        st.warning(str(exc))
        st.session_state.running = False

state = None
try:
    buf = svc["sim"].machines[machine].buffer
    if buf:
        last = buf[-1]
        key = (machine, last["cycle"], last["timestamp"])
        cache = st.session_state.setdefault("state_cache", {})
        state = cache.get(key)
        if state is None:
            state = machine_state(svc, machine)
            cache[key] = state
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
except RuntimeError as exc:
    st.warning(str(exc))

if state is None:
    st.subheader("Start a live simulation")
    st.write(
        "Use the sidebar to start telemetry, then watch MaintainAI move from "
        "raw sensor readings to a validated maintenance recommendation."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Predict**")
        st.caption("XGBoost estimates failure risk and remaining useful life from causal sensor features.")
    with c2:
        st.markdown("**Explain**")
        st.caption("The fine-tuned assistant grounds every answer in the validated assessment.")
    with c3:
        st.markdown("**Act**")
        st.caption("Operators get a clear action, urgency, and the evidence behind it.")
else:
    pred, analysis = state["prediction"], state["analysis"]
    cycle = state["telemetry"]["cycle"]
    if st.session_state.get("last_prob_cycle") != cycle:
        st.session_state.prob_hist.append(pred["failure_probability"])
        st.session_state.prob_hist = st.session_state.prob_hist[-200:]
        st.session_state.last_prob_cycle = cycle

    health = pred["health_state"]
    st.subheader(f"{machine} - :{STATUS_COLOR.get(health, 'gray')}[{health}]")
    st.caption(f"Cycle {cycle} | {state['telemetry']['timestamp']}")

    if analysis.get("meta", {}).get("demo"):
        st.caption("Demo backend active. Responses come from the rule mirror, not the fine-tuned model.")
    elif not analysis.get("meta", {}).get("valid", True):
        st.warning(
            "The assistant returned an invalid assessment. Showing a controlled "
            f"UNKNOWN result. Reason: {analysis.get('meta', {}).get('reason', 'unknown')}."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Failure probability",
        f"{pred['failure_probability']:.1%}",
        help=f"Risk classifier decision threshold: {DECISION_THRESHOLD:.0%}",
    )
    rul = pred.get("rul_cycles")
    m2.metric("Remaining useful life", f"{rul} cycles" if rul is not None else "n/a")
    m3.metric("Anomaly score", f"{pred['anomaly_score']:.2f}")
    m4.metric("Health state", health)

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Live telemetry")
        hist = state["history"]
        cycles = [reading["cycle"] for reading in hist]
        fig = go.Figure()
        for sensor, name in (
            ("s3", "HPC outlet temperature"),
            ("s4", "LPT outlet temperature"),
            ("s9", "Core speed"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=cycles,
                    y=[reading["sensors"][sensor] for reading in hist],
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Failure probability")
        fig2 = go.Figure(
            go.Scatter(
                y=st.session_state.prob_hist,
                mode="lines",
                name="Failure probability",
                line=dict(color="#2563eb", width=2),
            )
        )
        fig2.add_hline(
            y=DECISION_THRESHOLD,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="Decision threshold",
            annotation_position="top left",
        )
        fig2.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_white",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("Assessment")
        risk = analysis.get("risk_level", "UNKNOWN")
        urgency = analysis.get("urgency", "UNKNOWN")
        confidence = float(analysis.get("confidence", 0.0))
        condition = _pretty(analysis.get("likely_condition", "unknown"))
        action = _pretty(analysis.get("recommended_action", "no action available"))
        evidence = analysis.get("evidence") or ["no specific evidence"]

        r1, r2 = st.columns(2)
        r1.metric("Risk", risk)
        r2.metric("Urgency", urgency)
        st.markdown(f"**Condition:** {condition}")
        st.markdown(f"**Recommended action:** {action}")
        st.caption(f"Confidence: {confidence:.0%}")
        st.markdown("**Evidence**")
        st.markdown(" ".join(f"`{_pretty(item)}`" for item in evidence))

    st.subheader("Ask MaintainAI")
    st.caption("Answers are grounded in the current validated assessment.")
    prompt_cols = st.columns(3)
    clicked_prompt = None
    for index, prompt in enumerate(QUICK_PROMPTS):
        with prompt_cols[index % 3]:
            if st.button(
                prompt,
                key=f"quick_prompt_{index}",
                use_container_width=True,
            ):
                clicked_prompt = prompt
    if clicked_prompt:
        st.session_state.chat_hist.append(("user", clicked_prompt))
        answer = chat(svc, machine, clicked_prompt, analysis)["text"]
        st.session_state.chat_hist.append(("assistant", answer))
    for role, text in st.session_state.chat_hist:
        st.chat_message(role).write(text)
    if question := st.chat_input("Ask about risk, evidence, RUL, or next steps"):
        st.session_state.chat_hist.append(("user", question))
        answer = chat(svc, machine, question, analysis)["text"]
        st.session_state.chat_hist.append(("assistant", answer))
        st.rerun()

    with st.expander("Model performance and technical details"):
        metrics_path = Path("models/predictive/metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            st.markdown("**Predictive model - held-out NASA FD001 test engines**")
            st.table(
                [
                    {"metric": "RUL RMSE", "value": round(metrics["test_rul"]["rmse"], 2)},
                    {"metric": "RUL MAE", "value": round(metrics["test_rul"]["mae"], 2)},
                    {"metric": "NASA score", "value": round(metrics["test_rul"]["nasa_score"], 1)},
                    {"metric": "Risk precision", "value": round(metrics["test_clf_30cycle"]["precision"], 3)},
                    {"metric": "Risk recall", "value": round(metrics["test_clf_30cycle"]["recall"], 3)},
                    {"metric": "Risk F1", "value": round(metrics["test_clf_30cycle"]["f1"], 3)},
                    {"metric": "Risk ROC-AUC", "value": round(metrics["test_clf_30cycle"]["roc_auc"], 3)},
                ]
            )
        else:
            st.info("Predictive metrics are not available in this deployment.")

        comparison_path = Path("reports/metrics.json")
        if comparison_path.exists():
            comparison = json.loads(comparison_path.read_text())
            rows = []
            for name, system in comparison["systems"].items():
                if system["status"] != "measured":
                    rows.append({"system": name, "status": "pending"})
                    continue
                values = system["metrics"]
                rows.append(
                    {
                        "system": name,
                        "risk_acc": round(values.get("risk_accuracy", 0), 3),
                        "condition_acc": round(values.get("condition_accuracy", 0), 3),
                        "evidence_F1": round(values.get("evidence_f1_mean", 0), 3),
                        "validity": round(values.get("validity_rate", 0), 3),
                    }
                )
            st.markdown("**Assistant - base vs fine-tuned on the same 100 scenarios**")
            st.table(rows)

        st.markdown("**Runtime**")
        st.json(
            {
                "predictive_model": pred["model_version"],
                "assistant_backend": state["backend"],
                "prediction_latency_ms": round(pred["latency_ms"], 1),
                "analysis_latency_ms": round(
                    analysis.get("meta", {}).get("latency_ms", 0.0), 1
                ),
                "decision_threshold": DECISION_THRESHOLD,
                "feature_window": FEATURE_WINDOW,
                "rul_cap": RUL_CAP,
                "data_timestamp": state["telemetry"]["timestamp"],
                "analysis_valid": analysis.get("meta", {}).get("valid"),
            }
        )

if auto and st.session_state.running:
    time.sleep(2)
    st.rerun()
