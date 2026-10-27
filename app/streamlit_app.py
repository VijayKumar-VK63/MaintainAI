"""MaintainAI dashboard — maintenance assistant over simulated telemetry.

Normal users see: machine status, key metrics, live charts, AI maintenance
analysis, chat. Engineers expand Technical details / Evaluation.
Simulated telemetry — research/demo system, not safety software.
"""
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

# Ensure project root is in Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services_local import chat, create_services, machine_state  # noqa: E402

st.set_page_config(page_title="MaintainAI", layout="wide")

# Streamlit Cloud exposes secrets via st.secrets, not as environment variables.
# Bridge the SLM/API settings so src.config sees the same values locally/cloud.
try:
    for _key in ("SLM_BACKEND", "SLM_INFERENCE_ENDPOINT", "SLM_INFERENCE_PATH",
                 "SLM_API_KEY", "HF_TOKEN", "SLM_MODEL_NAME", "SLM_MODEL_PATH",
                 "SLM_ADAPTER_PATH", "SLM_LORA_SCALE", "SLM_CONSTRAINED",
                 "SLM_N_CTX", "SLM_N_THREADS", "SLM_N_BATCH",
                 "SLM_MAX_NEW_TOKENS", "SLM_TEMPERATURE", "SLM_QUANTIZATION"):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass
st.title("MaintainAI - Maintenance Assistant")
st.caption("Simulated telemetry for research/demo. Decision support only - "
           "qualified personnel make all maintenance decisions.")


@st.cache_resource
def _services():
    return create_services()


try:
    svc = _services()
except Exception as exc:
    st.error("The maintenance SLM could not start. This app does not fall back "
             "to the demo rule mirror.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info("Local: run `python scripts/prepare_local_slm.py` and set "
            "`SLM_MODEL_PATH` to the GGUF. Cloud: set `SLM_BACKEND=openai` and "
            "`SLM_INFERENCE_ENDPOINT` in Streamlit secrets.")
    st.stop()
for key, default in (("running", False), ("prob_hist", []), ("chat_hist", [])):
    st.session_state.setdefault(key, default)

STATUS_COLOR = {"HEALTHY": "green", "WARNING": "orange", "HIGH_RISK": "red", "CRITICAL": "red"}

with st.sidebar:
    st.header("Machine Simulation")
    st.caption(f"SLM backend: {svc['slm'].version}")
    if svc["slm"].version.startswith("demo"):
        st.error("Demo rule mirror active. Set SLM_MODEL_PATH (local GGUF) or "
                 "SLM_INFERENCE_ENDPOINT (remote) to use the fine-tuned SLM.")
    machine = st.selectbox("Machine", sorted(svc["sim"].machines))
    if st.session_state.get("active_machine") != machine:
        st.session_state.active_machine = machine
        st.session_state.prob_hist = []
        st.session_state.chat_hist = []
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring"):
            svc["sim"].start()
            st.session_state.running = True
    with col2:
        if st.button("Pause"):
            svc["sim"].stop()
            st.session_state.running = False
    if st.button("Reset"):
        svc["sim"].reset()
        st.session_state.update(running=False, prob_hist=[], chat_hist=[])
    scenario = st.selectbox("Scenario", ["NORMAL", "DEGRADATION", "CRITICAL", "FAILURE"])
    if st.button("Apply scenario"):
        svc["sim"].set_scenario(machine, scenario)
        st.success(f"{machine} -> {scenario}")
    speed = st.slider("Cycles per refresh", 1, 10, 3)
    auto = st.checkbox("Auto-refresh (near-real-time demo)", value=False)

tab_monitor, tab_eval = st.tabs(["Monitor", "Evaluation"])

with tab_monitor:
    if st.session_state.running:
        try:
            for _ in range(speed):
                svc["sim"].tick()
        except RuntimeError as exc:
            st.warning(str(exc))
            st.session_state.running = False

    try:
        state = machine_state(svc, machine) if svc["sim"].machines[machine].buffer else None
    except RuntimeError as exc:
        state = None
        st.warning(f"{exc}")

    if state is None:
        st.info("Press **Start Monitoring** to begin producing telemetry.")
    else:
        pred, analysis = state["prediction"], state["analysis"]
        st.session_state.prob_hist.append(pred["failure_probability"])
        st.subheader(f"{machine} — :{STATUS_COLOR.get(pred['health_state'], 'gray')}[{pred['health_state']}]")
        if analysis.get("meta", {}).get("demo"):
            st.error("Demo rule mirror is active - this is not the fine-tuned SLM. "
                     "Configure SLM_MODEL_PATH or SLM_INFERENCE_ENDPOINT.")
        elif not analysis.get("meta", {}).get("valid", True):
            st.warning("The SLM returned an invalid/empty response; showing a controlled UNKNOWN. "
                       f"Reason: {analysis.get('meta', {}).get('reason', 'unknown')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Failure probability", f"{pred['failure_probability']:.0%}")
        m2.metric("Predicted RUL (cycles)", pred["rul_cycles"])
        m3.metric("Anomaly score", f"{pred['anomaly_score']:.2f}")
        m4.metric("Health state", pred["health_state"])

        hist = state["history"]
        cycles = [r["cycle"] for r in hist]
        fig = go.Figure()
        for sensor, name in (("s3", "HPC outlet temp"), ("s4", "LPT outlet temp"), ("s9", "Core speed")):
            fig.add_trace(go.Scatter(x=cycles, y=[r["sensors"][sensor] for r in hist],
                                     mode="lines", name=name))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), title="Live telemetry")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Scatter(y=st.session_state.prob_hist, mode="lines", name="failure prob"))
        fig2.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title="Failure probability trend")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("AI maintenance analysis")
        a1, a2, a3 = st.columns(3)
        a1.metric("Risk level", analysis["risk_level"])
        a2.metric("Likely condition", analysis["likely_condition"])
        a3.metric("Urgency", analysis["urgency"])
        st.write(f"**Recommended action:** {analysis['recommended_action']} "
                 f"(confidence {analysis['confidence']:.0%})")
        st.write("**Evidence:** " + ", ".join(analysis["evidence"]))

        st.subheader("Maintenance assistant")
        for role, text in st.session_state.chat_hist:
            st.chat_message(role).write(text)
        if question := st.chat_input("Why is this machine at high risk?"):
            st.session_state.chat_hist.append(("user", question))
            answer = chat(svc, machine, question)["text"]
            st.session_state.chat_hist.append(("assistant", answer))
            st.rerun()

        with st.expander("Technical details (engineers)"):
            st.json({"predictive_model": pred["model_version"],
                     "slm_backend": state["backend"],
                     "prediction_latency_ms": round(pred["latency_ms"], 1),
                     "analysis_latency_ms": round(analysis.get("meta", {}).get("latency_ms", 0.0), 1),
                     "decision_threshold": 0.65, "feature_window": 30,
                     "data_timestamp": state["telemetry"]["timestamp"],
                     "analysis_valid": analysis.get("meta", {}).get("valid")})

with tab_eval:
    st.header("Evaluation (measured values only)")
    pred_m = json.loads(Path("models/predictive/metrics.json").read_text())
    st.subheader("Predictive model — held-out NASA test (100 engines)")
    st.table([{"metric": "RUL RMSE", "value": round(pred_m["test_rul"]["rmse"], 2)},
              {"metric": "RUL MAE", "value": round(pred_m["test_rul"]["mae"], 2)},
              {"metric": "NASA score", "value": round(pred_m["test_rul"]["nasa_score"], 1)},
              {"metric": "Risk precision", "value": round(pred_m["test_clf_30cycle"]["precision"], 3)},
              {"metric": "Risk recall", "value": round(pred_m["test_clf_30cycle"]["recall"], 3)},
              {"metric": "Risk F1", "value": round(pred_m["test_clf_30cycle"]["f1"], 3)},
              {"metric": "Risk ROC-AUC", "value": round(pred_m["test_clf_30cycle"]["roc_auc"], 3)}])
    st.subheader("SLM — base vs fine-tuned (same 100 held-out scenarios)")
    cmp_path = Path("reports/metrics.json")
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text())
        rows = []
        for key, s in cmp["systems"].items():
            if s["status"] != "measured":
                rows.append({"system": key, "status": "pending (unmeasured — see notes)"})
                continue
            m = s["metrics"]
            rows.append({"system": key,
                         "risk_acc": round(m.get("risk_accuracy", 0), 3),
                         "cond_acc": round(m.get("condition_accuracy", 0), 3),
                         "evidence_F1": round(m.get("evidence_f1_mean", 0), 3),
                         "validity": round(m.get("validity_rate", 0), 3)})
        st.table(rows)
    st.caption("Measured locally on CPU with llama.cpp (100 held-out FD001 examples); "
               "see reports/evaluation_report.md. Nothing here is fabricated.")

if auto and st.session_state.running:
    time.sleep(2)
    st.rerun()
