"""MaintainAI dashboard — Machine Monitoring Demo (simulated telemetry).

Normal users see: machine status, key metrics, live charts, AI maintenance
analysis, chat. Engineers expand Technical details / Evaluation.
Simulated telemetry — research/demo system, not safety software.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.services_local import chat, create_services, machine_state

st.set_page_config(page_title="MaintainAI", layout="wide")
st.title("MaintainAI — Machine Monitoring Demo")
st.caption("Simulated telemetry for research/demo. Decision support only — "
           "qualified personnel make all maintenance decisions.")


@st.cache_resource
def _services():
    return create_services()


svc = _services()
for key, default in (("running", False), ("prob_hist", []), ("chat_hist", [])):
    st.session_state.setdefault(key, default)

STATUS_COLOR = {"HEALTHY": "green", "WARNING": "orange", "HIGH_RISK": "red", "CRITICAL": "red"}

with st.sidebar:
    st.header("Machine Simulation")
    machine = st.selectbox("Machine", sorted(svc["sim"].machines))
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
            st.warning("AI analysis running in demo mode — the fine-tuned SLM loads in the Colab/GPU "
                       "deployment. Monitoring and numerical prediction remain fully available.")

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
    st.caption("Base-vs-fine-tuned Qwen2.5-3B cells fill in after notebooks 06/08 run in Colab. "
               "Nothing here is fabricated: pending means unmeasured.")

if auto and st.session_state.running:
    time.sleep(2)
    st.rerun()
