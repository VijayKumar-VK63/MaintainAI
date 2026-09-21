"""Phase 10 test: the exact pipeline the dashboard runs, plus app smoke checks."""
import py_compile

from app.services_local import chat, create_services, machine_state


def test_dashboard_pipeline_end_to_end():
    svc = create_services()
    svc["sim"].start()
    svc["sim"].set_scenario("M001", "DEGRADATION")
    for _ in range(30):
        svc["sim"].tick()
    state = machine_state(svc, "M001")
    assert state["prediction"]["failure_probability"] >= 0.0
    assert state["analysis"]["meta"]["valid"] is True
    ans = chat(svc, "M001", "What should the maintenance team inspect?")
    assert "M001" in ans["text"]


def test_streamlit_app_compiles():
    py_compile.compile("app/streamlit_app.py", doraise=True)
