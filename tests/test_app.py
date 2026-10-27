"""Phase 10 test: the exact pipeline the dashboard runs, plus app smoke checks."""
import py_compile

from streamlit.testing.v1 import AppTest

from app.services_local import chat, create_services, machine_state


def test_dashboard_pipeline_end_to_end():
    svc = create_services("configs/test.yaml")
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


def test_quick_prompt_button_answers_in_chat(monkeypatch):
    monkeypatch.setenv("SLM_BACKEND", "demo")
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120)
    at.run()
    at.button[0].click().run()
    at.button(key="quick_prompt_0").click().run()
    assert not at.exception
    assert [message.name for message in at.chat_message] == ["user", "assistant"]
