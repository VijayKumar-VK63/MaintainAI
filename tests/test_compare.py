"""Phase 7 tests: comparison builder handles measured + pending honestly."""
import json

from scripts.compare_slm import build_comparison, render_markdown


def test_pending_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = build_comparison()
    assert all(s["status"] == "pending" for s in c["systems"].values())
    md = render_markdown(c)
    assert md.count("pending") >= 4
    assert "16.01" in md  # predictive numbers are committed constants


def test_measured_demo_flows_through():
    c = build_comparison()  # repo has reports/metrics_demo.json and metrics_base.json
    assert c["systems"]["demo"]["status"] == "measured"
    assert c["systems"]["demo"]["metrics"]["validity_rate"] == 1.0
    assert c["systems"]["base"]["status"] == "measured"  # base model eval completed (notebook 06)
    md = render_markdown(c)
    assert "1.000" in md and "measured" in md


def test_no_fabrication_keys():
    c = build_comparison()
    assert set(c["systems"]) == {"demo", "pipeline_check", "base", "finetuned"}
    assert json.loads(json.dumps(c))  # JSON-serializable
