"""Phase 7 comparison builder: base vs fine-tuned SLM (plus local references).

Reads whichever of these exist (never invents missing ones):
  reports/metrics_demo.json            — local rule-mirror baseline (measured)
  reports/metrics_pipeline_check.json  — 135M HF-path validation (measured)
  reports/metrics_base.json            — Qwen2.5-3B base, Colab nb 06 (measured)
  reports/metrics_finetuned.json       — Qwen2.5-3B+LoRA, Colab nb 08 (measured)
Writes:
  reports/metrics.json           — combined machine-readable comparison
  reports/evaluation_report.md   — comparison table (pending cells stay pending)
"""
from __future__ import annotations

import json
from pathlib import Path

ROWS = [
    ("risk_accuracy", "Risk accuracy"),
    ("condition_accuracy", "Condition accuracy"),
    ("evidence_f1_mean", "Evidence F1"),
    ("recommendation_accuracy", "Recommendation accuracy"),
    ("validity_rate", "Validity rate"),
    ("latency_mean_ms", "Latency mean (ms)"),
]
SOURCES = [
    ("demo", "reports/metrics_demo.json", "demo-fallback rule-mirror (local, measured)"),
    ("pipeline_check", "reports/metrics_pipeline_check.json", "135M HF path validation (local, measured)"),
    ("base", "reports/metrics_base.json", "Qwen2.5-3B base (Colab nb 06, measured)"),
    ("finetuned", "reports/metrics_finetuned.json", "Qwen2.5-3B+LoRA (Colab nb 08, measured)"),
]


def _load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def build_comparison() -> dict:
    comparison = {"systems": {}, "notes": []}
    for key, path, label in SOURCES:
        m = _load(path)
        if m is None:
            comparison["systems"][key] = {"status": "pending", "label": label}
            comparison["notes"].append(f"{key}: unmeasured — run the corresponding notebook/script first")
        else:
            comparison["systems"][key] = {"status": "measured", "label": label, "metrics": m}
    return comparison


def render_markdown(comparison: dict) -> str:
    lines = ["# Evaluation report (actual measured values — pending cells stay pending)", "",
             "## Predictive model — held-out NASA FD001 test (100 engines)",
             "- RUL: RMSE 16.01, MAE 11.52, NASA score 416.7",
             "- Risk (failure ≤ 30 cycles): precision 0.958, recall 0.920, F1 0.939, ROC-AUC 0.994",
             "- Source: `models/predictive/metrics.json`", "",
             "## SLM — 100 held-out decision-point scenarios",
             "| System | risk_acc | cond_acc | evidence_F1 | rec_acc | validity | latency |",
             "|---|---|---|---|---|---|---|"]
    for key, _, _ in SOURCES:
        s = comparison["systems"][key]
        if s["status"] != "measured":
            lines.append(f"| {key} | pending | pending | pending | pending | pending | pending |")
            continue
        m = s["metrics"]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(" | ".join(["", key, fmt(m.get("risk_accuracy")), fmt(m.get("condition_accuracy")),
                                 fmt(m.get("evidence_f1_mean")), fmt(m.get("recommendation_accuracy")),
                                 fmt(m.get("validity_rate")), fmt(m.get("latency_mean_ms")), ""]))
    lines += ["", "## Notes"] + [f"- {n}" for n in comparison["notes"]]
    return "\n".join(lines) + "\n"


def main() -> dict:
    comparison = build_comparison()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(comparison, indent=1), encoding="utf-8")
    Path("reports/evaluation_report.md").write_text(render_markdown(comparison), encoding="utf-8")
    measured = [k for k, s in comparison["systems"].items() if s["status"] == "measured"]
    print("measured:", measured)
    return comparison


if __name__ == "__main__":
    main()
