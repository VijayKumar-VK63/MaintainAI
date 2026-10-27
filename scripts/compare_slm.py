"""Build the base-vs-fine-tuned comparison from measured reports.

Usage:
    python scripts/compare_slm.py

Reads (if present):
    reports/metrics_demo.json
    reports/metrics_base.json
    reports/metrics_finetuned.json
Writes:
    reports/metrics.json
    reports/evaluation_report.md
"""
from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")
SYSTEMS = {
    "demo-fallback": REPORTS / "metrics_demo.json",
    "qwen2.5-3b-base": REPORTS / "metrics_base.json",
    "qwen2.5-3b-lora": REPORTS / "metrics_finetuned.json",
}
METRIC_KEYS = [
    "risk_accuracy",
    "condition_accuracy",
    "evidence_f1_mean",
    "recommendation_accuracy",
    "validity_rate",
    "latency_mean_ms",
]


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    systems: dict[str, dict] = {}
    for name, path in SYSTEMS.items():
        metrics = load_metrics(path)
        if metrics is None:
            systems[name] = {
                "status": "pending",
                "metrics": None,
                "note": f"missing or invalid {path}",
            }
        elif metrics.get("n_errors", 0) >= metrics.get("n", 1):
            systems[name] = {
                "status": "invalid",
                "metrics": metrics,
                "note": "harness errors on all evaluation examples",
            }
        elif metrics.get("validity_rate", 0) == 0:
            systems[name] = {
                "status": "measured",
                "metrics": metrics,
                "note": "no schema-valid outputs",
            }
        else:
            systems[name] = {"status": "measured", "metrics": metrics}

    REPORTS.mkdir(parents=True, exist_ok=True)
    combined = {
        "n": next(
            (s["metrics"]["n"] for s in systems.values()
             if s["status"] == "measured"),
            None,
        ),
        "systems": systems,
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(combined, indent=1), encoding="utf-8"
    )

    n_label = combined["n"] if combined["n"] is not None else "held-out"
    lines = [
        "# SLM evaluation — base vs fine-tuned",
        "",
        f"Same {n_label} held-out NASA decision points; inference through `SLMService.analyze`.",
        "",
        "| System | Status | Risk acc | Condition acc | Evidence F1 | Recommendation acc | Validity | Latency ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in systems.items():
        m = entry.get("metrics") or {}
        row = [name, entry["status"]]
        for key in METRIC_KEYS:
            value = m.get(key)
            row.append("—" if value is None else (
                f"{value:.3f}" if isinstance(value, float) else str(value)
            ))
        lines.append("| " + " | ".join(row) + " |")
    zero_valid = [
        name for name, entry in systems.items()
        if entry["status"] == "measured"
        and (entry.get("metrics") or {}).get("validity_rate", 1) == 0
    ]
    if zero_valid:
        lines += [
            "",
            "> No schema-valid outputs: " + ", ".join(zero_valid)
            + ". Accuracy metrics count invalid outputs as incorrect.",
        ]
    backends = [
        (entry.get("metrics") or {}).get("backend", "")
        for entry in systems.values()
    ]
    if any("llamacpp:" in str(b) for b in backends):
        lines += [
            "",
            "> Local CPU inference via llama.cpp (Q4_K_M); latency is CPU-only "
            "and not comparable to GPU numbers. The commands below use "
            "`--constrained` (GBNF), so JSON shape/enums are guaranteed; any "
            "remaining invalid outputs are token-cap truncations or schema "
            "length-limit violations, not malformed JSON.",
        ]
    lines += [
        "",
        "> `pending` means unmeasured; `invalid` means the harness failed. "
        "Numbers are never zero-filled.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python -m src.slm_eval --backend demo --out reports/metrics_demo.json",
        "# CPU-only (llama.cpp, in-process)",
        "python -m src.slm_eval --backend llamacpp \\",
        "  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \\",
        "  --constrained --out reports/metrics_base.json",
        "python -m src.slm_eval --backend llamacpp \\",
        "  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \\",
        "  --adapter models/slm/maintainai-qwen2.5-3b-lora.gguf \\",
        "  --constrained --out reports/metrics_finetuned.json",
        "# GPU (transformers + PEFT)",
        "python -m src.slm_eval --backend hf --out reports/metrics_base.json",
        "python -m src.slm_eval --backend hf --adapter <adapter> --out reports/metrics_finetuned.json",
        "python scripts/compare_slm.py",
        "```",
        "",
    ]
    (REPORTS / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(combined, indent=1))


if __name__ == "__main__":
    main()
