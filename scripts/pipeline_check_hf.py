"""Pipeline check: notebook-06 code path with a REAL HF model on CPU.

Loads SmolLM2-135M via HFBackend (same class notebook 06 uses for Qwen),
scores N held-out scenarios through run_harness, saves to
reports/metrics_pipeline_check.json.

This is INFRASTRUCTURE validation, not the Qwen experiment: a 135M base model
with no task training is expected to score ~0 validity. It proves load ->
generate -> extract -> validate -> score works before Colab GPU time is spent.
"""
from __future__ import annotations

import json

from src.schemas import SLMAnalysis
from src.slm_eval import run_harness
from src.slm_service import SLMService, extract_json

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main(n: int = 5, out: str = "reports/metrics_pipeline_check.json") -> dict:
    from src.slm_service import HFBackend

    svc = SLMService(config={"slm": {"model_name": MODEL}},
                     backend=HFBackend(MODEL, quantization="none",
                                       max_new_tokens=128, temperature=0.0))
    test = [json.loads(line) for line in open("data/slm/test.jsonl", encoding="utf-8")][:n]

    def analyze_fn(e: dict) -> dict:
        raw = svc.backend.generate(e["system"] + "\n" + e["user"])  # type: ignore[union-attr]
        obj = extract_json(raw)
        try:
            SLMAnalysis.model_validate(obj or {})
            assert isinstance(obj, dict)
            return {**obj, "meta": {"backend": svc.version, "valid": True}}
        except Exception as exc:  # noqa: BLE001
            return {"meta": {"backend": svc.version, "valid": False, "reason": str(exc)[:200]}}

    m = run_harness(test, analyze_fn)
    m["backend"] = svc.version
    m["note"] = "pipeline validation on 135M base; NOT the Qwen2.5-3B baseline"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))
    return m


if __name__ == "__main__":
    main()
