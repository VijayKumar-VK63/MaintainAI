"""Held-out SLM evaluation harness (same scenarios for base vs fine-tuned).

Metrics over data/slm/test.jsonl (100 decision-point scenarios):
  - risk_accuracy: exact risk_level match
  - condition_accuracy: exact likely_condition match
  - evidence_f1: mean token-set F1 vs reference
  - recommendation_accuracy: exact recommended_action match
  - validity_rate: schema-valid outputs / total
  - uncertainty_skill: on reference unknown_anomaly cases, fraction with
    confidence <= 0.5 (admits uncertainty); on the rest, fraction > 0.5
  - latency_mean_ms

Usage (local, mock/demo backend):
    python -m src.slm_eval --backend demo --out reports/metrics_demo.json
Usage (Colab, real model — notebook 06/08):
    backend = HFBackend(...); run_harness(examples, service.analyze, ...)
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path


def _evidence_f1(ref: list[str], hyp: list[str]) -> float:
    r, h = set(ref), set(hyp)
    if not r and not h:
        return 1.0
    if not r or not h:
        return 0.0
    inter = len(r & h)
    prec, rec = inter / len(h), inter / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def run_harness(examples: list[dict], analyze_fn: Callable[[dict], dict]) -> dict:
    n = len(examples)
    risk_ok = cond_ok = rec_ok = valid = 0
    f1s: list[float] = []
    unk_total = unk_humble = known_total = known_conf = 0
    lat: list[float] = []
    failures: list[dict] = []
    for i, e in enumerate(examples):
        ref = json.loads(e["target"])
        t0 = time.perf_counter()
        try:
            out = analyze_fn(e)
        except Exception as exc:  # noqa: BLE001 — harness records, never stops
            failures.append({"i": i, "error": type(exc).__name__})
            continue
        lat.append((time.perf_counter() - t0) * 1000.0)
        meta = out.get("meta", {})
        if not meta.get("valid", False):
            failures.append({"i": i, "error": meta.get("reason", "invalid")})
            continue
        valid += 1
        risk_ok += out.get("risk_level") == ref["risk_level"]
        cond_ok += out.get("likely_condition") == ref["likely_condition"]
        rec_ok += out.get("recommended_action") == ref["recommended_action"]
        f1s.append(_evidence_f1(ref["evidence"], out.get("evidence", [])))
        conf = float(out.get("confidence", 0.0))
        if ref["likely_condition"] == "unknown_anomaly":
            unk_total += 1
            unk_humble += conf <= 0.5
        else:
            known_total += 1
            known_conf += conf > 0.5
    return {
        "n": n,
        "risk_accuracy": risk_ok / n,
        "condition_accuracy": cond_ok / n,
        "evidence_f1_mean": sum(f1s) / len(f1s) if f1s else 0.0,
        "recommendation_accuracy": rec_ok / n,
        "validity_rate": valid / n if n else 0.0,
        "uncertainty_skill": {
            "unknown_admits_uncertainty": (unk_humble / unk_total) if unk_total else None,
            "known_confident": (known_conf / known_total) if known_total else None,
            "n_unknown": unk_total,
        },
        "latency_mean_ms": sum(lat) / len(lat) if lat else 0.0,
        "n_failures": len(failures),
    }


def load_test_examples(path: str = "data/slm/test.jsonl") -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def example_to_context(e: dict):
    """Rebuild the MachineContext inputs from a stored example's user prompt.

    The stored user text already IS the rendered context; for HF backends we
    pass it straight through. Returns (system, user, reference).
    """
    return e["system"], e["user"], json.loads(e["target"])


def build_live_contexts(n_engines: int | None = None):
    """Rebuild live MachineContexts for NASA test engines (decision-point cycle).

    Full pipeline: raw history -> PredictionService -> ContextBuilder.
    Returns (contexts, references) aligned by engine for scoring.
    """
    from src.context_builder import build_context
    from src.data_cmapss import SENSOR_COLS, load_fd001
    from src.predictive import PredictionService
    from src.schemas import MachineContext, Prediction, Telemetry
    from src.slm_dataset import KNOWN_CONDITIONS

    _, te, _ = load_fd001("CMAPSSData")
    refs = {json.loads(l)["meta"]["engine"]: json.loads(l)["target"]
            for l in open("data/slm/test.jsonl", encoding="utf-8")}
    pred_svc = PredictionService("models/predictive")
    raw_cols = ["unit", "cycle", "op1", "op2", "op3"] + SENSOR_COLS
    contexts: list[MachineContext] = []
    references: list[dict] = []
    for u in sorted(te["unit"].unique()):
        if n_engines is not None and len(contexts) >= n_engines:
            break
        g = te[te["unit"] == u].sort_values("cycle")
        r = pred_svc.predict(g[raw_cols])
        last, prev = g.iloc[-1], g.iloc[-2]
        def tel(row) -> Telemetry:
            return Telemetry(
                machine_id=f"T{int(u):03d}", cycle=int(row["cycle"]),
                op_setting_1=float(row["op1"]), op_setting_2=float(row["op2"]),
                op_setting_3=float(row["op3"]),
                sensors={s: float(row[s]) for s in SENSOR_COLS})
        pred = Prediction(failure_probability=r["failure_probability"],
                          anomaly_score=r["anomaly_score"], health_state=r["health_state"],  # type: ignore[arg-type]
                          rul_cycles=r["rul_cycles"], model_version=r["model_version"], latency_ms=0.0)
        contexts.append(build_context(tel(last), tel(prev), pred, KNOWN_CONDITIONS))
        references.append(json.loads(refs[f"T{int(u):03d}"]))
    return contexts, references


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["demo", "hf"], default="demo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="reports/metrics_demo.json")
    args = ap.parse_args()

    from src.slm_service import HFBackend, SLMService

    if args.backend == "hf":
        svc = SLMService(config={"slm": {"model_name": args.model}},
                         backend=HFBackend(args.model, adapter_path=args.adapter))
    else:
        svc = SLMService(config={"slm": {"model_name": "demo-fallback"}})

    contexts, references = build_live_contexts()
    examples = [{"target": json.dumps(ref)} for ref in references]
    it = iter(contexts)

    def analyze_fn(e: dict) -> dict:
        return svc.analyze(next(it))

    metrics = run_harness(examples, analyze_fn)
    metrics["backend"] = svc.version
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
