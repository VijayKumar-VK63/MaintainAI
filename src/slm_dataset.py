"""SLM instruction-tuning dataset: grounded in real engine windows, NOT raw CSV dumps.

Each example: SYSTEM + rendered machine context (via ContextBuilder, the same
path used at inference) -> TARGET JSON validated by SLMAnalysis schema.
Targets are rule-derived from measured quantities (model outputs, causal trends,
true RUL stage); every rule is documented below. Synthetic edge cases are
labeled source='synthetic-edge' and kept separate.

Engine separation: SLM-train <- predictive train engines, SLM-val <- predictive
val engines, SLM-test <- NASA test engines. No engine crosses splits.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.context_builder import build_context, render_prompt
from src.data_cmapss import SENSOR_COLS, load_fd001
from src.predictive import MODEL_VERSION, health_state
from src.schemas import Prediction, SLMAnalysis, Telemetry

SYSTEM_PROMPT = (
    "You are a predictive maintenance assistant. "
    "Analyze machine condition ONLY from the provided evidence. "
    "Do not invent sensor measurements, maintenance history, inspections, "
    "or unsupported failure modes. Respond with structured JSON only."
)

WATCH = ("s3", "s4", "s9", "s11", "s12")
KNOWN_CONDITIONS = ["healthy_operation", "hpc_degradation", "unknown_anomaly"]

# Evidence vocabulary (closed set — generator may only emit these).
E_RUL_LOW = "rul_low"
E_RUL_HEALTHY = "rul_healthy_margin"
E_PROB_HIGH = "failure_probability_high"
E_PROB_LOW = "failure_probability_low"
E_BASELINE = "sensors_near_baseline"
E_MULTI_TREND = "trend_increasing_multiple_sensors"
E_UNKNOWN = "insufficient_evidence"


def sensor_evidence_token(sensor: str) -> str:
    return f"{sensor}_above_baseline"


def _above_baseline(row: pd.Series, s: str, k: float = 2.0) -> bool:
    mean = row.get(f"{s}_rollmean_30", float("nan"))
    std = row.get(f"{s}_rollstd_30", 0.0)
    if pd.isna(mean):
        return False
    floor = max(float(std), 1e-6)
    return bool(row[s] > mean + k * floor and row[s] > mean)


def derive_target(row: pd.Series, prob: float, rul_pred: float, threshold: float) -> dict:
    """Deterministic rules over measured quantities. Documented in SLM_DATASET.md."""
    evidence: list[str] = []
    drift_sensors = [s for s in WATCH if _above_baseline(row, s)]
    for s in drift_sensors:
        evidence.append(sensor_evidence_token(s))
    if len(drift_sensors) >= 2:
        evidence.append(E_MULTI_TREND)
    if rul_pred <= 30:
        evidence.append(E_RUL_LOW)
    elif rul_pred > 60:
        evidence.append(E_RUL_HEALTHY)
    if prob >= threshold:
        evidence.append(E_PROB_HIGH)
    elif prob < 0.30:
        evidence.append(E_PROB_LOW)
    if not drift_sensors and float(row.get("s3_trend", 0.0)) == 0.0 and prob < 0.30:
        evidence.append(E_BASELINE)

    hs = health_state(prob)
    risk = {"HEALTHY": "LOW", "WARNING": "MEDIUM", "HIGH_RISK": "HIGH", "CRITICAL": "CRITICAL"}[hs]
    hpc_pattern = len(drift_sensors) >= 2
    if hs == "HEALTHY" and not drift_sensors:
        condition = "healthy_operation"
    elif hpc_pattern:
        condition = "hpc_degradation"
    elif prob >= threshold:
        condition = "unknown_anomaly"  # high risk, no documented HPC pattern
    else:
        condition = "hpc_degradation" if drift_sensors else "healthy_operation"

    if not evidence:
        evidence = [E_UNKNOWN]
    if condition == "unknown_anomaly":
        confidence = 0.40
    elif (prob >= 0.85 or prob <= 0.15) and len(evidence) >= 2:
        confidence = 0.85
    elif len(evidence) >= 2:
        confidence = 0.70
    elif len(evidence) == 1 and evidence[0] not in (E_UNKNOWN, E_BASELINE):
        confidence = 0.55
    else:
        confidence = 0.35

    action, urgency = {
        "LOW": ("continue_monitoring", "ROUTINE"),
        "MEDIUM": ("schedule_hpc_inspection", "PRIORITY"),
        "HIGH": ("schedule_hpc_inspection", "PRIORITY"),
        "CRITICAL": ("escalate_immediate_review", "IMMEDIATE"),
    }[risk]
    target = {
        "risk_level": risk,
        "likely_condition": condition,
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": action,
        "urgency": urgency,
    }
    SLMAnalysis.model_validate(target)  # fail loudly on schema drift
    return target


def _telemetry_from_row(row: pd.Series, machine_id: str) -> Telemetry:
    return Telemetry(
        machine_id=machine_id,
        cycle=int(row["cycle"]),
        timestamp=datetime.now(timezone.utc),
        op_setting_1=float(row["op1"]),
        op_setting_2=float(row["op2"]),
        op_setting_3=float(row["op3"]),
        sensors={s: float(row[s]) for s in SENSOR_COLS},
    )


def build_examples(
    feat: pd.DataFrame,
    feature_columns: list[str],
    regressor,
    classifier,
    scaler,
    threshold: float,
    engines: list[int],
    machine_prefix: str,
    per_engine: int,
    seed: int,
    source: str,
    sample_mode: str = "stratified",  # or "last" (decision-point cycle per engine)
) -> list[dict]:
    rng = np.random.default_rng(seed)
    examples: list[dict] = []
    for u in engines:
        g = feat[feat["unit"] == u].sort_values("cycle").reset_index(drop=True)
        if len(g) < 5:
            continue
        if sample_mode == "last":
            picks = [len(g) - 1]
        else:
            # Stratify across life stages using TRUE rul column when present.
            stages = {"early": g[g["rul"] > 100], "mid": g[(g["rul"] <= 100) & (g["rul"] > 30)], "late": g[g["rul"] <= 30]} if "rul" in g else {"all": g}
            picks = []
            non_empty = [s for s in stages.values() if len(s)]
            for i in range(per_engine):
                pool = non_empty[i % len(non_empty)]
                picks.append(int(rng.choice(pool.index.to_numpy())))
        X = scaler.transform(g.loc[picks, feature_columns].to_numpy(dtype=np.float64))
        ruls = np.maximum(0.0, regressor.predict(X))
        probs = classifier.predict_proba(X)[:, 1]
        for idx, rul_p, prob in zip(picks, ruls, probs):
            row = g.loc[idx]
            prev = g.loc[idx - 1] if idx > 0 else None
            mid = f"{machine_prefix}{u:03d}"
            current = _telemetry_from_row(row, mid)
            previous = _telemetry_from_row(prev, mid) if prev is not None else None
            pred = Prediction(
                failure_probability=float(prob),
                anomaly_score=float(np.mean(np.abs(X[picks.index(idx)]))),
                health_state=health_state(float(prob)),  # type: ignore[arg-type]
                rul_cycles=int(round(float(rul_p))),
                model_version=MODEL_VERSION,
                latency_ms=0.0,
            )
            ctx = build_context(current, previous, pred, KNOWN_CONDITIONS)
            target = derive_target(row, float(prob), float(rul_p), threshold)
            examples.append({
                "system": SYSTEM_PROMPT,
                "user": render_prompt(ctx),
                "target": json.dumps(target),
                "meta": {
                    "engine": mid, "cycle": int(row["cycle"]),
                    "true_rul": float(row["rul"]) if "rul" in g else None,
                    "source": source, "model_version": MODEL_VERSION,
                },
            })
    return examples


def generate_dataset(
    raw_dir: str = "CMAPSSData",
    artifact_dir: str = "models/predictive",
    out_dir: str = "data/slm",
    per_engine_train: int = 8,
    per_engine_val: int = 7,
    per_engine_test: int = 1,
    seed: int = 42,
) -> dict:
    from src.features import build_features

    art = Path(artifact_dir)
    schema = json.loads((art / "feature_schema.json").read_text())
    meta = json.loads((art / "metadata.json").read_text())
    cols, tr_units = schema["feature_columns"], schema["train_units"]
    va_units, threshold = schema["val_units"], meta["decision_threshold"]
    scaler = joblib.load(art / "scaler_fd001.joblib")
    regressor = joblib.load(art / "rul_model.joblib")
    classifier = joblib.load(art / "risk_model.joblib")

    train_feat = pd.read_parquet("data/processed/train_features.parquet")
    _, te, true_rul = load_fd001(raw_dir)
    te_feat = build_features(te)
    te_feat["rul"] = te_feat.groupby("unit")["cycle"].transform(
        lambda c: (c.max() + true_rul.iloc[int(c.name) - 1]) - c
    )
    test_units = sorted(int(u) for u in te["unit"].unique())

    # Engine separation guard (train/test files share the 1-100 ID namespace).
    assert set(f"tr{u}" for u in tr_units).isdisjoint(f"te{u}" for u in test_units)

    splits = {
        "train": build_examples(train_feat, cols, regressor, classifier, scaler, threshold, tr_units, "M", per_engine_train, seed, "derived-train-engines"),
        "val": build_examples(train_feat, cols, regressor, classifier, scaler, threshold, va_units, "M", per_engine_val, seed + 1, "derived-val-engines"),
        "test": build_examples(te_feat, cols, regressor, classifier, scaler, threshold, test_units, "T", per_engine_test, seed + 2, "derived-test-engines", sample_mode="last"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, exs in splits.items():
        # Schema-validate every target before writing.
        for e in exs:
            SLMAnalysis.model_validate_json(e["target"])
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e) + "\n")
        summary[name] = len(exs)
    (out / "generation_config.json").write_text(json.dumps({
        "seed": seed, "per_engine": {"train": per_engine_train, "val": per_engine_val, "test": per_engine_test},
        "model_version": MODEL_VERSION, "threshold": threshold, "rules": "src/slm_dataset.py::derive_target",
    }, indent=1))
    return summary
