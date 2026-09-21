"""Predictive maintenance models for CMAPSS FD001.

Two heads, both trained on train engines, selected on val engines, evaluated
once on untouched NASA test trajectories:
  - RUL regressor (target: capped RUL, cap=125) — Ridge baseline vs XGBoost.
  - Risk classifier (target: failure within 30 cycles, i.e. rul <= 30).

PredictionService.predict(machine_history) is the ONLY entry point the UI/API
may call. Anomaly score = mean |scaled feature| (measured, >= 0).
Health bands on failure probability are fixed and documented, NOT tuned on test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score

from src.features import apply_scaler, build_features

MODEL_VERSION = "rul-xgb-clf-v1"
RUL_CAP = 125
HORIZON = 30  # maintenance-horizon cycles for the risk label


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


def classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Best-F1 threshold on VALIDATION engines only. Never call on test."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        f1 = f1_score(y_true, (np.asarray(y_prob) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def train_ridge(X: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    return model


def train_xgb_regressor(X: np.ndarray, y: np.ndarray):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_xgb_classifier(X: np.ndarray, y: np.ndarray):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def health_state(failure_probability: float) -> str:
    if failure_probability < 0.30:
        return "HEALTHY"
    if failure_probability < 0.60:
        return "WARNING"
    if failure_probability < 0.85:
        return "HIGH_RISK"
    return "CRITICAL"


class PredictionService:
    """Loads saved artifacts; predicts from a raw-cycle history DataFrame."""

    def __init__(self, artifact_dir: str | Path = "models/predictive"):
        d = Path(artifact_dir)
        self.scaler = joblib.load(d / "scaler_fd001.joblib")
        self.regressor = joblib.load(d / "rul_model.joblib")
        self.classifier = joblib.load(d / "risk_model.joblib")
        schema = json.loads((d / "feature_schema.json").read_text())
        self.feature_columns: list[str] = schema["feature_columns"]
        meta = json.loads((d / "metadata.json").read_text())
        self.model_version: str = meta["model_version"]
        self.threshold: float = meta["decision_threshold"]

    def predict(self, machine_history: pd.DataFrame) -> dict:
        """machine_history: raw cycles (unit/cycle/sensors) for ONE engine, causal order."""
        t0 = time.perf_counter()
        feat = build_features(machine_history)
        row = feat.iloc[[-1]]
        X = self.scaler.transform(row[self.feature_columns].to_numpy(dtype=np.float64))
        rul = float(max(0.0, self.regressor.predict(X)[0]))
        prob = float(self.classifier.predict_proba(X)[0, 1])
        return {
            "failure_probability": prob,
            "anomaly_score": float(np.mean(np.abs(X))),
            "health_state": health_state(prob),
            "rul_cycles": int(round(rul)),
            "model_version": self.model_version,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
