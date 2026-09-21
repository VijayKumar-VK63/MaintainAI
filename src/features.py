"""Causal feature engineering: all rolling stats use .shift(1) — no future peeking.

Per engine (unit), per informative sensor:
  {s}_rollmean_{w}, {s}_rollstd_{w}, {s}_trend (current - rollmean),
  {s}_delta_base (current - causal expanding mean baseline).
Raw columns are preserved; scaler is fit on train engines ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_cmapss import INFORMATIVE_SENSORS


def build_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    out = df.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)
    for s in INFORMATIVE_SENSORS:
        g = out.groupby("unit")[s].shift(1)  # causal: exclude current cycle
        roll = g.groupby(out["unit"])
        out[f"{s}_rollmean_{window}"] = roll.transform(lambda x: x.rolling(window, min_periods=1).mean())
        out[f"{s}_rollstd_{window}"] = (
            roll.transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0.0)
        )
        out[f"{s}_trend"] = out[s] - out[f"{s}_rollmean_{window}"].fillna(out[s])
        baseline = g.groupby(out["unit"]).transform(lambda x: x.expanding(min_periods=1).mean())
        out[f"{s}_delta_base"] = out[s] - baseline.fillna(out[s])
    # First cycle per engine has no history: backfill rollmean with current
    # value (trend 0 by construction) so no NaNs propagate to scaler/model.
    for s in INFORMATIVE_SENSORS:
        out[f"{s}_rollmean_{window}"] = out[f"{s}_rollmean_{window}"].fillna(out[s])
    trend_cols = [c for c in out.columns if c.endswith("_trend") or c.endswith("_delta_base")]
    out[trend_cols] = out[trend_cols].fillna(0.0)
    return out


def feature_columns(window: int = 30) -> list[str]:
    cols: list[str] = []
    for s in INFORMATIVE_SENSORS:
        cols += [f"{s}_rollmean_{window}", f"{s}_rollstd_{window}", f"{s}_trend", f"{s}_delta_base"]
    return cols


def fit_scaler(train: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train[columns].to_numpy(dtype=np.float64))
    return scaler


def apply_scaler(df: pd.DataFrame, columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[columns] = scaler.transform(df[columns].to_numpy(dtype=np.float64))
    return out
