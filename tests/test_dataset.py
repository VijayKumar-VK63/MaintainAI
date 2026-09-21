"""Phase 2 tests: RUL correctness, engine-level splits, causal features, scaler."""
import numpy as np
import pandas as pd

from src.data_cmapss import (
    CONSTANT_SENSORS,
    INFORMATIVE_SENSORS,
    add_capped_rul,
    add_rul,
    assert_disjoint,
    engine_split,
    load_fd001,
)
from src.features import apply_scaler, build_features, feature_columns, fit_scaler

RAW = "CMAPSSData"


def test_rul_last_cycle_zero():
    train, _, _ = load_fd001(RAW)
    df = add_rul(train)
    last = df.groupby("unit")["rul"].min()
    assert (last == 0).all()
    first = df.groupby("unit").first()
    assert ((first["cycle"] + first["rul"]) == df.groupby("unit")["cycle"].max()).all()


def test_rul_cap():
    train, _, _ = load_fd001(RAW)
    df = add_capped_rul(add_rul(train), cap=125)
    assert df["rul_capped"].max() <= 125
    assert (df.loc[df["rul"] > 125, "rul_capped"] == 125).all()


def test_engine_split_disjoint_and_deterministic():
    units = list(range(1, 101))
    tr1, va1 = engine_split(units, 0.2, 42)
    tr2, va2 = engine_split(units, 0.2, 42)
    assert tr1 == tr2 and va1 == va2
    assert len(va1) == 20 and len(tr1) == 80
    assert_disjoint(tr1, va1)


def test_constant_sensors_excluded():
    cols = feature_columns()
    for s in CONSTANT_SENSORS:
        assert not any(c.startswith(s + "_") for c in cols)
    assert len(INFORMATIVE_SENSORS) == 14
    assert len(cols) == 14 * 4


def test_features_causal_no_future_leak():
    # Rising sensor: causal rollmean at each row must only reflect strictly earlier rows.
    df = pd.DataFrame(
        {
            "unit": [1] * 5,
            "cycle": [1, 2, 3, 4, 5],
            **{s: ([10.0, 20.0, 30.0, 40.0, 50.0] if s == "s3" else [1.0] * 5) for s in INFORMATIVE_SENSORS},
        }
    )
    feat = build_features(df, window=30)
    # Row 3 (cycle 3): history = cycles 1-2 -> mean(10,20)=15
    assert feat.loc[2, "s3_rollmean_30"] == 15.0
    # First cycle: no history -> trend/delta zero
    assert feat.loc[0, "s3_trend"] == 0.0
    assert feat.loc[0, "s3_delta_base"] == 0.0


def test_scaler_fit_train_only():
    train, _, _ = load_fd001(RAW)
    df = add_capped_rul(add_rul(train), cap=125)
    cols = feature_columns()
    feat = build_features(df)
    tr_units, va_units = engine_split(sorted(df["unit"].unique()), 0.2, 42)
    tr = feat[feat["unit"].isin(tr_units)]
    scaler = fit_scaler(tr, cols)
    scaled = apply_scaler(tr, cols, scaler)
    assert abs(scaled[cols].to_numpy().mean()) < 0.05  # ~zero-mean on train
    assert np.isfinite(scaled[cols].to_numpy()).all()
