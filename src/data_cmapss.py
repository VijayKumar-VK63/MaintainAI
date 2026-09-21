"""CMAPSS FD001 loader: raw (immutable) -> processed (RUL, capped RUL, splits).

Leakage policy: split by ENGINE (unit), never by row. NASA test set + RUL file
are touched only for final evaluation (Phase 3), never for fitting.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
# ~constant in FD001 (verified in Phase 2 analysis); excluded from features.
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
INFORMATIVE_SENSORS = [s for s in SENSOR_COLS if s not in CONSTANT_SENSORS]


def load_fd001(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = Path(raw_dir)
    train = pd.read_csv(raw / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(raw / "test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(raw / "RUL_FD001.txt", header=None, names=["true_rul"])[ "true_rul"]
    return train, test, rul


def add_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Train trajectories run to failure: RUL = max_cycle(unit) - cycle."""
    df = train.copy()
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def add_capped_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    df = df.copy()
    df["rul_capped"] = df["rul"].clip(upper=cap)
    return df


def engine_split(units: list[int], val_fraction: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
    """Deterministic engine-level split. No engine appears on both sides."""
    import random

    rng = random.Random(seed)
    shuffled = sorted(units)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def assert_disjoint(*groups: list[int]) -> None:
    sets = [set(g) for g in groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j]), f"engine leakage between groups {i} and {j}"
