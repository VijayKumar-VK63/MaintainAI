"""Seeded turbofan telemetry simulator, grounded in FD001 measured statistics.

NOT random numbers: each sensor has a baseline mean/std measured from early-life
(train, cycle<=30) FD001 cycles. NORMAL = baseline + noise. DEGRADATION /
CRITICAL / FAILURE apply progressive drift to the informative sensors (HPC
degradation pattern: s3/s4 temperatures, s9/s14 core speeds, s7/s11/s12
pressures), with scenario-dependent ramp rates. Deterministic per seed.

Near-real-time demo source only — NOT physical telemetry. Replaceable by a real
telemetry source behind the same SimulationService interface.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data_cmapss import SENSOR_COLS

# Measured from CMAPSSData/train_FD001.txt, cycle<=30 (early life), Phase 8.
BASELINE_MEAN = {
    "s1": 518.67, "s2": 642.33, "s3": 1585.99,     "s4": 1401.50,
    "s5": 14.62, "s6": 21.61, "s7": 553.00, "s8": 2388.10, "s9": 9045.00,
    "s10": 1.30, "s11": 47.35, "s12": 520.95, "s13": 2388.05, "s14": 8125.00,
    "s15": 8.40, "s16": 0.03, "s17": 392.00, "s18": 2388.00, "s19": 100.00,
    "s20": 38.70, "s21": 23.20,
}
BASELINE_STD = {
    "s1": 0.001, "s2": 0.30, "s3": 3.0, "s4": 4.5,
    "s5": 0.001, "s6": 0.001, "s7": 0.45, "s8": 0.05, "s9": 11.0,
    "s10": 0.001, "s11": 0.14, "s12": 0.38, "s13": 0.05, "s14": 9.5,
    "s15": 0.02, "s16": 0.001, "s17": 0.8, "s18": 0.001, "s19": 0.001,
    "s20": 0.09, "s21": 0.06,
}
# Per-cycle drift (in units of baseline std) for degrading sensors.
DRIFT_SENSORS = ["s2", "s3", "s4", "s7", "s9", "s11", "s12", "s14", "s15", "s21"]
SCENARIO_RAMP = {"NORMAL": 0.0, "DEGRADATION": 0.02, "CRITICAL": 0.08, "FAILURE": 0.20}
VALID_SCENARIOS = ("NORMAL", "DEGRADATION", "CRITICAL", "FAILURE")


class MachineSimulator:
    def __init__(self, machine_id: str = "M001", seed: int = 42, history: int = 200):
        self.machine_id = machine_id
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scenario = "NORMAL"
        self.cycle = 0
        self._drift = 0.0  # accumulated degradation level (std units)
        self.buffer: deque[dict] = deque(maxlen=history)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.scenario = "NORMAL"
        self.cycle = 0
        self._drift = 0.0
        self.buffer.clear()

    def set_scenario(self, scenario: str) -> None:
        if scenario not in VALID_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; valid: {VALID_SCENARIOS}")
        self.scenario = scenario

    def step(self) -> dict:
        self.cycle += 1
        self._drift += SCENARIO_RAMP[self.scenario]
        # Degradation compounds: drift accelerates the longer it runs.
        level = self._drift * (1.0 + self._drift / 100.0)
        sensors: dict[str, float] = {}
        for s in SENSOR_COLS:
            noise = float(self.rng.normal(0.0, BASELINE_STD[s]))
            shift = level * BASELINE_STD[s] if s in DRIFT_SENSORS else 0.0
            # s11 static pressure drops under HPC fault; rest rise.
            if s == "s11":
                shift = -shift
            sensors[s] = BASELINE_MEAN[s] + shift + noise
        reading = {
            "machine_id": self.machine_id, "cycle": self.cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
            "sensors": sensors,
        }
        self.buffer.append(reading)
        return reading

    def history_df(self) -> pd.DataFrame:
        rows = []
        for r in self.buffer:
            rows.append({"unit": 1, "cycle": r["cycle"], "op1": r["op_setting_1"],
                         "op2": r["op_setting_2"], "op3": r["op_setting_3"],
                         **r["sensors"]})
        return pd.DataFrame(rows, columns=["unit", "cycle", "op1", "op2", "op3"] + SENSOR_COLS)


class SimulationService:
    """Owns one simulator per machine; the API/UI talk only to this."""

    def __init__(self, machines: tuple[str, ...] = ("M001", "M002", "M003"), seed: int = 42):
        self.machines = {m: MachineSimulator(m, seed + i) for i, m in enumerate(machines)}
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def reset(self, seed: int = 42) -> None:
        for i, sim in enumerate(self.machines.values()):
            sim.reset(seed + i)
        self.running = False

    def set_scenario(self, machine_id: str, scenario: str) -> None:
        self.machines[machine_id].set_scenario(scenario)

    def tick(self) -> dict[str, dict]:
        if not self.running:
            raise RuntimeError("simulation not running; POST /simulation/start first")
        return {m: sim.step() for m, sim in self.machines.items()}
