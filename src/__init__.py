"""MaintainAI: End-to-end predictive maintenance with fine-tuned SLM."""
from __future__ import annotations

__version__ = "0.1.0"

from src.config import load_config
from src.context_builder import build_context, render_prompt
from src.data_cmapss import (
    SENSOR_COLS,
    INFORMATIVE_SENSORS,
    load_fd001,
    add_rul,
    add_capped_rul,
    engine_split,
    assert_disjoint,
)
from src.features import build_features, feature_columns, fit_scaler, apply_scaler
from src.logging_utils import get_logger, log_event
from src.predictive import (
    PredictionService,
    health_state,
    MODEL_VERSION,
    RUL_CAP,
    HORIZON,
)
from src.schemas import Telemetry, Prediction, SLMAnalysis, MachineContext
from src.simulation import MachineSimulator, SimulationService, VALID_SCENARIOS
from src.slm_dataset import generate_dataset, KNOWN_CONDITIONS
from src.slm_eval import run_harness, load_test_examples
from src.slm_service import SLMService, HFBackend, DemoBackend, RemoteBackend
from src.train_slm import train, audit_resources
from src.validation import validate_slm_output, controlled_fallback, extract_json
from src.knowledge import KnowledgeService
from src.services import ServiceFactory, create_services, machine_state, chat

__all__ = [
    "load_config",
    "build_context",
    "render_prompt",
    "SENSOR_COLS",
    "INFORMATIVE_SENSORS",
    "load_fd001",
    "add_rul",
    "add_capped_rul",
    "engine_split",
    "assert_disjoint",
    "build_features",
    "feature_columns",
    "fit_scaler",
    "apply_scaler",
    "get_logger",
    "log_event",
    "PredictionService",
    "health_state",
    "MODEL_VERSION",
    "RUL_CAP",
    "HORIZON",
    "Telemetry",
    "Prediction",
    "SLMAnalysis",
    "MachineContext",
    "MachineSimulator",
    "SimulationService",
    "VALID_SCENARIOS",
    "generate_dataset",
    "KNOWN_CONDITIONS",
    "run_harness",
    "load_test_examples",
    "SLMService",
    "HFBackend",
    "DemoBackend",
    "RemoteBackend",
    "train",
    "audit_resources",
    "validate_slm_output",
    "controlled_fallback",
    "extract_json",
    "KnowledgeService",
    "ServiceFactory",
    "create_services",
    "machine_state",
    "chat",
]