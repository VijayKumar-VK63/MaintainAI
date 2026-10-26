"""MaintainAI: End-to-end predictive maintenance with fine-tuned SLM."""
from __future__ import annotations

__version__ = "0.1.0"

from src.config import load_config
from src.context_builder import build_context, render_prompt
from src.engine import (
    MaintainAIEngine,
    create_engine,
    PredictiveUnavailableError,
    TelemetryStaleError,
    UnknownMachineError,
)
from src.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_user_message,
    render_chat,
)
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
from src.slm_service import (
    SLMService,
    HFBackend,
    TransformersBackend,
    DemoBackend,
    OpenAICompatibleBackend,
    RemoteBackend,
)
from src.validation import validate_slm_output, controlled_fallback, extract_json
from src.knowledge import KnowledgeService
from src.services import ServiceFactory, create_services, machine_state, chat

__all__ = [
    "load_config",
    "build_context",
    "render_prompt",
    "SYSTEM_PROMPT",
    "build_messages",
    "build_user_message",
    "render_chat",
    "MaintainAIEngine",
    "create_engine",
    "UnknownMachineError",
    "PredictiveUnavailableError",
    "TelemetryStaleError",
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
    "SLMService",
    "HFBackend",
    "TransformersBackend",
    "DemoBackend",
    "OpenAICompatibleBackend",
    "RemoteBackend",
    "validate_slm_output",
    "controlled_fallback",
    "extract_json",
    "KnowledgeService",
    "ServiceFactory",
    "create_services",
    "machine_state",
    "chat",
]