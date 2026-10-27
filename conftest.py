import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
# api.main builds its module-level app at import; tests must not need the 1.8 GB
# local GGUF. Individual tests use configs/test.yaml or clear this via the fixture.
os.environ.setdefault("SLM_BACKEND", "demo")


@pytest.fixture(autouse=True)
def _clear_slm_env(monkeypatch):
    """Keep a developer's local .env from leaking into deterministic tests."""
    for key in list(os.environ):
        if key.startswith("SLM_") or key in ("HF_TOKEN", "API_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
