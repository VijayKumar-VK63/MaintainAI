# Architecture

```mermaid
flowchart TD
    SIM[Simulator] --> ENG[MaintainAIEngine]
    ENG --> PRE[Causal features]
    PRE --> ML[XGBoost: RUL + risk]
    ML --> CTX[ContextBuilder]
    CTX --> MSG[prompts.build_messages]
    MSG --> BACKEND{SLM backend (auto)}
    BACKEND -->|llamacpp| LLAMA[GGUF + LoRA, CPU]
    BACKEND -->|transformers| HF[Qwen + LoRA 4-bit]
    BACKEND -->|openai| HTTP[llama.cpp / vLLM / TGI / HF]
    BACKEND -->|demo, explicit only| DEMO[Rule mirror]
    LLAMA --> VAL[Validation + fallback]
    DEMO --> VAL
    HF --> VAL
    HTTP --> VAL
    VAL --> API[FastAPI]
    VAL --> UI[Streamlit]
```

## Invariants
- `MaintainAIEngine` is the only orchestrator; API and UI are transport layers.
- `src/prompts.py` is the only prompt builder; training and inference use the
  same chat messages.
- SLM failures fall back to a schema-valid UNKNOWN analysis, never crash.
- No configured backend -> explicit startup error; `demo` requires opt-in.
- Predictive failures return 503; stale telemetry returns 409.
- No large model files or secrets in git.
