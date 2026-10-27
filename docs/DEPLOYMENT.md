# Deployment

## Measured resource requirements

- Predictive pipeline: ~68 ms per prediction on CPU; artifacts ~1.8 MB.
- Qwen2.5-3B 4-bit: roughly 2 GB VRAM for inference, 4-6 GB for QLoRA.
- CPU llama.cpp Q4_K_M: ~2.5 GB RAM at 2048 context, ~12 s per constrained
  analysis on the measured 16-core CPU.
- Free Streamlit Community Cloud (CPU, ~1 GB RAM, no GPU) cannot serve the SLM.

## Supported topologies

### 1. Local fine-tuned SLM via llama.cpp (default, CPU)
Prepare the GGUF files once (see `docs/GGUF_SETUP.md`):
```bash
python scripts/prepare_local_slm.py
python api/main.py
# or: uvicorn api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```
`configs/development.yaml` uses `slm.backend: auto`, which selects the local
GGUF + GGUF LoRA when present. `.env` is loaded automatically. There is no
silent demo fallback: if no model/endpoint is configured, startup fails with a
clear message.

### 2. Dashboard on Streamlit Cloud + remote SLM
- Commit the small `models/predictive/*.joblib` artifacts (no longer gitignored).
- The 1.8 GB GGUF is gitignored and too large for the free tier; use a hosted
  endpoint instead.
- Set Streamlit secrets:
  ```toml
  SLM_BACKEND = "openai"
  SLM_INFERENCE_ENDPOINT = "https://your-endpoint.example.com"
  SLM_INFERENCE_PATH = "/v1/chat/completions"
  SLM_API_KEY = "hf_xxxxxxxxxxxx"
  ```
- `app/streamlit_app.py` bridges `st.secrets` into environment variables before
  config loading. Without an endpoint the app shows an explicit error and stops
  rather than switching to the demo rule mirror.

### 3. Local llama.cpp server (OpenAI-compatible)
```bash
llama-server -m models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --lora models/slm/maintainai-qwen2.5-3b-lora.gguf \
  -c 2048 --port 8001 --host 0.0.0.0 --no-webui
export SLM_BACKEND=openai
export SLM_INFERENCE_ENDPOINT=http://localhost:8001
export SLM_INFERENCE_PATH=/v1/chat/completions
```

### 4. Transformers + PEFT (GPU/Colab)
```bash
export SLM_BACKEND=transformers
export SLM_ADAPTER_PATH=/path/to/slm/checkpoints/final
```

### 5. vLLM / TGI / HF Inference Endpoint
Use the `openai` backend with the endpoint base URL and bearer token. TGI's
legacy endpoint is supported by setting `SLM_INFERENCE_PATH=/generate`.

## API routes
- `GET /` service info
- `GET /health` backend, predictive status, simulator status
- `GET /machines`, `GET /machines/{id}`, `GET /machines/{id}/telemetry`
- `POST /simulation/{start,stop,reset,tick,scenario}`
- `POST /assistant/chat`

`python api/main.py` starts Uvicorn on `127.0.0.1:8000` (override with
`API_HOST`/`API_PORT`).

## Safety
- The SLM never receives or invents raw telemetry beyond the rendered context.
- Invalid model output becomes a schema-valid UNKNOWN fallback.
- Missing predictor -> HTTP 503; stale telemetry -> HTTP 409.
- No secrets are committed; `.env` is gitignored.
