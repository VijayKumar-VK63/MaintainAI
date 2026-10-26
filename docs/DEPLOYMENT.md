# Deployment

## Measured resource requirements

- Predictive pipeline: ~68 ms per prediction on CPU; artifacts ~1.8 MB.
- Qwen2.5-3B 4-bit: roughly 2 GB VRAM for inference, 4–6 GB for QLoRA.
- Free Streamlit Community Cloud (CPU, ~1 GB RAM, no GPU) cannot serve the SLM.

## Supported topologies

### 1. Local demo (works today, no GPU)
```bash
uvicorn api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```
`SLM_BACKEND=demo` (default): monitoring + XGBoost predictions are live; the AI
analysis is a rule-based mirror with a visible banner.

### 2. Dashboard on Streamlit Cloud + remote SLM
- Commit the small `models/predictive/*.joblib` artifacts (no longer gitignored).
- Set Streamlit secrets:
  ```toml
  SLM_BACKEND = "openai"
  SLM_INFERENCE_ENDPOINT = "https://your-endpoint.example.com"
  SLM_INFERENCE_PATH = "/v1/chat/completions"
  SLM_API_KEY = "hf_xxxxxxxxxxxx"
  ```
- `app/streamlit_app.py` bridges `st.secrets` into environment variables before
  config loading.

### 3. Local fine-tuned model via llama.cpp (CPU)
Convert the base + adapter to GGUF (see `docs/GGUF_SETUP.md`), then:
```bash
llama-server -m maintainai-qwen2.5-3b.q4_k_m.gguf --lora adapter.gguf \
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

## Safety
- The SLM never receives or invents raw telemetry beyond the rendered context.
- Invalid model output becomes a schema-valid UNKNOWN fallback.
- Missing predictor -> HTTP 503; stale telemetry -> HTTP 409.
- No secrets are committed; `.env` is gitignored.
