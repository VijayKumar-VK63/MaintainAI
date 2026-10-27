# MaintainAI — End-to-End Predictive Maintenance with a Fine-Tuned SLM

> **A complete, free-first predictive maintenance system combining traditional ML (XGBoost) with a fine-tuned Small Language Model (Qwen2.5-3B + LoRA) for interpretable maintenance recommendations.**

---

## 🎯 What This Project Does

**MaintainAI** simulates an industrial turbofan engine, predicts when it will fail, and uses an AI assistant to explain *why* and *what to do about it* — all running locally on CPU with optional GPU inference via Hugging Face.

### The Complete Flow
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Simulated  │────▶│  Feature    │────▶│  XGBoost    │────▶│  Context    │────▶│  Fine-Tuned │
│  Telemetry  │     │  Engineering│     │  (RUL +     │     │  Builder    │     │  Qwen2.5-3B │
│  (CMAPSS)   │     │  (Causal)   │     │   Risk)     │     │             │     │  + LoRA     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                                       │
                                                                                       ▼
                                                                              ┌─────────────┐
                                                                              │ Structured  │
                                                                              │ JSON Output │
                                                                              │ Risk,       │
                                                                              │ Condition,  │
                                                                              │ Evidence,   │
                                                                              │ Action      │
                                                                              └─────────────┘
                                                                                       │
                                                                                       ▼
                                                                              ┌─────────────┐
                                                                              │  Streamlit  │
                                                                              │  Dashboard  │
                                                                              │  + Chat     │
                                                                              └─────────────┘
```

---

## 🤔 Why This Project Exists

### The Problem
Industrial machines fail unexpectedly, causing:
- **Unplanned downtime** — Costs $50K–$500K/hour in manufacturing
- **Safety risks** — Catastrophic failures can injure people
- **Wasted maintenance** — Fixed-schedule maintenance replaces parts too early/late

### The Solution Approach
Traditional ML (XGBoost) is great at **predicting numbers** (RUL, failure probability) but can't explain *what to do*. LLMs are great at **explaining** but hallucinate sensor values.

**MaintainAI bridges this gap:**
1. **XGBoost** predicts: "Failure probability: 87%, RUL: 23 cycles"
2. **Fine-tuned SLM** interprets: "HIGH risk — HPC degradation detected. Evidence: rising HPC outlet temp (s3), falling core speed (s9). Recommend: Schedule HPC inspection immediately (URGENCY: HIGH). Confidence: 84%"

---

## 📊 What's Been Built (Complete Status)

### ✅ Phase 0–1: Foundation & Architecture
- **Project structure** with clean separation: `src/`, `api/`, `app/`, `notebooks/`, `configs/`, `data/`, `models/`, `knowledge/`, `tests/`
- **Configuration system** — YAML + environment variables, no hardcoded paths/secrets
- **Structured logging** — JSON format, secrets redacted automatically
- **Pydantic schemas** — Telemetry, Prediction, SLMAnalysis, MachineContext with validation

### ✅ Phase 2: Dataset (NASA CMAPSS FD001)
- **Source**: NASA PCoE CMAPSS Jet Engine Simulated Data (public domain)
- **Scope**: 100 training + 100 test engines, single operating condition, HPC degradation fault
- **Sensors**: 21 raw → 14 informative (constant sensors removed: s1, s5, s6, s10, s16, s18, s19)
- **Leakage prevention**: Engine-level splits (never row-level), scalers fit on train only, NASA test never touched during training

### ✅ Phase 3: Preprocessing & Causal Features
- **Causal rolling features** — `.shift(1)` ensures no future leakage
- **Features per sensor**: rolling mean/std (window=30), trend, delta-from-baseline
- **Train-only scaler** — StandardScaler fit on 80 train engines only
- **Output**: `data/processed/train_features.parquet`, `models/predictive/scaler_fd001.joblib`

### ✅ Phase 4: Predictive Model (XGBoost)
- **Two heads**: 
  - RUL Regressor (capped at 125 cycles)
  - Risk Classifier (failure within 30 cycles)
- **Model selection**: XGBoost beats Ridge on validation engines
- **Threshold tuning**: Best-F1 on validation only (0.65)
- **Held-out test (100 NASA engines)**:
  - RUL: RMSE 16.01, MAE 11.52, NASA Score 416.7
  - Risk: Precision 0.958, Recall 0.920, F1 0.939, ROC-AUC 0.994

### ✅ Phase 5: SLM Dataset (Grounded Instruction Tuning)
- **880 examples** (640 train / 140 val / 100 test) — **engine-disjoint splits**
- **Zero hand-written answers** — All targets rule-derived from:
  - Predictive model outputs (failure prob, RUL, anomaly score)
  - Causal sensor trends (baseline deviation >2σ)
  - True RUL stages (early/mid/late stratification)
- **Schema**: `SLMAnalysis` — risk_level, likely_condition, confidence, evidence[], recommended_action, urgency
- **Evidence vocabulary**: Closed set (e.g., `s3_above_baseline`, `rul_low`, `trend_increasing_multiple_sensors`)

### ✅ Phase 6: QLoRA Fine-Tuning (Colab GPU)
- **Base**: Qwen2.5-3B-Instruct (4-bit NF4 quantization)
- **Method**: QLoRA — LoRA r=16, α=32, dropout=0.05, targets all attention + MLP layers
- **Training**: 3 epochs configured (`configs/training.yaml`), batch=1, grad_accum=4, lr=2e-4, fp16 on T4
- **Checkpoints**: Saved to Google Drive, best model selected on validation loss
- **Adapter pushed to HF Hub**: `Vijay-kumar-63/maintainai-qwen2.5-3b-lora`
- **Existing local adapter**: `slm/checkpoints/final` (1-epoch run, 160 steps); rerun notebook 07 for the 3-epoch adapter

### ✅ Phase 7: Evaluation (Base vs Fine-Tuned)
- **Same 100 held-out scenarios** for both models
- **Metrics**: Risk accuracy, Condition accuracy, Evidence F1, Recommendation accuracy, Validity rate, Uncertainty skill, Latency
- **Single evaluation path**: `src.slm_eval.run_service_eval` calls the same `SLMService.analyze` used by the API/dashboard
- **Commands**: `python -m src.slm_eval --backend demo|hf|llamacpp|openai ...` then `python scripts/compare_slm.py`
- **Status**: demo-fallback, Qwen2.5-3B base, and 1-epoch LoRA measured locally on CPU (llama.cpp Q4_K_M, grammar-constrained); LoRA is 100% valid with 0.980 risk, 0.730 condition, 0.701 evidence F1; 3-epoch adapter GPU rerun pending

### ✅ Phase 8: Simulator & FastAPI
- **Physics-grounded simulator** — FD001 sensor baselines + degradation drift patterns
- **Scenarios**: NORMAL → DEGRADATION → CRITICAL → FAILURE (deterministic per seed)
- **FastAPI endpoints**: `/machines`, `/machines/{id}`, `/simulation/*`, `/assistant/chat`
- **Error contract**: SLM failure → demo fallback (HTTP 200), Predictive failure → HTTP 503, Stale telemetry → HTTP 409

### ✅ Phase 9: Context Builder & SLM Service
- **ContextBuilder** — Single source of truth for prompt construction
- **SLMService** — Backend protocol (HFBackend, RemoteBackend, DemoBackend)
- **Structured output validation** — Pydantic + retry + controlled fallback
- **Chat assistant** — Grounded in current machine context

### ✅ Phase 10: Streamlit Dashboard (Improved UI)
- **Live Monitor** — Real-time sensor charts, failure probability trend, health state
- **AI Analysis** — Risk level, condition, confidence, evidence tags, recommended action
- **Maintenance Chat** — Ask questions, quick-action buttons
- **Evaluation Tab** — Predictive + SLM metrics tables
- **About Tab** — Complete architecture documentation

### ✅ Phase 11: Deployment Ready
- **Local demo** — CPU-only, SLM falls back to demo mode with banner
- **Local fine-tuned inference** — llama.cpp / vLLM / TGI via the OpenAI-compatible backend
- **Production** — Streamlit Cloud (dashboard) + HF Inference Endpoint or llama.cpp (SLM)
- **Config-driven** — `slm.backend` = `demo` | `transformers` | `openai`; endpoints and keys come from env/secrets

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **ML/MLOps** | scikit-learn, XGBoost, Pandas, NumPy, PyArrow, Joblib |
| **SLM/Fine-Tuning** | Hugging Face Transformers, PEFT, TRL, bitsandbytes, Accelerate |
| **Quantization** | 4-bit NF4 (QLoRA) |
| **API** | FastAPI, Pydantic, Uvicorn |
| **UI** | Streamlit, Plotly |
| **Config** | YAML, python-dotenv |
| **Testing** | pytest (46 passing, 1 optional network test skipped) |
| **Inference** | Transformers+PEFT, llama.cpp, vLLM, TGI, HF Inference Endpoints |

---

## 🚀 Quickstart (Local CPU Demo)

### Prerequisites
- Python 3.9+
- Git
- 4GB+ RAM

### Installation
```bash
# Clone
git clone https://github.com/VijayKumar-VK63/MaintainAI.git
cd MaintainAI

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest -q
# Should show: 45 passed, 1 skipped
```

### Run the Demo (2 Terminals)

**Terminal 1 — API Server:**
```bash
cd MaintainAI
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit Dashboard:**
```bash
cd MaintainAI
# Windows PowerShell:
$env:PYTHONPATH="C:\path\to\MaintainAI"; streamlit run app/streamlit_app.py

# Linux/Mac:
PYTHONPATH=/path/to/MaintainAI streamlit run app/streamlit_app.py
```

### Access the Dashboard
- **API Health**: http://127.0.0.1:8000/health
- **Dashboard**: http://localhost:8501

---

## ☁️ Streamlit Cloud Deployment (Free)

Deploy the dashboard directly to Streamlit Community Cloud — no Docker, no GPU needed.

### One-Click Deploy
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select your fork → Main branch → `app/streamlit_app.py`
4. Click "Deploy"

### How It Works on Streamlit Cloud
- **Predictive model (XGBoost)** — Runs locally on CPU with no external server.
  Commit the small `models/predictive/*.joblib` artifacts (no longer gitignored).
- **SLM (Qwen2.5-3B+LoRA)** — Uses **demo fallback** unless an endpoint is configured
  - Publish your LoRA adapter to the HF Hub and serve it with llama.cpp/vLLM/TGI
    or an HF Inference Endpoint, then set the secrets below
- **Simulator** — Generates realistic FD001 telemetry in real-time

### Streamlit Secrets
`app/streamlit_app.py` bridges `st.secrets` into environment variables before
config loading, so the same config works locally and on Streamlit Cloud.
```toml
# .streamlit/secrets.toml (Streamlit Cloud dashboard)
SLM_BACKEND = "openai"
SLM_INFERENCE_ENDPOINT = "https://your-endpoint.example.com"
SLM_INFERENCE_PATH = "/v1/chat/completions"
SLM_API_KEY = "hf_xxxxxxxxxxxx"
HF_TOKEN = "hf_xxxxxxxxxxxx"
```

> **Note**: Without an endpoint, the app runs in **demo mode** — numerical predictions (RUL, failure probability) are fully functional; AI analysis uses a rule-based fallback that mirrors the predictive model's outputs. With `SLM_BACKEND=transformers`, the app loads the adapter locally (GPU/Colab only).

---

## 🎮 How to Demo

1. **Select Machine** → M001 (or M002, M003)
2. **Click "Start Monitoring"** → Telemetry begins streaming
3. **Select "DEGRADATION" scenario** → Click "Apply Scenario"
4. **Watch live charts** → Sensors drift, failure probability rises
5. **Observe AI Analysis** → Risk level, condition, evidence, action
6. **Chat with Assistant** → Ask: "Why is risk HIGH?", "What should maintenance inspect?"

### Scenarios Explained
| Scenario | Behavior | Expected Result |
|----------|----------|-----------------|
| **NORMAL** | Stable sensors, small noise | HEALTHY, failure prob < 30% |
| **DEGRADATION** | Gradual sensor drift (HPC temp ↑, pressure ↓) | WARNING → HIGH_RISK, prob 30–85% |
| **CRITICAL** | Rapid sensor divergence | HIGH_RISK → CRITICAL, prob > 85% |
| **FAILURE** | Extreme anomalies | CRITICAL, RUL → 0 |

---

## 📁 Project Structure Explained

```
MaintainAI/
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py    # Main UI (Monitor + AI analysis + Evaluation)
│   ├── services_local.py   # Local service factory for Streamlit
│   └── __init__.py
├── api/                    # FastAPI server
│   ├── main.py             # Endpoints + error handling
│   └── __init__.py
├── src/                    # Core library (importable)
│   ├── config.py           # YAML + env var config loader
│   ├── schemas.py          # Pydantic models (Telemetry, Prediction, SLMAnalysis)
│   ├── data_cmapss.py      # NASA FD001 loader + RUL + engine splits
│   ├── features.py         # Causal feature engineering (shift(1))
│   ├── predictive.py       # XGBoost RUL + Risk models + PredictionService
│   ├── simulation.py       # Seeded turbofan simulator (FD001 grounded)
│   ├── prompts.py          # Canonical system/user messages (train == serve)
│   ├── context_builder.py  # MachineContext → evidence message
│   ├── engine.py           # Single telemetry→prediction→analysis pipeline
│   ├── slm_dataset.py      # Grounded instruction dataset generator
│   ├── slm_eval.py         # Held-out evaluation (production path)
│   ├── slm_service.py      # SLMService + Demo/Transformers/OpenAI backends
│   ├── train_slm.py        # QLoRA training script (Colab)
│   ├── validation.py       # SLM output validation + fallback
│   ├── knowledge.py        # YAML knowledge loader (failure modes, actions)
│   ├── services.py         # Compatibility ServiceFactory over the engine
│   └── logging_utils.py    # Structured JSON logging (secrets redacted)
├── notebooks/              # Colab notebooks (01–08)
│   ├── 01_environment_check.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_predictive_model.ipynb
│   ├── 05_slm_dataset_creation.ipynb
│   ├── 06_base_slm_evaluation.ipynb
│   ├── 07_qlora_finetuning.ipynb
│   └── 08_finetuned_evaluation.ipynb
├── configs/                # YAML configs (dev, training, deployment)
├── data/                   # Processed data + SLM datasets
├── models/                 # Saved models (predictive + SLM adapter)
├── knowledge/              # YAML knowledge base
├── scripts/                # compare_slm.py and utilities
├── tests/                  # 45 pytest tests (unit + integration)
├── docs/                   # Documentation
└── reports/                # Evaluation metrics + comparison tables
```

---

## 🔧 Configuration

### Environment Variables (`.env`)
```bash
# Copy .env.example to .env and fill in
SLM_BACKEND=demo                          # demo | transformers | llamacpp | openai
SLM_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
SLM_MODEL_PATH=                           # Base .gguf when SLM_BACKEND=llamacpp
SLM_ADAPTER_PATH=                         # PEFT dir or GGUF LoRA
SLM_LORA_SCALE=1.0                        # llama.cpp LoRA multiplier
SLM_CONSTRAINED=true                      # Force schema-valid JSON (GBNF)
SLM_INFERENCE_ENDPOINT=                   # Used when SLM_BACKEND=openai
SLM_INFERENCE_PATH=/v1/chat/completions   # llama.cpp/vLLM/TGI (use /generate for TGI)
SLM_API_KEY=                              # Bearer token (falls back to HF_TOKEN)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
API_BASE_URL=http://127.0.0.1:8000
MODEL_DIR=models
```

### SLM Backend Modes
| Mode | When to use | What loads |
|------|-------------|------------|
| `demo` | Default local/CPU/Streamlit Cloud | Rule-based mirror, clearly bannered |
| `llamacpp` | CPU-only local, no GPU/server | Quantized GGUF + optional GGUF LoRA, in-process, optional GBNF JSON grammar |
| `transformers` | Colab/GPU/local with adapter | 4-bit Qwen + PEFT adapter |
| `openai` | llama.cpp, vLLM, TGI, HF endpoint | HTTP chat-completions backend |

### Config Files
| File | Purpose |
|------|---------|
| `configs/development.yaml` | Local dev (demo backend) |
| `configs/training.yaml` | Colab QLoRA training (3 epochs) |
| `configs/deployment.yaml` | Production (OpenAI-compatible endpoint) |

---

## 🧪 Testing

```bash
# Run all tests
pytest -q

# Run specific test files
pytest tests/test_predictive.py -v
pytest tests/test_slm_service.py -v
pytest tests/test_simulation_api.py -v

# Coverage
pytest --cov=src --cov-report=html
```

### SLM Evaluation
```bash
# Demo backend (no GPU) — measured baseline of the rule mirror
python -m src.slm_eval --backend demo --out reports/metrics_demo.json

# Base model (Colab/GPU)
python -m src.slm_eval --backend hf --out reports/metrics_base.json

# Fine-tuned adapter (Colab/GPU, or a Hub repo id)
python -m src.slm_eval --backend hf --adapter slm/checkpoints/final \
  --out reports/metrics_finetuned.json

# CPU-only, in-process llama.cpp (no server, no GPU)
python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --constrained --out reports/metrics_base.json
python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --adapter models/slm/maintainai-qwen2.5-3b-lora.gguf \
  --constrained --out reports/metrics_finetuned.json

# Remote llama.cpp/vLLM/TGI endpoint
python -m src.slm_eval --backend openai --endpoint http://localhost:8001 \
  --out reports/metrics_finetuned.json

python scripts/compare_slm.py
```

---

## 📈 Evaluation Results (Measured)

### Predictive Model (Held-out 100 NASA test engines)
| Metric | Value |
|--------|-------|
| RUL RMSE | 16.01 cycles |
| RUL MAE | 11.52 cycles |
| NASA Score | 416.7 |
| Risk Precision | 0.958 |
| Risk Recall | 0.920 |
| Risk F1 | 0.939 |
| ROC-AUC | 0.994 |

### SLM Comparison (Same 100 Scenarios)
| System | Risk Acc | Cond Acc | Evidence F1 | Validity | Status |
|--------|----------|----------|-------------|----------|--------|
| Demo Fallback | 1.000 | 0.730 | 0.235 | 1.000 | ✅ Measured |
| Qwen2.5-3B Base (Q4_K_M, CPU, grammar) | 0.650 | 0.590 | 0.000 | 0.800 | ✅ Measured |
| Qwen2.5-3B + LoRA (Q4_K_M, CPU, grammar) | 0.980 | 0.730 | 0.701 | 1.000 | ✅ Measured |

> Local CPU run, 12 threads, `max_new_tokens=160`, grammar-constrained JSON
> (`--constrained`). LoRA: recommendation acc 0.940, unknown-uncertainty 0.909,
> known-confidence 0.962, 12.1 s/example. Base: recommendation acc 0.000,
> unknown-uncertainty 0.000, 32.3 s/example; 20/100 generations hit the token
> cap while rambling inside a field, so validity is 0.800. The fine-tuned model
> is 100% schema-valid and beats the base on every content metric. The local
> adapter is the 1-epoch checkpoint (160 steps); the 3-epoch Colab adapter
> should improve condition accuracy. `pending` means unmeasured, never
> zero-filled. Unconstrained ablation: base 0% valid, LoRA 64% valid
> (`reports/metrics_*_unconstrained.json`).

---

## 🚀 Running the Fine-Tuned Model

### Option A — llama.cpp (CPU, local)
Run the quantized model in-process (no server) or behind an OpenAI-compatible
server. Prepare the GGUF files once:
```bash
python scripts/prepare_local_slm.py   # clone llama.cpp, download base, convert LoRA
```
In-process (used by the dashboard and `src.slm_eval`):
```bash
export SLM_BACKEND=llamacpp
export SLM_MODEL_PATH=models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf
export SLM_ADAPTER_PATH=models/slm/maintainai-qwen2.5-3b-lora.gguf
export SLM_CONSTRAINED=true
```
Server mode (same files):
```bash
llama-server -m models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --lora models/slm/maintainai-qwen2.5-3b-lora.gguf \
  -c 2048 --port 8001 --host 0.0.0.0 --no-webui
export SLM_BACKEND=openai
export SLM_INFERENCE_ENDPOINT=http://localhost:8001
export SLM_INFERENCE_PATH=/v1/chat/completions
```

### Option B — Transformers + PEFT (GPU/Colab)
```bash
export SLM_BACKEND=transformers
export SLM_ADAPTER_PATH=/path/to/slm/checkpoints/final
```

### Option C — HF Inference Endpoint / vLLM / TGI
```bash
export SLM_BACKEND=openai
export SLM_INFERENCE_ENDPOINT=https://your-endpoint.example.com
export SLM_API_KEY=hf_xxxxxxxxxxxx
export SLM_INFERENCE_PATH=/v1/chat/completions
```

The dashboard/API banner reports which backend is active (`slm_backend`).
No endpoint configured -> demo fallback with a visible warning.

---

## 🔮 Near-Future Roadmap

### Immediate (Next 1–2 Weeks)
- [ ] **Complete notebook 08** — 3-epoch adapter evaluation on Colab GPU
- [x] **Update comparison table** — Base vs 1-epoch LoRA measured locally on CPU (Q4_K_M)
- [ ] **Deploy HF Inference Endpoint** — Enable production SLM
- [ ] **Add SHAP explanations** — Feature importance for XGBoost predictions

### Short Term (1–2 Months)
- [ ] **Multi-fault support** — FD002/FD003/FD004 (multiple operating conditions)
- [ ] **Automated retraining pipeline** — Scheduled model updates
- [ ] **Real-time streaming** — Kafka/Redis for production telemetry
- [ ] **CMMS integration** — Export work orders to Maximo/SAP PM
- [ ] **Uncertainty quantification** — Prediction intervals for RUL

### Medium Term (3–6 Months)
- [ ] **Edge deployment** — ONNX/TensorRT for edge devices
- [ ] **Multi-language SLM** — Support non-English maintenance teams
- [ ] **Explainable AI** — SHAP/LIME integration in dashboard
- [ ] **A/B testing framework** — Compare model versions in production
- [ ] **Automated drift detection** — Data/model drift alerts

### Long Term (6+ Months)
- [ ] **Digital twin integration** — Physics-informed predictions
- [ ] **Federated learning** — Cross-fleet learning without data sharing
- [ ] **Regulatory compliance** — ISO 13374, MIMOSA OSA-CBM
- [ ] **Generative maintenance procedures** — SLM writes step-by-step work instructions

---

## ⚠️ Limitations & Disclaimers

| Limitation | Impact |
|------------|--------|
| **Simulated telemetry only** | Not validated on physical hardware |
| **Single fault mode (FD001)** | HPC degradation only; no multi-fault generalization |
| **Dataset-dependent** | Predictions only valid for similar turbofan profiles |
| **Free-tier HF endpoint** | Rate limits, cold starts, no SLA |
| **Demo mode locally** | SLM falls back to rule-based mirror on CPU |
| **Not safety-certified** | Decision support only — humans make maintenance decisions |

---

## 📚 Documentation

| File | Description |
|------|-------------|
| `docs/DATASET.md` | FD001 details, license, leakage policy |
| `docs/SLM_DATASET.md` | Instruction dataset generation methodology |
| `docs/TRAINING.md` | QLoRA training config & results |
| `docs/EVALUATION.md` | Predictive + SLM evaluation metrics |
| `docs/DEPLOYMENT.md` | Free-tier deployment architecture |
| `docs/ARCHITECTURE.md` | Mermaid architecture diagram |
| `docs/COLAB_GUIDE.md` | Step-by-step Colab execution guide |
| `docs/GGUF_SETUP.md` | llama.cpp conversion + CPU inference |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest -q`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` file for details.

**Dataset**: NASA CMAPSS (public domain, NASA Open Data Portal)

---

## 🙏 Acknowledgments

- **NASA PCoE** — CMAPSS dataset
- **Hugging Face** — Transformers, PEFT, TRL, Inference Endpoints
- **Qwen Team** — Qwen2.5-3B-Instruct base model
- **XGBoost** — Gradient boosting library
- **Streamlit** — Rapid dashboard development
- **FastAPI** — Modern API framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/VijayKumar-VK63/MaintainAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VijayKumar-VK63/MaintainAI/discussions)

---

**Built with ❤️ for predictive maintenance research and education.**

> *MaintainAI demonstrates that combining conventional ML (for accurate predictions) with fine-tuned SLMs (for interpretable recommendations) creates a more actionable maintenance system than either alone.*