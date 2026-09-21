# Colab Training & Deployment Guide

## Overview
This guide walks through running the full training pipeline in Google Colab (free GPU) and deploying the fine-tuned model for inference.

---

## 1. One-Time Setup

### 1.1 Create Google Drive Structure
```bash
# In Colab notebook (run once)
from google.colab import drive
drive.mount('/content/drive')

# Create directory structure
!mkdir -p /content/drive/MyDrive/MaintainAI/{datasets,models,checkpoints,experiments,logs,exports,reports}
!mkdir -p /content/drive/MyDrive/MaintainAI/code
```

### 1.2 Copy Repository to Drive
```bash
# Option A: Clone directly in Colab
%cd /content/drive/MyDrive/MaintainAI
!git clone https://github.com/YOUR_USERNAME/MaintainAI.git code

# Option B: Upload zip via Colab file browser and unzip
# !unzip -q MaintainAI.zip -d /content/drive/MyDrive/MaintainAI/code
```

### 1.3 Verify Structure
```
/content/drive/MyDrive/MaintainAI/
├── code/                    # This repo (src/, notebooks/, configs/, etc.)
├── datasets/                # Large datasets (optional, CMAPSSData can stay in repo)
├── models/
│   └── predictive/          # Predictive model artifacts (from notebook 04)
├── checkpoints/
│   └── slm/                 # QLoRA checkpoints
├── experiments/
│   └── exp_001/             # Experiment tracking
├── logs/
├── exports/
└── reports/                 # Evaluation metrics (metrics_base.json, metrics_finetuned.json)
```

---

## 2. Notebook Execution Order

Run notebooks **in sequence** from `/content/drive/MyDrive/MaintainAI/code/notebooks/`:

| Notebook | Purpose | GPU Required | Runtime (T4) |
|----------|---------|--------------|--------------|
| `01_environment_check.ipynb` | Verify GPU, versions, VRAM | Yes | 30 sec |
| `02_dataset_analysis.ipynb` | Analyze FD001, document in `docs/DATASET.md` | No | 1 min |
| `03_preprocessing.ipynb` | Build causal features, fit scaler | No | 2 min |
| `04_predictive_model.ipynb` | Train XGBoost RUL + risk, evaluate on NASA test | No | 3 min |
| `05_slm_dataset_creation.ipynb` | Generate 880 grounded instruction examples | No | 1 min |
| `06_base_slm_evaluation.ipynb` | **MANDATORY** baseline before fine-tuning | Yes | 5-10 min |
| `07_qlora_finetuning.ipynb` | 4-bit QLoRA fine-tune Qwen2.5-3B | Yes | 30-45 min |
| `08_finetuned_evaluation.ipynb` | Evaluate fine-tuned model on same 100 scenarios | Yes | 5-10 min |

### Critical Rules
- **Run 06 BEFORE 07** — base model baseline is required
- **Never train on test data** — engine separation is enforced in code
- **Select checkpoint on VAL loss** — not train loss
- **Save all metrics to Drive** — `/content/drive/MyDrive/MaintainAI/reports/`

---

## 3. QLoRA Training Details (Notebook 07)

### Configuration (from `configs/training.yaml`)
```yaml
slm:
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  quantization: "4bit"
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  max_seq_length: 1024
  warmup_steps: 20
  seed: 42
  output_dir: "/content/drive/MyDrive/MaintainAI/slm/checkpoints/exp_001"
```

### Smoke Test First
Set `SMOKE = True` in notebook 07:
- 20 training examples
- 10 steps
- Verifies: data loading, model loading, forward/backward, checkpoint save/load
- Takes ~2 minutes on T4

### Full Training
Set `SMOKE = False`:
- 640 train + 140 val examples
- 3 epochs, eval every 50 steps
- Best model selected on validation loss
- Final adapter saved to `output_dir/final/`

### Memory Management
- 4-bit quantization: ~2.5 GB VRAM for base model
- LoRA adapters: ~1.9 MB (r=16)
- Gradient accumulation (4 steps) simulates batch size 4
- fp16 compute on T4

---

## 4. Post-Training: Copy Artifacts Locally

```bash
# From Colab terminal or notebook
!cp -r /content/drive/MyDrive/MaintainAI/reports/*.json /content/drive/MyDrive/MaintainAI/code/reports/
!cp -r /content/drive/MyDrive/MaintainAI/slm/checkpoints/exp_001/final /content/drive/MyDrive/MaintainAI/code/models/slm/adapter_latest

# Or download via Colab file browser
```

### Local Structure After Copy
```
models/
├── predictive/           # From notebook 04 (already in repo)
│   ├── rul_model.joblib
│   ├── risk_model.joblib
│   ├── scaler_fd001.joblib
│   ├── metadata.json
│   ├── metrics.json
│   └── feature_schema.json
└── slm/
    └── adapter_latest/   # Fine-tuned LoRA adapter (from Colab)
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer.json
        └── ...
```

---

## 5. Remote Inference Endpoint Deployment

### Option A: Hugging Face Inference Endpoints (Recommended)
1. Push adapter to HF Hub (from notebook 07):
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path='/content/drive/MyDrive/MaintainAI/slm/checkpoints/exp_001/final',
       repo_id='your-username/maintainai-qwen2.5-3b-lora',
       repo_type='model'
   )
   ```
2. Create Inference Endpoint on HF Hub (GPU: T4 or A10G)
3. Get endpoint URL: `https://api-inference.huggingface.co/models/your-username/maintainai-qwen2.5-3b-lora`
4. Update `configs/deployment.yaml`:
   ```yaml
   slm:
     inference_endpoint: "https://api-inference.huggingface.co/models/your-username/maintainai-qwen2.5-3b-lora"
   ```

### Option B: RunPod / Lambda Labs (vLLM)
1. Rent GPU instance (RTX 3090/4090, A100)
2. Deploy vLLM server with LoRA:
   ```bash
   # On remote server
   docker run --gpus all -p 8000:8000 \
     -v /path/to/adapter:/adapter \
     vllm/vllm-openai:latest \
     --model Qwen/Qwen2.5-3B-Instruct \
     --load-format auto \
     --enable-lora \
     --lora-modules maintainai=/adapter
   ```
3. Update `configs/deployment.yaml`:
   ```yaml
   slm:
     inference_endpoint: "http://YOUR_RUNPOD_IP:8000/v1/completions"
   ```

### Option C: Colab as Inference Server (Temporary)
```python
# In a persistent Colab notebook
from src.slm_service import SLMService, HFBackend
from fastapi import FastAPI
import uvicorn

app = FastAPI()
svc = SLMService(config={'slm': {'model_name': 'Qwen/Qwen2.5-3B-Instruct'}},
                 backend=HFBackend('Qwen/Qwen2.5-3B-Instruct',
                                   quantization='4bit',
                                   adapter_path='/content/drive/MyDrive/MaintainAI/slm/checkpoints/exp_001/final'))

@app.post("/generate")
async def generate(request: dict):
    prompt = request["prompt"]
    raw = svc.backend.generate(prompt)
    return {"generated_text": raw}

# Run with ngrok for public URL
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 6. Local Deployment (Demo Mode)

Works **today without GPU** — SLM falls back to tagged demo mode.

```bash
# Terminal 1: API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Streamlit
streamlit run app/streamlit_app.py
```

Dashboard shows:
- ✅ Live telemetry simulation
- ✅ Numerical predictions (RUL, failure probability, health state)
- ⚠️ SLM analysis in demo mode (banner displayed)
- ✅ Chat assistant (demo responses)

---

## 7. Free-Tier Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Streamlit      │     │  Remote SLM      │
│  Community Cloud│────▶│  Inference       │
│  (CPU, free)    │     │  (HF/RunPod)     │
└─────────────────┘     └──────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  FastAPI        │     │  Qwen2.5-3B      │
│  (local/API)    │     │  4-bit + LoRA    │
│  Predictive +   │     │  (GPU)           │
│  Simulator      │     │                  │
└─────────────────┘     └──────────────────┘
```

- **Streamlit**: Free Community Cloud (sleeps after inactivity)
- **FastAPI**: Can run on same free tier or Railway/Render free tier
- **SLM**: MUST run on GPU — decoupled via `inference_endpoint` config
- **No code changes** — swap backend via config only

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `max_seq_length` to 512, increase `gradient_accumulation_steps` |
| Drive disconnect | Re-run `drive.mount()`; use persistent Colab (Pro) |
| Slow generation | Use vLLM endpoint instead of transformers pipeline |
| Adapter not found | Verify `adapter_path` in config matches Drive/HF location |
| Test metrics not updating | Run `python scripts/compare_slm.py` after copying JSON files |

---

## 9. Reproducibility Checklist

Before claiming results:
- [ ] `01_environment_check` output saved (GPU, VRAM, versions)
- [ ] `06_base_slm_evaluation` metrics recorded (`metrics_base.json`)
- [ ] `07_qlora_finetuning` training loss curves saved
- [ ] Checkpoint selected on **VAL loss** (not train loss)
- [ ] `08_finetuned_evaluation` metrics recorded (`metrics_finetuned.json`)
- [ ] `scripts/compare_slm.py` run to generate comparison table
- [ ] `docs/EVALUATION.md` updated with actual numbers
- [ ] Adapter pushed to HF Hub (LoRA only, no base weights)
- [ ] `configs/deployment.yaml` updated with inference endpoint

---

## 10. Cost Estimate (Free Tier)

| Resource | Free Option | Paid Upgrade |
|----------|-------------|--------------|
| Colab GPU | T4 (15 GB VRAM, 12h limit) | Colab Pro: A100, longer sessions |
| HF Inference | 30k chars/month free | $0.032/hr for dedicated T4 |
| RunPod | - | $0.44/hr RTX 3090 |
| Streamlit | Community Cloud (sleeps) | - |
| API Hosting | Railway/Render free tier | $5-7/mo for always-on |

**Total free**: $0 (with HF free tier + Colab + Streamlit)
**Production**: ~$15-30/mo for always-on GPU inference