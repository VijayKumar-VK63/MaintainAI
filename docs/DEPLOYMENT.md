# Deployment (Phase 11 — decided from measured resource requirements)

- **Measured**: predictive pipeline ~68 ms on CPU, ~2 KB artifacts — serves anywhere.
  Qwen2.5-3B 4-bit needs ~2 GB (inference) / ~4–6 GB (QLoRA) VRAM — verified
  against published Unsloth/Colab reports; free Streamlit Community Cloud (CPU,
  ~1 GB RAM, no GPU) CANNOT serve it. No production-availability claim.
- **Chosen free-first architecture**: Streamlit frontend (Community Cloud or
  local) + remote inference endpoint for the SLM (Colab-hosted, RunPod/vLLM,
  or HF Inference Endpoints — pick at deploy time via `SLM_ADAPTER_PATH` /
  `inference_endpoint` config). No code changes needed: `SLMService` backend swap.
- **Local/demo mode** (works today): `streamlit run app/streamlit_app.py` +
  `uvicorn api.main:app`. SLM falls back to tagged demo mode with a banner;
  monitoring + numerical prediction fully live. API: `uvicorn api.main:app --reload`.
- **Colab training**: notebooks 01–08; Drive layout `MaintainAI/{datasets,models,checkpoints,experiments,logs,exports}/`;
  adapter published to Hugging Face Hub (LoRA only). App never depends on personal Drive.
