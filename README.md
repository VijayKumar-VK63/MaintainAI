# MaintainAI — End-to-End Predictive Maintenance with a Fine-Tuned SLM

Research/demo system. Telemetry is **simulated**; SLM output is **decision support only** —
a qualified human maintenance professional makes all maintenance decisions.
Not certified industrial safety software.

## Status: Phases 0–10 complete locally (33/33 tests pass)

Predictive model trained + held-out evaluated (test RMSE 16.01, risk F1 0.939).
SLM dataset (880) generated; SLMService + eval harness ready; simulator + API +
dashboard live in demo mode. QLoRA fine-tuning (Phase 6) + base-vs-finetuned
comparison (Phase 7) run in Colab (notebooks 06–08) — pending GPU runtime.

## Quickstart (local, CPU)

```bash
pip install -r requirements.txt
pytest -q
streamlit run app/streamlit_app.py   # Phase 10 (placeholder until then)
```

## Architecture

Telemetry → Preprocessing/Features → Predictive ML (risk/RUL) → ContextBuilder →
SLMService (Qwen2.5-3B + LoRA, configurable) → Validation → Streamlit UI.

The predictive model answers "what is the risk/state?"; the SLM answers
"what does it mean, what evidence supports it, what action is appropriate?"

## Docs

- `docs/DATASET.md` — FD001 selection, license, leakage policy
- `docs/ARCHITECTURE.md`, `docs/SLM_DATASET.md`, `docs/TRAINING.md`,
  `docs/EVALUATION.md`, `docs/DEPLOYMENT.md` — skeletons filled per phase

## Limitations (abridged)

Simulated telemetry; dataset-dependent predictions; free-tier deployment limits;
synthetic simulator ≠ physical machine.
