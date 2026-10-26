# Evaluation

## Predictive model (measured, held-out NASA test)

- **Split**: 80 train / 20 val engines (disjoint); NASA test 100 trajectories
  evaluated once.
- **Target**: RUL regressor on capped RUL (cap=125); risk classifier = failure
  within 30 cycles.
- **Selection (val engines)**: Ridge RMSE 17.70 / MAE 14.41 -> XGBoost RMSE
  14.49 / MAE 10.41. Selected: XGBoost. Threshold tuned on val only: 0.65
  (val F1 0.916).
- **Held-out NASA test (100 engines, uncapped official RUL)**:
  - RUL: RMSE **16.01**, MAE **11.52**, NASA score **416.7**
  - Risk (<=30 cycles): precision **0.958**, recall **0.920**, F1 **0.939**,
    ROC-AUC **0.994**
- **Latency**: ~68 ms per prediction (local CPU).
- **Limitations**: single condition/fault mode; cap=125 biases early-life RUL
  high; health bands (0.30/0.60/0.85) are fixed heuristics, not tuned.

## SLM evaluation protocol

The evaluation harness calls the **same** `SLMService.analyze(context)` used by
the API and dashboard. Examples come from `data/slm/test.jsonl` (100 NASA
decision points). Metrics: risk accuracy, condition accuracy, evidence F1,
recommendation accuracy, schema validity, uncertainty skill, latency, and
failure count.

```bash
python -m src.slm_eval --backend demo --out reports/metrics_demo.json
python -m src.slm_eval --backend hf --out reports/metrics_base.json
python -m src.slm_eval --backend hf --adapter slm/checkpoints/final \
  --out reports/metrics_finetuned.json
python scripts/compare_slm.py
```

`pending` means unmeasured; `invalid` means the harness failed. Numbers are
never zero-filled.

## Current status

| System | risk_acc | cond_acc | evidence_F1 | validity | Status |
|---|---|---|---|---|---|
| demo-fallback rule mirror | 1.000 | 0.730 | 0.235 | 1.000 | measured |
| Qwen2.5-3B base | — | — | — | — | pending GPU rerun |
| Qwen2.5-3B + LoRA | — | — | — | — | pending GPU rerun |

> Earlier notebook 06/08 runs recorded 100/100 failures and all-zero metrics.
> Root cause: a bad `extract_json` import in the notebook-only evaluation path.
> The harness and notebooks now use `run_service_eval`, which uses the production
> analysis path and the canonical chat prompt.
