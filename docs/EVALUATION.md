# Evaluation — predictive model (measured Phase 3, no fabrication)

- **Split**: 80 train / 20 val engines (disjoint); NASA test 100 trajectories evaluated once.
- **Target**: RUL regressor on capped RUL (cap=125); risk classifier = failure within 30 cycles.
- **Selection (val engines)**: Ridge RMSE 17.70 / MAE 14.41 → XGBoost RMSE 14.49 /
  MAE 10.41. Selected: XGBoost. Threshold tuned on val only: 0.65 (val F1 0.916).
- **Held-out NASA test (100 engines, uncapped official RUL)**:
  - RUL: RMSE **16.01**, MAE **11.52**, NASA score **416.7**
  - Risk (≤30 cycles): precision **0.958**, recall **0.920**, F1 **0.939**, ROC-AUC **0.994**
- **Latency**: ~68 ms per prediction (local CPU).
- **Limitations**: single condition/fault mode; cap=125 biases early-life RUL high;
  health bands (0.30/0.60/0.85) are fixed heuristics, not tuned. Artifacts:
  `models/predictive/{rul_model,risk_model,scaler_fd001}.joblib`,
  `metadata.json`, `metrics.json`, `feature_schema.json`.

## SLM comparison — Phase 7 (same 100 held-out scenarios; `scripts/compare_slm.py`)

| System | risk_acc | cond_acc | evidence_F1 | validity | Status |
|---|---|---|---|---|---|
| demo-fallback rule-mirror | 1.000 | 0.730 | 0.235 | 1.000 | measured locally |
| 135M base HF path check | 0.000 | 0.000 | 0.000 | 0.000 | measured locally (untrained tiny base echoes prompt — proves the harness path, not model quality) |
| Qwen2.5-3B base | pending | pending | pending | pending | Colab nb 06 |
| Qwen2.5-3B + LoRA | pending | pending | pending | pending | Colab nb 08 |

Combined machine-readable comparison: `reports/metrics.json`; narrative:
`reports/evaluation_report.md`. Pending = unmeasured, never zero-filled.
