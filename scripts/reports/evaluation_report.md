# Evaluation report (actual measured values — pending cells stay pending)

## Predictive model — held-out NASA FD001 test (100 engines)
- RUL: RMSE 16.01, MAE 11.52, NASA score 416.7
- Risk (failure ≤ 30 cycles): precision 0.958, recall 0.920, F1 0.939, ROC-AUC 0.994
- Source: `models/predictive/metrics.json`

## SLM — 100 held-out decision-point scenarios
| System | risk_acc | cond_acc | evidence_F1 | rec_acc | validity | latency |
|---|---|---|---|---|---|---|
| demo | pending | pending | pending | pending | pending | pending |
| pipeline_check | pending | pending | pending | pending | pending | pending |
| base | pending | pending | pending | pending | pending | pending |
| finetuned | pending | pending | pending | pending | pending | pending |

## Notes
- demo: unmeasured — run the corresponding notebook/script first
- pipeline_check: unmeasured — run the corresponding notebook/script first
- base: unmeasured — run the corresponding notebook/script first
- finetuned: unmeasured — run the corresponding notebook/script first
