# Evaluation report (actual measured values — pending cells stay pending)

## Predictive model — held-out NASA FD001 test (100 engines)
- RUL: RMSE 16.01, MAE 11.52, NASA score 416.7
- Risk (failure ≤ 30 cycles): precision 0.958, recall 0.920, F1 0.939, ROC-AUC 0.994
- Source: `models/predictive/metrics.json`

## SLM — 100 held-out decision-point scenarios
| System | risk_acc | cond_acc | evidence_F1 | rec_acc | validity | latency |
|---|---|---|---|---|---|---|
 | demo | 1.000 | 0.730 | 0.235 | 1.000 | 1.000 | 0.008 | 
 | pipeline_check | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 16651.854 | 
| base | pending | pending | pending | pending | pending | pending |
| finetuned | pending | pending | pending | pending | pending | pending |

## Notes
- base: unmeasured — run the corresponding notebook/script first
- finetuned: unmeasured — run the corresponding notebook/script first
