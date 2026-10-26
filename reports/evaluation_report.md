# SLM evaluation — base vs fine-tuned

Same 100 held-out NASA decision points; inference through `SLMService.analyze`.

| System | Status | Risk acc | Condition acc | Evidence F1 | Recommendation acc | Validity | Latency ms |
|---|---|---|---|---|---|---|---|
| demo-fallback | measured | 1.000 | 0.730 | 0.235 | 1.000 | 1.000 | 0.021 |
| qwen2.5-3b-base | pending | — | — | — | — | — | — |
| qwen2.5-3b-lora | pending | — | — | — | — | — | — |

> `pending` means unmeasured; `invalid` means the harness failed. Numbers are never zero-filled.

Regenerate with:

```bash
python -m src.slm_eval --backend demo --out reports/metrics_demo.json
python -m src.slm_eval --backend hf --out reports/metrics_base.json
python -m src.slm_eval --backend hf --adapter <adapter> --out reports/metrics_finetuned.json
python scripts/compare_slm.py
```
