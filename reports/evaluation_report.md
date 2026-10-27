# SLM evaluation — base vs fine-tuned

Same 100 held-out NASA decision points; inference through `SLMService.analyze`.

| System | Status | Risk acc | Condition acc | Evidence F1 | Recommendation acc | Validity | Latency ms |
|---|---|---|---|---|---|---|---|
| demo-fallback | measured | 1.000 | 0.730 | 0.235 | 1.000 | 1.000 | 0.021 |
| qwen2.5-3b-base | measured | 0.650 | 0.590 | 0.000 | 0.000 | 0.800 | 32267.842 |
| qwen2.5-3b-lora | measured | 0.980 | 0.730 | 0.701 | 0.940 | 1.000 | 12102.105 |

> Local CPU inference via llama.cpp (Q4_K_M); latency is CPU-only and not comparable to GPU numbers. The commands below use `--constrained` (GBNF), so JSON shape/enums are guaranteed; any remaining invalid outputs are token-cap truncations or schema length-limit violations, not malformed JSON.

> `pending` means unmeasured; `invalid` means the harness failed. Numbers are never zero-filled.

Regenerate with:

```bash
python -m src.slm_eval --backend demo --out reports/metrics_demo.json
# CPU-only (llama.cpp, in-process)
python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --constrained --out reports/metrics_base.json
python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --adapter models/slm/maintainai-qwen2.5-3b-lora.gguf \
  --constrained --out reports/metrics_finetuned.json
# GPU (transformers + PEFT)
python -m src.slm_eval --backend hf --out reports/metrics_base.json
python -m src.slm_eval --backend hf --adapter <adapter> --out reports/metrics_finetuned.json
python scripts/compare_slm.py
```
