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

### CPU-only local run (no GPU, no server)

`scripts/prepare_local_slm.py` clones llama.cpp, downloads the Q4_K_M base GGUF,
and converts the PEFT adapter to GGUF. Then run the same harness in-process:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python scripts/prepare_local_slm.py

python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --n-threads 12 --constrained --out reports/metrics_base.json
python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --adapter models/slm/maintainai-qwen2.5-3b-lora.gguf \
  --n-threads 12 --constrained --out reports/metrics_finetuned.json
python scripts/compare_slm.py
```

Use `--limit N` for a quick subset and `--lora-scale` to adjust the adapter
strength. Expect roughly 15-40 s per example on a laptop CPU.

### Constrained JSON output

`--constrained` (or `SLM_CONSTRAINED=true`) applies a GBNF grammar that enforces
the `SLMAnalysis` shape, risk/urgency enums, and the 0-1 confidence range. The
model still chooses the values, so accuracy remains a model-quality metric, but
malformed JSON and wrong enum values are impossible and the production path no
longer falls back to UNKNOWN on malformed JSON. Schema length limits (128/512
characters, at most 10 evidence items) are still validated by pydantic; the
fine-tuned model stays within them on all 100 examples. The official local
reports use constrained decoding; `reports/metrics_*_unconstrained.json` keeps
the ablation that shows the format gap.

`pending` means unmeasured; `invalid` means the harness failed. Numbers are
never zero-filled.

## Current status

| System | risk_acc | cond_acc | evidence_F1 | validity | Status |
|---|---|---|---|---|---|
| demo-fallback rule mirror | 1.000 | 0.730 | 0.235 | 1.000 | measured |
| Qwen2.5-3B base (Q4_K_M, CPU, grammar) | 0.650 | 0.590 | 0.000 | 0.800 | measured |
| Qwen2.5-3B + LoRA (Q4_K_M, CPU, grammar) | 0.980 | 0.730 | 0.701 | 1.000 | measured |

Full local run (100 held-out scenarios, 12 CPU threads, `max_new_tokens=160`):

| System | Recommendation acc | Uncertainty (unknown / known) | Latency ms |
|---|---|---|---|
| demo-fallback | 1.000 | 0.091 / 0.013 | 0.021 |
| Qwen2.5-3B base | 0.000 | 0.000 / 1.000 | 32268 |
| Qwen2.5-3B + LoRA | 0.940 | 0.909 / 0.962 | 12102 |

> **Constrained decoding.** The official local reports use `--constrained`
> (GBNF) so the output shape, enums, and confidence range are guaranteed. The
> fine-tuned model is **100% schema-valid** on all 100 examples. The base model
> is 80% valid: 20 generations hit the 160-token cap while rambling inside a
> field, so `extract_json` could not close the object. The fine-tuned model
> completes every example within the same cap and is 2.7x faster.
>
> The fine-tuned adapter is the 1-epoch checkpoint (`slm/checkpoints/final`,
> 160 steps); the 3-epoch Colab adapter is not published yet. Condition accuracy
> (0.730) matches the rule mirror, while evidence F1 (0.701 vs 0.235) and
> risk/recommendation accuracy are much higher than the base model. Latency is
> CPU-only and not comparable to GPU numbers.
>
> **Unconstrained ablation** (`reports/metrics_*_unconstrained.json`): without
> the grammar, the base model is 0% valid (invented types/formats) and the
> fine-tuned model is 64% valid (36 schema-invalid evidence objects). The
> grammar removes that failure mode, so the same model's content is scored
> instead of being replaced by the UNKNOWN fallback.

> Earlier notebook 06/08 runs recorded 100/100 failures and all-zero metrics.
> Root cause: a bad `extract_json` import in the notebook-only evaluation path.
> The harness and notebooks now use `run_service_eval`, which uses the production
> analysis path and the canonical chat prompt.
