# exp_001 notes

- Historical run completed: 1 epoch / 160 optimizer steps, train loss ~0.4642,
  eval loss ~0.4647 (best checkpoint-160). Adapter saved at
  `slm/checkpoints/final`.
- `configs/training.yaml` now specifies 3 epochs and the canonical chat-message
  format. Rerun notebook 07 before publishing final metrics.
- Select checkpoint on VAL loss, never train loss.
- Evaluation must use `src.slm_eval.run_service_eval` (same path as production).
- Publish only the adapter to the HF Hub; keep checkpoints out of git.
