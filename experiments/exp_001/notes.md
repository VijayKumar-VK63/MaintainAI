# exp_001 notes

- 2026-09-20: staged. Awaiting Colab T4 runtime.
- Smoke first (`--smoke`, 20 examples): must show falling loss before full run.
- Select checkpoint on VAL loss, never train loss.
- After run: fill metrics.json with train/val loss, duration, GPU memory, adapter location.
