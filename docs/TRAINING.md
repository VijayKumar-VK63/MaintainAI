# Training — QLoRA on Qwen2.5-3B-Instruct

- **Method**: 4-bit NF4 base (`Qwen/Qwen2.5-3B-Instruct`) + LoRA adapters
  (r=16, alpha=32, dropout 0.05, targets q/k/v/o + gate/up/down), PEFT + TRL
  SFTTrainer. Full 3B fine-tuning is not attempted (resource audit).
- **Config**: `configs/training.yaml` (lr 2e-4, 3 epochs, batch 1, grad-accum 4,
  max_seq 1024, fp16 on T4, seed 42).
- **Format**: training examples store canonical chat `messages`
  (`system`, `user`, `assistant`). `src/train_slm.py` applies the tokenizer
  chat template — the same messages used by inference/evaluation.
- **Data**: `data/slm/train.jsonl` (640) / `val.jsonl` (140); `test.jsonl` (100)
  is NEVER trained on. Token audit with the Qwen tokenizer: max 639 tokens,
  p50 608 — below `max_seq_length=1024`, so no training examples are truncated.
- **Checkpoints**: best model selected on validation loss; adapter saved to
  `output_dir/final`; `train_metrics.json` records train/eval loss, step,
  runtime, and counts.
- **Launch**:
  ```bash
  python -m src.train_slm --config configs/training.yaml
  # smoke: python -m src.train_slm --config configs/training.yaml --smoke
  ```
- **Existing artifact**: `slm/checkpoints/final` is from a 1-epoch / 160-step run
  (train loss ~0.464, eval loss ~0.4647). Rerun notebook 07 for the 3-epoch
  adapter before publishing evaluation numbers.
- **Publish**: upload the LoRA adapter (only) to the HF Hub. Do not commit
  checkpoints to git; `slm/` is gitignored.
