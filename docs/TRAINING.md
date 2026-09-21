# Training — QLoRA (Colab GPU; UNRUN — record real numbers here when run)

- **Method**: 4-bit NF4 base (`Qwen/Qwen2.5-3B-Instruct`) + LoRA adapters
  (r=16, alpha=32, dropout 0.05, targets q/k/v/o + gate/up/down), PEFT + TRL
  SFTTrainer. Full 3B fine-tuning explicitly NOT attempted (resource audit).
- **Config**: `configs/training.yaml` (lr 2e-4, epochs 3, batch 1, grad-accum 4,
  max_seq 1024, fp16 on T4, seed 42). Smoke test (20 examples, 10 steps) first.
- **Data**: `data/slm/train.jsonl` (640) / val (140); test (100) NEVER trained on.
- **Checkpoints**: Drive `MaintainAI/slm/checkpoints/exp_001/`; select on VAL loss.
- **Persistence**: final adapter → Drive + Hugging Face Hub (LoRA only, ~50–100MB).
- **Status**: full Qwen2.5-3B run not yet executed (no GPU locally; staged in
  `experiments/exp_001/`, launch: `python -m src.train_slm --config configs/training.yaml`).
- **CPU smoke test (EXECUTED 2026-09-20, local)**: SmolLM2-135M + LoRA (r=8) for
  8 steps on 20 real Phase-4 examples — train loss 2.047, adapter saved
  (1.9 MB) to `experiments/exp_001-smoke/adapter/`, reloaded, generation
  attempted. Sample output NOT valid JSON — expected after 8 steps on a 135M
  model; proves the format/train/save/reload loop, teaches nothing. The real
  experiment (3B + QLoRA + 640 examples + Colab T4) is a different regime.
