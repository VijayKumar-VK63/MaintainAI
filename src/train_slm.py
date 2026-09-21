"""Config-driven QLoRA fine-tuning (Colab GPU target; CPU smoke-testable).

Real run (Colab, T4):
    python -m src.train_slm --config configs/training.yaml
Checkpoints + final adapter -> output_dir (Drive), then optionally Hub push.

The script audits resources FIRST and refuses full 3B training without a GPU
unless --allow-cpu is passed (reserved for tiny-model smoke tests).
Training data: data/slm/train.jsonl + val.jsonl (grounded Phase-4 examples).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_config


def audit_resources() -> dict:
    import platform
    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "missing"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(props.total_memory / 1e9, 2)
    else:
        info["gpu"] = None
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        info["ram_gb"] = "unknown"
    return info


def format_text(system: str, user: str, target: str, tokenizer=None) -> str:
    """Single canonical training format. ChatML when the tokenizer supports it."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": target}],
                tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return (f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{target}")


def load_examples(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


def train(config_path: str, smoke: bool = False, allow_cpu: bool = False) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = load_config(config_path)
    slm = cfg.get("slm", {})
    res = audit_resources()
    print("resource audit:", json.dumps(res, indent=1))
    if not res["cuda"] and not allow_cpu:
        raise SystemExit("no GPU detected; refusing 3B training (pass --allow-cpu only for tiny-model smoke tests)")

    model_name = slm.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
    max_len = int(slm.get("max_seq_length", 1024))
    out_dir = slm.get("output_dir", "models/slm/exp_001")
    if smoke:
        out_dir = out_dir.rstrip("/") + "-smoke"

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if res["cuda"]:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", trust_remote_code=False)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)

    lora = LoraConfig(r=int(slm.get("lora_r", 16)), lora_alpha=int(slm.get("lora_alpha", 32)),
                      lora_dropout=float(slm.get("lora_dropout", 0.05)),
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def gen(split: str):
        n = 20 if smoke else None
        for e in load_examples(f"data/slm/{split}.jsonl", limit=n):
            yield {"text": format_text(e["system"], e["user"], e["target"], tok)}

    train_ds = Dataset.from_list(list(gen("train")))
    val_ds = Dataset.from_list(list(gen("val")))

    from trl import SFTConfig, SFTTrainer
    epochs = 1 if smoke else int(slm.get("num_train_epochs", 3))
    args = SFTConfig(dataset_text_field="text", max_seq_length=max_len,
                     per_device_train_batch_size=int(slm.get("per_device_train_batch_size", 1)),
                     gradient_accumulation_steps=int(slm.get("gradient_accumulation_steps", 4)),
                     num_train_epochs=epochs, learning_rate=float(slm.get("learning_rate", 2e-4)),
                     warmup_steps=int(slm.get("warmup_steps", 20)),
                     fp16=not torch.cuda.is_bf16_supported() if res["cuda"] else False,
                     logging_steps=5, eval_steps=50, save_steps=50,
                     eval_strategy="steps", save_strategy="steps",
                     load_best_model_at_end=not smoke,
                     output_dir=out_dir, report_to="none", seed=int(slm.get("seed", 42)))
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=train_ds,
                         eval_dataset=val_ds, args=args)
    stats = trainer.train()
    final = f"{out_dir}/final"
    trainer.save_model(final)
    tok.save_pretrained(final)
    return {"train_loss": float(stats.training_loss), "output_dir": final,
            "audit": res, "smoke": smoke}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    print(json.dumps(train(args.config, args.smoke, args.allow_cpu), indent=1))


if __name__ == "__main__":
    main()
