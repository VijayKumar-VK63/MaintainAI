"""CPU smoke test: proves the LoRA loop end-to-end on a tiny model.

Trains SmolLM2-135M + LoRA for a handful of steps on REAL Phase-4 examples,
saves the adapter, reloads base + adapter, generates, schema-validates.
This validates formatting/training/persistence — it is NOT the Qwen2.5-3B
QLoRA experiment (that needs Colab; see configs/training.yaml + nb 07).
"""
from __future__ import annotations

import json
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.schemas import SLMAnalysis
from src.train_slm import audit_resources, format_text, load_examples

MODEL = "HuggingFaceTB/SmolLM2-135M"
OUT = "experiments/exp_001-smoke/adapter"
STEPS = 8


class JsonlDataset(TorchDataset):
    def __init__(self, texts, tok, max_len=256):
        self.tok = tok
        self.max_len = max_len
        self.enc = [tok(t, truncation=True, max_length=max_len) for t in texts]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        ids = self.enc[i]["input_ids"]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids)),
                "labels": torch.tensor(ids)}


def main() -> dict:
    print("audit:", json.dumps(audit_resources()))
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    tok.pad_token = tok.eos_token
    exs = load_examples("data/slm/train.jsonl", limit=20)
    texts = [format_text(e["system"], e["user"], e["target"], tok) for e in exs]
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir="experiments/exp_001-smoke/ckpt", per_device_train_batch_size=1,
                             max_steps=STEPS, logging_steps=2, save_steps=STEPS,
                             save_total_limit=1, report_to="none", seed=42)

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids = torch.stack([torch.cat([b["input_ids"], torch.full((maxlen - len(b["input_ids"]),), pad)]) for b in batch])
        mask = (ids != pad).long()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    stats = Trainer(model=model, args=args,
                    train_dataset=JsonlDataset(texts, tok)).train()
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    # Reload base + adapter, generate on a held-out val prompt.
    val = load_examples("data/slm/val.jsonl", limit=1)[0]
    tok2 = AutoTokenizer.from_pretrained(OUT, trust_remote_code=False)
    m2 = AutoModelForCausalLM.from_pretrained(MODEL)
    m2 = PeftModel.from_pretrained(m2, OUT)
    m2.eval()
    prompt_ids = tok2(format_text(val["system"], val["user"], "", tok2),
                      return_tensors="pt", truncation=True, max_length=256)["input_ids"]
    with torch.no_grad():
        gen = m2.generate(prompt_ids, max_new_tokens=64, do_sample=False, pad_token_id=tok2.eos_token_id)
    decoded = tok2.decode(gen[0][len(prompt_ids[0]):])
    print("sample generation:", decoded[:300])
    from src.slm_service import extract_json
    obj = extract_json(decoded)
    valid = False
    if obj is not None:
        try:
            SLMAnalysis.model_validate(obj)
            valid = True
        except Exception:
            pass
    result = {"train_loss": float(stats.training_loss), "adapter": OUT,
              "sample_valid_json": valid, "steps": STEPS, "model": MODEL}
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
