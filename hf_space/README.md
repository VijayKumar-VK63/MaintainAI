---
title: MaintainAI Demo
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
---

# MaintainAI - Fine-Tuned SLM Demo

Static browser demo for the **MaintainAI** fine-tuned Qwen2.5-3B + LoRA model
for predictive maintenance.

## Model
- **Base**: Qwen2.5-3B-Instruct
- **Method**: QLoRA (4-bit NF4, r=16, α=32)
- **Dataset**: NASA CMAPSS FD001 (grounded instruction tuning)
- **Adapter**: `Vijay-kumar-63/maintainai-qwen2.5-3b-lora`

## Usage
1. Set machine context (cycle, health state, failure probability, RUL)
2. Adjust key sensor values (s3, s4, s9, s11, s12)
3. Click **Analyze** for structured maintenance interpretation
4. Chat with the assistant for follow-up questions

## API
`index.html` is a static browser demo that calls the public Hugging Face
Inference API. Static Spaces cannot keep secrets, so the model must be publicly
callable. For authenticated or production inference, use the Streamlit app or
the FastAPI server with `SLM_BACKEND=openai` and a server-side endpoint.
