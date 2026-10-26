# llama.cpp setup (CPU inference)

Convert the Qwen2.5-3B base and the MaintainAI LoRA adapter to GGUF, then serve
them with llama.cpp's OpenAI-compatible API. The dashboard uses
`SLM_BACKEND=openai` and `SLM_INFERENCE_PATH=/v1/chat/completions`.

## 1. Build llama.cpp (WSL2 recommended on Windows)

```bash
sudo apt update && sudo apt install -y build-essential cmake curl git python3-pip
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j$(nproc)
```

Binaries are in `build/bin/` (`llama-server`, `llama-quantize`).

## 2. Convert base model

```bash
# Qwen2.5-3B-Instruct downloaded locally (or pass the Hub id if the script supports it)
python convert_hf_to_gguf.py /path/to/Qwen2.5-3B-Instruct \
  --outtype f16 --outfile base-f16.gguf

# Quantize for CPU
./build/bin/llama-quantize base-f16.gguf maintainai-qwen2.5-3b.q4_k_m.gguf Q4_K_M
```

## 3. Convert the LoRA adapter

```bash
python convert_lora_to_gguf.py \
  --base-model-id Qwen/Qwen2.5-3B-Instruct \
  --outtype f16 \
  --outfile adapter.gguf \
  /path/to/MaintainAI/slm/checkpoints/final
```

Run `python convert_lora_to_gguf.py --help` if your llama.cpp revision uses
different flag names.

## 4. Serve

```bash
./build/bin/llama-server \
  -m maintainai-qwen2.5-3b.q4_k_m.gguf \
  --lora adapter.gguf \
  -c 2048 --port 8001 --host 0.0.0.0 --no-webui
```

Windows native alternative: use `llama-server.exe` from the official releases
and pass the same flags.

## 5. Connect MaintainAI

```powershell
# PowerShell
$env:SLM_BACKEND="openai"
$env:SLM_INFERENCE_ENDPOINT="http://localhost:8001"
$env:SLM_INFERENCE_PATH="/v1/chat/completions"
python -m pytest -q
streamlit run app/streamlit_app.py
```

```bash
# bash
export SLM_BACKEND=openai
export SLM_INFERENCE_ENDPOINT=http://localhost:8001
export SLM_INFERENCE_PATH=/v1/chat/completions
```

## Verify

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say OK"}]}'

curl http://localhost:8000/health
```

`/health` should report `"slm_backend": "remote:http://localhost:8001/v1/chat/completions"`.

## Quantization trade-offs

| Quantization | Size | Speed | Quality |
|---|---|---|---|
| Q4_K_M | ~2 GB | Fast | Good |
| Q5_K_M | ~2.5 GB | Medium | Better |
| Q8_0 | ~4 GB | Slower | Best |
| f16 | ~6 GB | Slowest | Original |

Recommended: `Q4_K_M` for CPU demos.
