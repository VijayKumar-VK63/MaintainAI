# llama.cpp setup (CPU inference)

Two ways to run the quantized model:

- **In-process** (`SLM_BACKEND=llamacpp`) — no server; the dashboard and
  `src.slm_eval` load the GGUF directly through `llama-cpp-python`.
- **Server** (`SLM_BACKEND=openai`) — `llama-server` exposes an
  OpenAI-compatible `/v1/chat/completions` endpoint.

The quickest path is the automated helper:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python scripts/prepare_local_slm.py
```

It clones llama.cpp into `models/slm/llama.cpp`, downloads
`models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf`, and converts
`slm/checkpoints/final` to `models/slm/maintainai-qwen2.5-3b-lora.gguf`.
Skip steps with `--skip-download` / `--skip-convert`.

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

To constrain the server too, export the grammar and add `--grammar-file`:

```bash
python -c "from src.slm_grammar import SLM_JSON_GRAMMAR; open('slm_json.gbnf','w',encoding='utf-8').write(SLM_JSON_GRAMMAR)"
./build/bin/llama-server -m maintainai-qwen2.5-3b.q4_k_m.gguf \
  --lora adapter.gguf --grammar-file slm_json.gbnf \
  -c 2048 --port 8001 --host 0.0.0.0 --no-webui
```

## 5. Connect MaintainAI

### In-process (no server)

`configs/development.yaml` already uses `slm.backend: auto` and the paths below,
so `python api/main.py` / `streamlit run app/streamlit_app.py` work once the
GGUF files exist. `.env` is loaded automatically; explicit overrides are:

```powershell
$env:SLM_BACKEND="llamacpp"
$env:SLM_MODEL_PATH="models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
$env:SLM_ADAPTER_PATH="models/slm/maintainai-qwen2.5-3b-lora.gguf"
$env:SLM_LORA_SCALE="1.0"
$env:SLM_CONSTRAINED="true"
streamlit run app/streamlit_app.py
```

```bash
# bash
export SLM_BACKEND=llamacpp
export SLM_MODEL_PATH=models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf
export SLM_ADAPTER_PATH=models/slm/maintainai-qwen2.5-3b-lora.gguf
export SLM_CONSTRAINED=true
```

The native backend preloads `llama_cpp` before pandas/pyarrow/torch to avoid a
Windows DLL conflict; importing `src` triggers this automatically.
`SLM_CONSTRAINED=true` applies the GBNF grammar in-process, so the JSON shape,
enums, and confidence range are guaranteed.

### Server mode (OpenAI-compatible)

```powershell
# PowerShell
$env:SLM_BACKEND="openai"
$env:SLM_INFERENCE_ENDPOINT="http://localhost:8001"
$env:SLM_INFERENCE_PATH="/v1/chat/completions"
streamlit run app/streamlit_app.py
```

```bash
# bash
export SLM_BACKEND=openai
export SLM_INFERENCE_ENDPOINT=http://localhost:8001
export SLM_INFERENCE_PATH=/v1/chat/completions
```

## Verify

In-process smoke test:

```bash
python -m src.slm_eval --backend llamacpp \
  --gguf models/slm/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --adapter models/slm/maintainai-qwen2.5-3b-lora.gguf \
  --limit 3 --n-threads 12 --constrained --out reports/_smoke.json
```

Server mode:

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
