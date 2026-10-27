# MaintainAI

### Predictive maintenance that predicts failures, explains the evidence, and recommends the next action.

MaintainAI is an end-to-end predictive maintenance console. It streams simulated
turbofan telemetry, predicts failure risk and remaining useful life with XGBoost,
validates a structured maintenance assessment with a fine-tuned Qwen2.5-3B + LoRA
model, and answers operator questions grounded in that assessment.

Most maintenance AI stops at a score. MaintainAI connects the full decision:

```text
Telemetry -> causal features -> XGBoost risk + RUL -> validated assessment
          -> grounded assistant -> operator action
```

The result is a product experience where an operator can see the machine state,
understand why it matters, and ask what to do next without the language model
inventing sensor values or overriding the predictive model.

---

## Product Experience

- **Live operations view** - health state, failure probability, RUL, anomaly score,
  sensor trends, and the failure-probability trend against the decision threshold.
- **Validated assessment** - risk level, likely condition, confidence, evidence,
  recommended action, and urgency.
- **Grounded assistant** - clickable maintenance prompts and free-form chat.
- **Guardrails** - exact values for common questions, contradiction detection,
  safe handling of shutdown, guarantee, and prompt-injection questions.
- **Engineer details on demand** - measured model performance and runtime details
  stay out of the operator's way until expanded.

---

## What Makes It Different

### Prediction and explanation are separated

XGBoost owns the numbers. The language model owns the explanation. The assistant
receives the validated assessment as authoritative context and is instructed not
to invent probabilities, thresholds, sensor values, or maintenance history.

### Common questions are deterministic

Questions such as "What is the risk?", "What is the RUL?", and "What should the
operator do next?" are answered directly from the validated assessment. This keeps
the most important answers fast and exact.

### Model answers are checked

Free-form answers are compared against the validated risk, failure probability,
and RUL. Contradictory answers are replaced with the grounded assessment. Truncated
model output is trimmed to a complete sentence.

### The same path runs locally and in the cloud

The default local setup runs the quantized GGUF + LoRA adapter on CPU. Production
deployments can point the same service at any OpenAI-compatible endpoint.

---

## Quickstart

### Prerequisites

- Python 3.9+
- Git
- About 4 GB free RAM for the local 3B Q4 model

### Install

```bash
git clone https://github.com/VijayKumar-VK63/MaintainAI.git
cd MaintainAI
pip install -r requirements.txt
```

### Prepare the local model

```bash
python scripts/prepare_local_slm.py
```

This downloads the base GGUF and converts the LoRA adapter once. The default
config uses `slm.backend: auto`: a configured remote endpoint first, then the
local GGUF, then a local Transformers adapter. There is no silent demo fallback.

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501.

### Run the API (optional)

```bash
python api/main.py
```

- API root: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## Demo Flow

1. Select a machine.
2. Click **Start**.
3. Choose **DEGRADATION** and click **Apply scenario**.
4. Watch the sensor trends and failure probability rise.
5. Read the assessment: risk, condition, evidence, action, urgency.
6. Click a suggested prompt or ask a follow-up question.

The sidebar also supports **CRITICAL** and **FAILURE** scenarios for a fast
high-risk demonstration.

Walkthrough video: [MaintainAI DEMO.mp4](<MaintainAI DEMO.mp4>).

---

## Architecture

```text
src/
  simulation.py       seeded turbofan telemetry from FD001 baselines
  features.py         causal rolling features, no future leakage
  predictive.py       XGBoost RUL regressor + 30-cycle risk classifier
  context_builder.py  validated machine context for the assistant
  prompts.py          canonical training/serving prompts
  chat_templates.py   deterministic grounded answers and guardrails
  slm_service.py      llama.cpp / Transformers / OpenAI-compatible backends
  validation.py       schema validation and controlled fallback
  engine.py           telemetry -> prediction -> assessment -> chat
api/
  main.py             FastAPI service and error contract
app/
  streamlit_app.py    operator console
```

The assistant is a fine-tuned Qwen2.5-3B-Instruct model with a QLoRA adapter:
[Vijay-kumar-63/maintainai-qwen2.5-3b-lora](https://huggingface.co/Vijay-kumar-63/maintainai-qwen2.5-3b-lora).

---

## Measured Results

### Predictive model - held-out NASA FD001 test engines

| Metric | Value |
| --- | ---: |
| RUL RMSE | 16.01 cycles |
| RUL MAE | 11.52 cycles |
| NASA score | 416.7 |
| Risk precision | 0.958 |
| Risk recall | 0.920 |
| Risk F1 | 0.939 |
| Risk ROC-AUC | 0.994 |

### Assistant - same 100 held-out scenarios

| System | Risk accuracy | Condition accuracy | Evidence F1 | Validity |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B base (Q4_K_M, CPU) | 0.650 | 0.590 | 0.000 | 0.800 |
| Qwen2.5-3B + LoRA (Q4_K_M, CPU) | 0.980 | 0.730 | 0.701 | 1.000 |

The local adapter is the measured 1-epoch checkpoint. The fine-tuned model is
100% schema-valid and improves every content metric over the base model. See
[docs/EVALUATION.md](docs/EVALUATION.md) for the full protocol and ablations.

---

## Configuration

Copy `.env.example` to `.env` for local development. The most important settings:

| Variable | Purpose |
| --- | --- |
| `SLM_BACKEND` | `auto`, `llamacpp`, `transformers`, or `openai` (`demo` is test-only) |
| `SLM_MODEL_PATH` | Local base GGUF |
| `SLM_ADAPTER_PATH` | Local GGUF LoRA adapter |
| `SLM_INFERENCE_ENDPOINT` | Remote OpenAI-compatible endpoint |
| `SLM_API_KEY` | Bearer token for the remote endpoint |
| `SLM_CHAT_MAX_NEW_TOKENS` | Free-form chat generation limit (default 96) |
| `SLM_CHAT_TEMPERATURE` | Free-form chat temperature (default 0.0) |

For Streamlit Cloud, the dashboard cannot run the local 1.8 GB GGUF. Set
`SLM_BACKEND=openai` and `SLM_INFERENCE_ENDPOINT` in Streamlit secrets. See
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Testing

```bash
pytest -q
python -m ruff check src app api tests conftest.py
```

The fast test suite uses `configs/test.yaml` and the demo backend, so it does not
download or load the 3B model.

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation](docs/EVALUATION.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Local GGUF setup](docs/GGUF_SETUP.md)
- [Training](docs/TRAINING.md)
- [Dataset](docs/DATASET.md)

---

## Safety and Limitations

MaintainAI uses simulated telemetry and is a decision-support demonstration, not
safety software. It does not include inspection history, maintenance records, or
future operating changes. Qualified personnel make all maintenance and shutdown
decisions.

---

## License

MIT. See [LICENSE](LICENSE).

NASA CMAPSS FD001 is used as the predictive and simulation dataset.
