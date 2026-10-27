"""SLMService: config-driven maintenance interpretation.

The UI calls SLMService.analyze(machine_context) — never a model directly.
Backends implement Backend.generate(messages) -> str, so the model can be
swapped via config without touching business logic or the UI.

Backends:
  - DemoBackend: deterministic rule-mirror over the machine context, tagged
    backend='demo-fallback'. Explicitly NOT the fine-tuned SLM and only used
    when `SLM_BACKEND=demo` is set on purpose (tests / offline demos).
  - TransformersBackend: transformers + 4-bit base + PEFT LoRA adapter
    (Colab/GPU/local). Applies the tokenizer chat template.
  - LlamaCppBackend: CPU-only, in-process quantized GGUF base + optional
    GGUF LoRA through llama-cpp-python. No GPU or server required.
  - OpenAICompatibleBackend: llama.cpp / vLLM / TGI / OpenAI-compatible HTTP,
    with optional bearer auth.

analyze() NEVER raises and NEVER returns malformed data as trusted: outputs
are JSON-extracted, schema-validated (SLMAnalysis), retried once, and replaced
by a controlled UNKNOWN fallback (valid=False) when validation fails.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from src.context_builder import render_prompt
from src.knowledge import KnowledgeService
from src.predictive import health_state
from src.prompts import build_messages, render_chat
from src.schemas import MachineContext, SLMAnalysis
from src.slm_grammar import SLM_JSON_GRAMMAR
from src.validation import (
    controlled_fallback,
    extract_json,
    validate_slm_output_with_retry,
)

logger = logging.getLogger("maintainai.slm")

Message = dict[str, str]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class Backend(Protocol):
    name: str

    def generate(self, messages: list[Message] | str) -> str: ...


class DemoBackend:
    """Graceful-degradation mirror. Tagged demo-fallback; banner in UI."""

    name = "demo-fallback"

    def generate(self, messages: list[Message] | str) -> str:
        raise NotImplementedError("DemoBackend is invoked via analyze_context, not generate")

    def analyze_context(self, context: MachineContext) -> dict[str, Any]:
        p = context.prediction
        risk = {"HEALTHY": "LOW", "WARNING": "MEDIUM", "HIGH_RISK": "HIGH", "CRITICAL": "CRITICAL"}[p.health_state]
        rising = sorted(k.replace("_rel_change", "") for k, v in context.trends.items() if v > 0.01)
        evidence = [f"{s}_above_baseline" for s in rising if s in ("s3", "s4", "s9", "s11", "s12")]
        if len(rising) >= 2:
            evidence.append("trend_increasing_multiple_sensors")
        if (p.rul_cycles or 999) <= 30:
            evidence.append("rul_low")
        if p.failure_probability >= 0.65:
            evidence.append("failure_probability_high")
        if not evidence:
            evidence = ["insufficient_evidence"]
        condition = "healthy_operation" if risk == "LOW" and not rising else ("hpc_degradation" if len(rising) >= 2 else "unknown_anomaly")
        action, urgency = {
            "LOW": ("continue_monitoring", "ROUTINE"),
            "MEDIUM": ("schedule_hpc_inspection", "PRIORITY"),
            "HIGH": ("schedule_hpc_inspection", "PRIORITY"),
            "CRITICAL": ("escalate_immediate_review", "IMMEDIATE"),
        }[risk]
        return {
            "risk_level": risk, "likely_condition": condition,
            "confidence": 0.6 if len(evidence) >= 2 else 0.35,
            "evidence": evidence, "recommended_action": action, "urgency": urgency,
        }


class TransformersBackend:
    """Lazy transformers backend: base repo + optional LoRA adapter, 4-bit."""

    def __init__(self, model_name: str, quantization: str = "4bit",
                 adapter_path: str | None = None, max_new_tokens: int = 512,
                 temperature: float = 0.2):
        self.model_name = model_name
        self.quantization = quantization
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.name = f"hf:{model_name}" + (f"+{adapter_path}" if adapter_path else "")
        self._pipe = None
        self._tok = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("4-bit quantization needs bitsandbytes (Colab notebook 07 installs it)") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=False, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
        pipe_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens,
                                          "do_sample": self.temperature > 0,
                                          "return_full_text": False}
        if self.temperature > 0:
            pipe_kwargs["temperature"] = self.temperature
        self._tok = tok
        self._pipe = pipeline("text-generation", model=model, tokenizer=tok, **pipe_kwargs)

    def generate(self, messages: list[Message] | str) -> str:
        if self._pipe is None:
            self._load()
        assert self._pipe is not None
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = render_chat(self._tok, messages, add_generation_prompt=True)
        return str(self._pipe(prompt)[0]["generated_text"])


# Backwards-compatible name used by notebooks and older imports.
HFBackend = TransformersBackend


class LlamaCppBackend:
    """Native llama.cpp backend: quantized GGUF base + optional GGUF LoRA.

    CPU-friendly alternative to the 4-bit transformers backend, which needs
    bitsandbytes + a CUDA GPU. Loads lazily, so importing this module never
    requires llama-cpp-python.
    """

    def __init__(self, model_path: str, adapter_path: str | None = None,
                 n_ctx: int = 2048, n_threads: int | None = None,
                 n_batch: int = 512, max_new_tokens: int = 512,
                 temperature: float = 0.2, seed: int = 42,
                 lora_scale: float = 1.0, constrained: bool = False,
                 grammar: str | None = None, verbose: bool = False):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.lora_scale = lora_scale
        self.constrained = constrained
        self.grammar = grammar
        self.verbose = verbose
        name = f"llamacpp:{Path(model_path).name}"
        if adapter_path:
            name += f"+{Path(adapter_path).name}"
        self.name = name
        self._llm = None
        self._grammar = None
        self._lock = threading.Lock()

    def _load(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the CPU wheel with: "
                "pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ) from exc
        if self.constrained or self.grammar:
            from llama_cpp import LlamaGrammar
            self._grammar = LlamaGrammar.from_string(
                self.grammar or SLM_JSON_GRAMMAR, verbose=self.verbose
            )
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "seed": self.seed,
            "verbose": self.verbose,
        }
        if self.n_threads:
            kwargs["n_threads"] = self.n_threads
        if self.adapter_path:
            kwargs["lora_path"] = self.adapter_path
            kwargs["lora_scale"] = self.lora_scale
        from llama_cpp._utils import suppress_stdout_stderr

        with suppress_stdout_stderr(disable=self.verbose):
            self._llm = Llama(**kwargs)

    def _complete(self, messages: list[Message] | str, grammar: Any = None) -> str:
        # llama.cpp contexts are stateful and not thread-safe. Streamlit can
        # serve multiple sessions and FastAPI runs sync handlers in threads.
        with self._lock:
            if self._llm is None:
                self._load()
            assert self._llm is not None
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            out = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                repeat_penalty=1.05,
                seed=self.seed,
                grammar=grammar,
            )
            return str(out["choices"][0]["message"]["content"])

    def generate(self, messages: list[Message] | str) -> str:
        return self._complete(messages, self._grammar)

    def generate_free(self, messages: list[Message] | str) -> str:
        return self._complete(messages, None)

    def close(self) -> None:
        with self._lock:
            if self._llm is not None:
                self._llm.close()
                self._llm = None


class OpenAICompatibleBackend:
    """HTTP backend for llama.cpp / vLLM / TGI / OpenAI-style endpoints."""

    def __init__(self, endpoint: str, model_name: str = "remote",
                 timeout: int = 60, api_key: str | None = None,
                 inference_path: str = "/v1/chat/completions",
                 max_new_tokens: int = 512, temperature: float = 0.2):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key
        self.inference_path = inference_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.name = f"remote:{self.url}"

    @property
    def url(self) -> str:
        if self.inference_path.startswith("http"):
            return self.inference_path
        return f"{self.endpoint}/{self.inference_path.lstrip('/')}"

    def generate(self, messages: list[Message] | str) -> str:
        import requests

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        parameters = {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature}
        if self.inference_path.endswith("/generate"):
            payload = {
                "inputs": render_chat(None, messages, add_generation_prompt=True),
                "parameters": parameters,
            }
        else:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }

        resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            if data.get("choices"):
                return str(data["choices"][0]["message"]["content"])
            if data.get("generated_text"):
                return str(data["generated_text"])
        return str(data)


class RemoteBackend(OpenAICompatibleBackend):
    """Backwards-compatible remote backend (OpenAI-compatible by default)."""


class SLMService:
    def __init__(self, config: dict[str, Any] | None = None,
                 backend: Backend | None = None):
        self.config = config or {}
        slm = self.config.get("slm", {})
        self.model_name: str = slm.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        self.knowledge = KnowledgeService()
        self.include_knowledge = bool(slm.get("include_knowledge", False))

        if backend is not None:
            self.backend: Backend = backend
        else:
            self.backend = self._build_backend(slm)
        self.version = self.backend.name
        logger.debug("slm backend selected: %s", self.version)

    def _build_backend(self, slm: dict[str, Any]) -> Backend:
        kind = str(slm.get("backend") or os.getenv("SLM_BACKEND") or "auto").strip().lower()
        endpoint = slm.get("inference_endpoint") or os.getenv("SLM_INFERENCE_ENDPOINT")
        adapter = slm.get("adapter_path") or os.getenv("SLM_ADAPTER_PATH")
        model_path = (
            slm.get("model_path")
            or slm.get("gguf_path")
            or os.getenv("SLM_MODEL_PATH")
        )
        api_key = slm.get("api_key") or os.getenv("SLM_API_KEY") or os.getenv("HF_TOKEN")
        constrained = _as_bool(
            slm.get("constrained"),
            _as_bool(os.getenv("SLM_CONSTRAINED"), False),
        )

        if kind in ("", "auto"):
            if endpoint:
                kind = "openai"
            elif model_path and Path(str(model_path)).exists():
                kind = "llamacpp"
            elif adapter and Path(str(adapter)).exists():
                kind = "transformers"
            else:
                raise RuntimeError(
                    "No SLM backend is configured. Set SLM_MODEL_PATH to a local "
                    "GGUF (run scripts/prepare_local_slm.py) or set "
                    "SLM_INFERENCE_ENDPOINT to a remote OpenAI-compatible endpoint. "
                    "The demo rule mirror is only available with an explicit "
                    "SLM_BACKEND=demo."
                )

        if kind in ("llamacpp", "llama.cpp", "llama-cpp", "gguf"):
            if not model_path:
                raise RuntimeError(
                    "llamacpp backend selected but SLM_MODEL_PATH/model_path is not set"
                )
            threads = slm.get("n_threads")
            lora_scale = slm.get("lora_scale")
            if lora_scale is None:
                lora_scale = os.getenv("SLM_LORA_SCALE", 1.0)
            return LlamaCppBackend(
                str(model_path),
                adapter_path=adapter,
                n_ctx=int(slm.get("n_ctx", 2048)),
                n_threads=int(threads) if threads else None,
                n_batch=int(slm.get("n_batch", 512)),
                max_new_tokens=int(slm.get("max_new_tokens", 256)),
                temperature=float(slm.get("temperature", 0.2)),
                seed=int(slm.get("seed", 42)),
                lora_scale=float(lora_scale),
                constrained=constrained,
            )

        if kind in ("transformers", "hf", "local"):
            return TransformersBackend(
                self.model_name,
                quantization=slm.get("quantization", "4bit"),
                adapter_path=adapter,
                max_new_tokens=int(slm.get("max_new_tokens", 256)),
                temperature=float(slm.get("temperature", 0.2)),
            )

        if kind in ("openai", "remote", "vllm", "tgi"):
            if not endpoint:
                raise RuntimeError(
                    "openai backend selected but SLM_INFERENCE_ENDPOINT is not set"
                )
            return OpenAICompatibleBackend(
                endpoint,
                model_name=self.model_name,
                api_key=api_key,
                inference_path=slm.get("inference_path", "/v1/chat/completions"),
                max_new_tokens=int(slm.get("max_new_tokens", 256)),
                temperature=float(slm.get("temperature", 0.2)),
            )

        if kind == "demo":
            return DemoBackend()

        raise RuntimeError(f"Unknown SLM backend: {kind!r}")

    def _messages(self, context: MachineContext) -> list[Message]:
        messages = build_messages(context)
        if self.include_knowledge:
            messages[0]["content"] += "\n\n" + self.knowledge.format_for_slm()
        return messages

    def _generate(self, request: list[Message] | str) -> str:
        backend = self.backend
        if isinstance(backend, DemoBackend):
            raise TypeError("DemoBackend has no generate(); use analyze_context path")
        return backend.generate(request)  # type: ignore[union-attr]

    def analyze(self, context: MachineContext) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if isinstance(self.backend, DemoBackend):
                result = self.backend.analyze_context(context)
                SLMAnalysis.model_validate(result)
                result["meta"] = {"backend": self.backend.name, "valid": True,
                                  "model": self.model_name, "demo": True}
            else:
                messages = self._messages(context)
                result = validate_slm_output_with_retry(
                    generate_fn=self._generate,
                    prompt=messages,
                    model_name=self.model_name,
                    max_retries=1,
                )
        except Exception as exc:  # noqa: BLE001 — absolute last resort
            logger.exception("slm analyze unexpected failure")
            fb = controlled_fallback(f"unexpected: {type(exc).__name__}")
            fb["meta"].update({"model": self.model_name,
                               "latency_ms": (time.perf_counter() - t0) * 1000.0})
            return fb
        result["meta"]["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    def chat(self, question: str, context: MachineContext) -> dict[str, Any]:
        """Grounded assistant turn. Free text, but evidence-scoped prompt."""
        question = question[:1000]
        t0 = time.perf_counter()
        if isinstance(self.backend, DemoBackend):
            a = self.analyze(context)
            text = (
                f"Machine {context.machine_id} is at {a['risk_level']} risk "
                f"(likely condition: {a['likely_condition']}, "
                f"failure probability {context.prediction.failure_probability:.0%}, "
                f"RUL ~{context.prediction.rul_cycles} cycles). "
                f"Evidence: {', '.join(a['evidence'])}. "
                f"Recommended action: {a['recommended_action']} "
                f"(urgency {a['urgency']}). Demo rule mirror - not the fine-tuned SLM."
            )
            return {"text": text, "meta": {"backend": self.backend.name, "demo": True,
                                           "latency_ms": (time.perf_counter() - t0) * 1000.0}}
        messages = self._messages(context)
        messages[-1]["content"] += (
            "\n\nAnswer in 1-3 plain-text sentences (not JSON). Answer ONLY from "
            f"the evidence above; say when evidence is insufficient.\nQuestion: {question}"
        )
        try:
            free = getattr(self.backend, "generate_free", None)
            text = free(messages) if callable(free) else self._generate(messages)
        except Exception as exc:  # noqa: BLE001
            return {"text": "AI analysis temporarily unavailable. Machine monitoring and numerical prediction remain available.",
                    "meta": {"backend": self.backend.name, "valid": False, "reason": type(exc).__name__}}
        return {"text": text, "meta": {"backend": self.backend.name,
                                       "latency_ms": (time.perf_counter() - t0) * 1000.0}}

    @staticmethod
    def health_from_prediction(failure_probability: float) -> str:
        return health_state(failure_probability)


# Kept for callers that imported the legacy flat prompt helper from this module.
__all__ = [
    "Backend",
    "DemoBackend",
    "TransformersBackend",
    "HFBackend",
    "LlamaCppBackend",
    "OpenAICompatibleBackend",
    "RemoteBackend",
    "SLMService",
    "extract_json",
    "render_prompt",
]
