"""SLMService: config-driven maintenance interpretation.

The UI calls SLMService.analyze(machine_context) — never a model directly.
Backends implement Backend.generate(prompt) -> str, so Qwen2.5-3B + LoRA can be
swapped via config without touching business logic or the UI.

Backends:
  - HFBackend: transformers + 4-bit base + PEFT LoRA adapter (Colab/GPU).
    Loaded lazily; failures degrade to DemoBackend, never crash the caller.
  - DemoBackend: deterministic rule-mirror over the machine context, tagged
    backend='demo-fallback'. Explicitly NOT the fine-tuned SLM — the dashboard
    must banner it. Exists so monitoring + numerical prediction stay usable
    where the SLM cannot run (spec section 32).

analyze() NEVER raises and NEVER returns malformed data as trusted: outputs
are JSON-extracted, schema-validated (SLMAnalysis), retried once, and replaced
by a controlled UNKNOWN fallback (valid=False) when validation fails.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from src.context_builder import render_prompt
from src.knowledge import KnowledgeService
from src.predictive import health_state
from src.schemas import MachineContext, SLMAnalysis
from src.validation import controlled_fallback, validate_slm_output_with_retry

logger = logging.getLogger("maintainai.slm")


class Backend(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class DemoBackend:
    """Graceful-degradation mirror. Tagged demo-fallback; banner in UI."""

    name = "demo-fallback"

    def generate(self, prompt: str) -> str:
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


class HFBackend:
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
        self._pipe = pipeline("text-generation", model=model, tokenizer=tok, **pipe_kwargs)

    def generate(self, prompt: str) -> str:
        if self._pipe is None:
            self._load()
        assert self._pipe is not None
        return str(self._pipe(prompt)[0]["generated_text"])


class RemoteBackend:
    """Remote inference endpoint (vLLM/HF Inference Endpoint/RunPod)."""

    def __init__(self, endpoint: str, model_name: str = "remote", timeout: int = 30):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.name = f"remote:{endpoint}"

    def generate(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.endpoint}/generate",
            json={"prompt": prompt, "max_new_tokens": 512, "temperature": 0.2},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("generated_text", "")


class SLMService:
    def __init__(self, config: dict[str, Any] | None = None,
                 backend: Backend | None = None):
        self.config = config or {}
        slm = self.config.get("slm", {})
        self.model_name: str = slm.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        self.knowledge = KnowledgeService()

        if backend is not None:
            self.backend: Backend = backend
        else:
            endpoint = slm.get("inference_endpoint")
            if endpoint:
                self.backend = RemoteBackend(endpoint, self.model_name)
            else:
                self.backend = DemoBackend()
        self.version = self.backend.name

    def _generate(self, prompt: str) -> str:
        backend = self.backend
        if isinstance(backend, DemoBackend):
            raise TypeError("DemoBackend has no generate(); use analyze_context path")
        return backend.generate(prompt)  # type: ignore[union-attr]

    def _build_enhanced_prompt(self, context: MachineContext) -> str:
        """Build prompt with maintenance knowledge injected."""
        base_prompt = render_prompt(context)
        knowledge_text = self.knowledge.format_for_slm()
        return f"{base_prompt}\n\n{knowledge_text}"

    def analyze(self, context: MachineContext) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if isinstance(self.backend, DemoBackend):
                result = self.backend.analyze_context(context)
                SLMAnalysis.model_validate(result)
                result["meta"] = {"backend": self.backend.name, "valid": True,
                                  "model": self.model_name, "demo": True}
            else:
                prompt = self._build_enhanced_prompt(context)
                result = validate_slm_output_with_retry(
                    generate_fn=self._generate,
                    prompt=prompt,
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
                f"(urgency {a['urgency']}). Demo fallback — not the fine-tuned SLM."
            )
            return {"text": text, "meta": {"backend": self.backend.name, "demo": True,
                                           "latency_ms": (time.perf_counter() - t0) * 1000.0}}
        prompt = (self._build_enhanced_prompt(context) + "\n\nUser question (answer ONLY from the "
                  f"evidence above; say when evidence is insufficient): {question}\nAnswer:")
        try:
            text = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return {"text": "AI analysis temporarily unavailable. Machine monitoring and numerical prediction remain available.",
                    "meta": {"backend": self.backend.name, "valid": False, "reason": type(exc).__name__}}
        return {"text": text, "meta": {"backend": self.backend.name,
                                       "latency_ms": (time.perf_counter() - t0) * 1000.0}}

    @staticmethod
    def health_from_prediction(failure_probability: float) -> str:
        return health_state(failure_probability)
