"""LLM access for the agents: one call returns a validated Pydantic object (structured output).

Ollama is called over its HTTP API with the response's JSON schema as `format`, so the model is
constrained to valid JSON of the right shape. Bad JSON and timeouts are retried; after that the error is
raised to the agent, which records it and degrades (e.g. marks fields "not found") instead of crashing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from agents.tracing import Step
from settings import section

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    pass


class LLM(Protocol):
    model: str

    def structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        step: Step | None = None,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
    ) -> T: ...


class OllamaLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
        max_attempts: int = 2,
    ):
        cfg = section("agents")
        self.model = model or os.getenv("LLM_MODEL", "qwen3:4b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = float(cfg.get("temperature", 0) if temperature is None else temperature)
        self.timeout_s = float(timeout_s or cfg.get("llm_timeout_s", 600))
        self.num_ctx = int(cfg.get("num_ctx", 8192))
        self.keep_alive = str(cfg.get("keep_alive", "30m"))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 1500))
        self.max_attempts = max_attempts

    def check(self) -> None:
        """Fail fast with setup instructions if Ollama is not running or the model is not pulled."""
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=10).json()
        except requests.RequestException as exc:
            raise LLMError(
                f"Ollama is not reachable at {self.base_url}. Start the Ollama app (or `ollama serve`)."
            ) from exc
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMError(f"Model {self.model} is not pulled. Run: ollama pull {self.model}")

    def preload(self) -> None:
        """Load the model into memory now, so the first question does not wait for it (best effort)."""
        try:
            requests.post(
                f"{self.base_url}/api/generate", json={"model": self.model, "keep_alive": self.keep_alive}, timeout=120
            )
        except requests.RequestException as exc:
            log.warning("could not preload %s: %s", self.model, exc)

    def structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        step: Step | None = None,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """timeout_s: a hard limit for this call (Q&A answer budget); a timeout is then raised, not retried.
        max_tokens: output cap for this call; every call is capped (agents.max_output_tokens), because a small
        model can loop inside a JSON string and generate for many minutes."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,  # qwen3 reasoning traces cost minutes on CPU and are not needed here
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens or self.max_output_tokens,
            },
            "keep_alive": self.keep_alive,  # keep the model and its prompt cache loaded between questions
        }
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout_s or self.timeout_s)
                resp.raise_for_status()
                body = resp.json()
            except requests.Timeout as exc:
                last_error = f"timeout after {timeout_s or self.timeout_s:.0f}s"
                if timeout_s is not None:
                    raise LLMTimeout(last_error) from exc
                continue
            except requests.RequestException as exc:
                raise LLMError(f"Ollama request failed: {exc}") from exc
            finally:
                if step is not None:
                    step.tracer.count_llm_call()
            if step is not None:
                step.usage(body.get("prompt_eval_count", 0), body.get("eval_count", 0))
                step.tool(
                    "llm",
                    {"model": self.model, "attempt": attempt, "schema": schema.__name__},
                    {"prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count")},
                    time.perf_counter() - started,
                )
            content = body.get("message", {}).get("content", "")
            try:
                return schema.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"bad JSON: {exc}"
                log.warning("LLM returned invalid %s (attempt %d): %s", schema.__name__, attempt, exc)
                # Repeat the request with the error, so the model can correct itself.
                payload["messages"] = [
                    *payload["messages"][:2],
                    {"role": "assistant", "content": content[:4000]},
                    {"role": "user", "content": f"That was not valid for the schema ({exc}). Return corrected JSON."},
                ]
        raise LLMError(f"{schema.__name__}: {last_error}")
