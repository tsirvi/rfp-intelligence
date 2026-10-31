"""Structured trace of every agent step: input, tool calls, output, tokens and latency, as JSON lines.

One file per run under outputs/traces/<run_id>.jsonl, so a run can be replayed and audited afterwards.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from settings import ROOT

log = logging.getLogger("agents.trace")


def _short(value: Any, limit: int = 2000) -> Any:
    """Keep traces readable: long strings are truncated, models are dumped."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} more chars]"
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_short(v, limit) for v in value]
    return value


class Step:
    """One agent step. Tool calls and LLM usage are attached while it runs."""

    def __init__(self, tracer: Tracer, agent: str, action: str, input: Any):
        self.tracer, self.agent, self.action = tracer, agent, action
        self.record: dict[str, Any] = {"agent": agent, "action": action, "input": _short(input), "tool_calls": []}
        self.prompt_tokens = self.output_tokens = 0

    def tool(self, name: str, args: Any, result_summary: Any, latency_s: float) -> None:
        self.record["tool_calls"].append(
            {"tool": name, "args": _short(args), "result": _short(result_summary), "latency_s": round(latency_s, 3)}
        )

    def usage(self, prompt_tokens: int, output_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens

    def output(self, value: Any) -> None:
        self.record["output"] = _short(value)


class Tracer:
    def __init__(self, run_id: str | None = None, out_dir: Path | None = None, enabled: bool = True):
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.path = (out_dir or ROOT / "outputs" / "traces") / f"{self.run_id}.jsonl"
        self.enabled = enabled
        self._lock = threading.Lock()
        self.totals = {"steps": 0, "llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "errors": 0}
        self.agent_seconds: dict[str, float] = {}  # time spent per agent (parallel steps overlap)

    @contextmanager
    def step(self, agent: str, action: str, input: Any = None) -> Iterator[Step]:
        step = Step(self, agent, action, input)
        started = time.perf_counter()
        error: str | None = None
        try:
            yield step
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            step.record.update(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "run_id": self.run_id,
                    "latency_s": round(time.perf_counter() - started, 3),
                    "prompt_tokens": step.prompt_tokens,
                    "output_tokens": step.output_tokens,
                    "error": error,
                }
            )
            self._write(step.record)

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.totals["steps"] += 1
            self.totals["prompt_tokens"] += record["prompt_tokens"]
            self.totals["output_tokens"] += record["output_tokens"]
            self.totals["errors"] += bool(record["error"])
            agent = record["agent"]
            self.agent_seconds[agent] = round(self.agent_seconds.get(agent, 0.0) + record["latency_s"], 3)
            log.info(
                "step agent=%s action=%s latency=%.1fs tokens=%d/%d error=%s",
                record["agent"],
                record["action"],
                record["latency_s"],
                record["prompt_tokens"],
                record["output_tokens"],
                record["error"],
            )
            if self.enabled:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")

    def count_llm_call(self) -> None:
        with self._lock:
            self.totals["llm_calls"] += 1
