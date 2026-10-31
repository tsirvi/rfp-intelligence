"""Cost and latency per run: one JSON line per extraction, question, comparison or assessment.

outputs/runs.jsonl is the ledger; `summary()` aggregates it (for GET /usage and the UI). Token prices come from
config.yaml (`costs`); the local model costs nothing to call, so they default to 0 and exist to estimate what
the same run would cost on a hosted model.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from agents.tracing import Tracer
from settings import section

_lock = threading.Lock()


def estimate_cost(prompt_tokens: int, output_tokens: int) -> float:
    prices = section("costs")
    per_in, per_out = float(prices.get("prompt_per_1k", 0)), float(prices.get("output_per_1k", 0))
    return round(prompt_tokens / 1000 * per_in + output_tokens / 1000 * per_out, 6)


def record_run(outputs_dir: Path, kind: str, tracer: Tracer, bids: list[str], seconds: float, **extra: object) -> dict:
    """Append one run to the ledger and return the entry (also stored on the record or answer)."""
    t = tracer.totals
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": tracer.run_id,
        "kind": kind,
        "bids": bids,
        "seconds": round(seconds, 2),
        "llm_calls": t["llm_calls"],
        "prompt_tokens": t["prompt_tokens"],
        "output_tokens": t["output_tokens"],
        "est_cost": estimate_cost(t["prompt_tokens"], t["output_tokens"]),
        "currency": section("costs").get("currency", "USD"),
        "agent_seconds": dict(tracer.agent_seconds),
        **extra,
    }
    with _lock:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        with (outputs_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return entry


def summary(outputs_dir: Path) -> dict:
    """Totals and averages per kind of run, plus the Q&A cache hit rate."""
    path = outputs_dir / "runs.jsonl"
    runs = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()] if path.exists() else []
    by_kind: dict[str, dict] = {}
    for r in runs:
        k = by_kind.setdefault(
            r["kind"],
            {"runs": 0, "seconds": 0.0, "llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "est_cost": 0.0},
        )
        k["runs"] += 1
        for key in ("seconds", "llm_calls", "prompt_tokens", "output_tokens", "est_cost"):
            k[key] += r.get(key, 0)
    for k in by_kind.values():
        k["avg_seconds"] = round(k["seconds"] / k["runs"], 2)
        k["seconds"], k["est_cost"] = round(k["seconds"], 2), round(k["est_cost"], 6)
    asks = [r for r in runs if r["kind"] == "ask"]
    cached = sum(bool(r.get("cached")) for r in asks)
    return {
        "runs": len(runs),
        "by_kind": by_kind,
        "ask_cache_hit_rate": round(cached / len(asks), 3) if asks else None,
        "ask_modes": {m: sum(r.get("mode") == m for r in asks) for m in sorted({r.get("mode") for r in asks} - {None})},
        "currency": section("costs").get("currency", "USD"),
    }
