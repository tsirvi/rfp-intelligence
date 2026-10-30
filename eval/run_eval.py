"""Retrieval evaluation: Recall@k and MRR for several search configurations.

python -m eval.run_eval                     # all configurations, writes eval/results.md and results.json
python -m eval.run_eval --configs bm25 full
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from schemas.chunks import Chunk
from search.hybrid import HybridSearcher
from search.reranker import FastReranker
from search.store import IndexStore

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5)
DEPTH = 10  # results retrieved per question; MRR counts ranks up to this depth

# Each configuration adds one stage to the previous one, so the table shows what each stage contributes.
_OFF = {"dense": False, "bm25": False, "expand": False, "dedupe": False, "rerank": False}
CONFIGS: dict[str, dict[str, bool]] = {
    "bm25": {**_OFF, "bm25": True},
    "dense": {**_OFF, "dense": True},
    "hybrid": {**_OFF, "dense": True, "bm25": True},
    "+expand": {**_OFF, "dense": True, "bm25": True, "expand": True},
    "+dedupe (default)": {"dense": True, "bm25": True, "expand": True, "dedupe": True, "rerank": False},
    "+rerank": {"dense": True, "bm25": True, "expand": True, "dedupe": True, "rerank": True},
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", " ")).strip().lower()


def is_match(chunk: Chunk, passage: dict[str, str]) -> bool:
    return passage["file"].lower() in chunk.file_name.lower() and normalise(passage["contains"]) in normalise(
        chunk.text
    )


def load_questions(path: Path = HERE / "questions.yaml") -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text("utf-8"))


def check_ground_truth(store: IndexStore, questions: list[dict[str, Any]]) -> list[str]:
    """Every expected passage must exist in the index, or the metrics would silently undercount."""
    chunks = store.chunks()
    return [
        f"{q['id']}: no chunk matches {p}"
        for q in questions
        for p in q["relevant"]
        if not any(c.bid_id == q["bid"] and is_match(c, p) for c in chunks)
    ]


def score_question(ranked: list[Chunk], relevant: list[dict[str, str]]) -> dict[str, Any]:
    # Rank (1-based) at which each expected passage is first found, or None.
    found = [next((i for i, c in enumerate(ranked, 1) if is_match(c, p)), None) for p in relevant]
    first = min((r for r in found if r), default=None)
    out: dict[str, Any] = {"ranks": found, "rr": 1.0 / first if first else 0.0}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in found if r and r <= k) / len(relevant)
    return out


def evaluate(searcher: HybridSearcher, questions: list[dict[str, Any]], config: dict[str, bool]) -> dict[str, Any]:
    per_question, latencies = {}, []
    for q in questions:
        started = time.perf_counter()
        result = searcher.search(q["question"], filters={"bid_id": q["bid"]}, top_k=DEPTH, **config)
        latencies.append(time.perf_counter() - started)
        per_question[q["id"]] = score_question([h.chunk for h in result.hits], q["relevant"])
    n = len(questions)
    summary = {f"recall@{k}": sum(r[f"recall@{k}"] for r in per_question.values()) / n for k in KS}
    summary["mrr"] = sum(r["rr"] for r in per_question.values()) / n
    summary["avg_latency_s"] = sum(latencies) / n
    return {"summary": summary, "questions": per_question}


def to_markdown(results: dict[str, dict[str, Any]], questions: list[dict[str, Any]]) -> str:
    cols = [f"recall@{k}" for k in KS] + ["mrr", "avg_latency_s"]
    lines = [
        f"# Retrieval evaluation\n\n{len(questions)} questions over {len({q['bid'] for q in questions})} bids, "
        + "each search filtered to its bid. Latency is per query, with models already loaded.\n",
        "| config | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, res in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{res['summary'][c]:.3f}" for c in cols) + " |")
    lines += ["", "## Misses in the best configuration (first relevant rank > 3)", ""]
    best = max(results, key=lambda n: results[n]["summary"]["mrr"])
    for q in questions:
        r = results[best]["questions"][q["id"]]
        if not r["rr"] or 1 / r["rr"] > 3:
            lines.append(f"- {q['id']} ({q['bid']}): {q['question']} — passage ranks {r['ranks']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    store = IndexStore()
    questions = load_questions()
    if problems := check_ground_truth(store, questions):
        raise SystemExit("Ground truth not found in the index (index the sample bids first):\n" + "\n".join(problems))

    reranker = FastReranker()
    store.embedder.embed_query("warm-up")  # load both models before timing anything
    reranker.score("warm-up", ["warm-up"])
    results = {}
    for name in args.configs:
        # A fresh searcher per configuration, so no configuration benefits from another's cached results.
        results[name] = evaluate(HybridSearcher(store, reranker=reranker), questions, CONFIGS[name])
        s = results[name]["summary"]
        print(f"{name:<15} " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), "utf-8")
    (args.out / "results.md").write_text(to_markdown(results, questions), "utf-8")
    print(f"\nwrote {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
