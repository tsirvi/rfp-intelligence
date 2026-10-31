"""Semantic cache for the Q&A agents: a repeated question (same meaning, same bids) gets the stored answer.

A question matches an earlier one when their embeddings are at least `qa.cache_similarity` apart (cosine; 0.94
by default, measured: rewordings score 0.945-0.995, different questions up to 0.909) and contain the same
numbers ("Addendum 1" never matches "Addendum 2"). Entries expire when the index or a bid's record changes,
so an answer is never served from documents or extractions that have since been replaced.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from search.embeddings import Embedder
from settings import section

_NUMBER = re.compile(r"\d+")


class SemanticCache:
    def __init__(self, path: Path, embedder: Embedder, threshold: float | None = None, limit: int = 500):
        self.path = path
        self.embedder = embedder
        self.threshold = float(threshold if threshold is not None else section("qa").get("cache_similarity", 0.94))
        self.limit = limit
        self._lock = threading.Lock()
        self._entries: list[dict] = json.loads(path.read_text("utf-8")) if path.exists() else []

    def lookup(self, question: str, bids: list[str], state: str) -> tuple[dict, float] | None:
        """The best stored answer for this question, bids and data state, with its similarity; or None."""
        vector = self.embedder.embed_query(question)
        numbers = sorted(_NUMBER.findall(question))
        best, score = None, self.threshold
        with self._lock:
            for e in self._entries:
                if e["bids"] != sorted(bids) or e["state"] != state or e["numbers"] != numbers:
                    continue
                similarity = sum(a * b for a, b in zip(vector, e["vector"], strict=True))
                if similarity >= score:
                    best, score = e, similarity
        return (best["answer"], round(score, 3)) if best else None

    def store(self, question: str, bids: list[str], state: str, answer: dict) -> None:
        entry = {
            "question": question,
            "bids": sorted(bids),
            "state": state,
            "numbers": sorted(_NUMBER.findall(question)),
            "vector": self.embedder.embed_query(question),
            "answer": answer,
        }
        with self._lock:
            self._entries = [*self._entries[-(self.limit - 1) :], entry]  # stale entries never match
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._entries), "utf-8")
