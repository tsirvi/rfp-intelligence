"""Cross-encoder reranking (fastembed / ONNX Runtime)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property
from typing import Protocol

from search.embeddings import model_cache_dir


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FastReranker:
    def __init__(self, model_name: str | None = None, batch_size: int = 8):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.batch_size = batch_size

    @cached_property
    def _model(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(self.model_name, cache_dir=model_cache_dir())

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, list(texts), batch_size=self.batch_size)]
