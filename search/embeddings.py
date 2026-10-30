"""Local embedding model (fastembed / ONNX Runtime; no PyTorch needed)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Protocol

# BGE v1.5 retrieves best when queries (not passages) carry this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def model_cache_dir() -> str:
    # fastembed's default is the system temp folder, which macOS clears; keep models somewhere stable.
    return os.getenv("MODEL_CACHE_DIR") or str(Path.home() / ".cache" / "fastembed")


class Embedder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str | None = None, batch_size: int = 16):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.batch_size = batch_size

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding(self.model_name, cache_dir=model_cache_dir())

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts), batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return next(iter(self._model.embed([prefix + text]))).tolist()
