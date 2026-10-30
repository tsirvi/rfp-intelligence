"""Hybrid retrieval: dense + BM25 over every query variant, fused with Reciprocal Rank Fusion, then
reranked with a cross-encoder. Each stage can be switched off, which is how the evaluation compares
configurations.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from schemas.chunks import Chunk
from search.expansion import expand_query
from search.reranker import FastReranker, Reranker
from search.store import Filters, IndexStore, tokenize
from settings import section

log = logging.getLogger(__name__)
_CACHE_LIMIT = 10_000


class SearchConfig(BaseModel):
    dense: bool = True
    bm25: bool = True
    expand: bool = True
    rerank: bool = False  # off by default: see eval/results.md for the cost/benefit on CPU
    dedupe: bool = True
    dedupe_threshold: float = 0.5  # token Jaccard above which two hits from one document count as one
    top_k_dense: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 15
    top_k: int = 6  # results returned
    max_variants: int = 3
    variant_weight: float = 1.0  # RRF weight of expanded variants; 0.25 and 0.5 tested, no clear gain

    @classmethod
    def from_settings(cls, **overrides: Any) -> SearchConfig:
        cfg = section("retrieval")
        if "top_k_rerank" in cfg:
            cfg["top_k"] = cfg.pop("top_k_rerank")
        return cls(**{**cfg, **overrides})


class SearchHit(BaseModel):
    chunk: Chunk
    score: float  # rerank score if reranked, else RRF score
    rank: int
    dense_rank: int | None = None  # best rank across query variants, 1-based
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return self.chunk.citation


class SearchResult(BaseModel):
    query: str
    variants: list[str]
    filters: Filters = Field(default_factory=dict)
    hits: list[SearchHit]
    timings: dict[str, float] = Field(default_factory=dict)


def rrf_fuse(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of weight / (k + rank)."""
    scores: dict[str, float] = {}
    for i, ranking in enumerate(rankings):
        weight = weights[i] if weights else 1.0
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return scores


class HybridSearcher:
    def __init__(
        self,
        store: IndexStore,
        reranker: Reranker | None = None,
        config: SearchConfig | None = None,
        expander: Callable[[str], list[str]] | None = None,
    ):
        self.store = store
        self._reranker = reranker
        self.config = config or SearchConfig.from_settings()
        self.expander = expander
        # Caches keyed by index version, so repeated queries (evaluation, agents retrying) are cheap.
        self._retrieval_cache: dict[tuple, list[tuple[Chunk, float]]] = {}
        self._rerank_cache: dict[tuple, float] = {}

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = FastReranker()
        return self._reranker

    def _retrieve(self, mode: str, query: str, k: int, filters: Filters | None) -> list[tuple[Chunk, float]]:
        key = (mode, query, k, json.dumps(filters or {}, sort_keys=True, default=str), self.store.version())
        if key not in self._retrieval_cache:
            if len(self._retrieval_cache) > _CACHE_LIMIT:
                self._retrieval_cache.clear()
            fn = self.store.dense_search if mode == "dense" else self.store.bm25_search
            self._retrieval_cache[key] = fn(query, k, filters)
        return self._retrieval_cache[key]

    def _rerank_scores(self, query: str, chunks: list[Chunk]) -> list[float]:
        version = self.store.version()
        missing = [c for c in chunks if (query, c.chunk_id, version) not in self._rerank_cache]
        if len(self._rerank_cache) > _CACHE_LIMIT:
            self._rerank_cache.clear()
        if missing:
            for c, s in zip(missing, self.reranker.score(query, [c.indexed_text for c in missing]), strict=True):
                self._rerank_cache[(query, c.chunk_id, version)] = s
        return [self._rerank_cache[(query, c.chunk_id, version)] for c in chunks]

    @staticmethod
    def _dedupe(ordered: list[str], chunks: dict[str, Chunk], cfg: SearchConfig) -> list[str]:
        """Drop a hit that mostly repeats a higher-ranked hit from the same document, such as a table
        and the page text it was extracted from, so the results cover more distinct passages."""
        if not cfg.dedupe:
            return ordered
        kept: list[tuple[str, str, set[str]]] = []
        for cid in ordered:
            chunk = chunks[cid]
            tokens = set(tokenize(chunk.text))
            if any(
                doc_id == chunk.doc_id and len(tokens & other) / max(1, len(tokens | other)) >= cfg.dedupe_threshold
                for _, doc_id, other in kept
            ):
                continue
            kept.append((cid, chunk.doc_id, tokens))
        return [cid for cid, _, _ in kept]

    def search(
        self, query: str, filters: Filters | None = None, top_k: int | None = None, **overrides: Any
    ) -> SearchResult:
        cfg = self.config.model_copy(update=overrides) if overrides else self.config
        top_k = top_k or cfg.top_k
        timings: dict[str, float] = {}
        started = time.perf_counter()

        if cfg.expand:
            variants = (self.expander or (lambda q: expand_query(q, cfg.max_variants)))(query)
            variants = list(dict.fromkeys([query, *variants]))
        else:
            variants = [query]

        chunks: dict[str, Chunk] = {}
        rankings: list[list[str]] = []
        weights: list[float] = []
        best_rank: dict[str, dict[str, int]] = {"dense": {}, "bm25": {}}
        for mode, enabled, k in (("dense", cfg.dense, cfg.top_k_dense), ("bm25", cfg.bm25, cfg.top_k_bm25)):
            if not enabled:
                continue
            t = time.perf_counter()
            for variant in variants:
                hits = self._retrieve(mode, variant, k, filters)
                rankings.append([c.chunk_id for c, _ in hits])
                # Rewrites support the original question; they must not outvote it.
                weights.append(1.0 if variant == query else cfg.variant_weight)
                for rank, (c, _) in enumerate(hits, start=1):
                    chunks[c.chunk_id] = c
                    best_rank[mode][c.chunk_id] = min(rank, best_rank[mode].get(c.chunk_id, rank))
            timings[mode] = round(time.perf_counter() - t, 3)

        fused = rrf_fuse(rankings, cfg.rrf_k, weights)
        ordered = sorted(fused, key=lambda cid: fused[cid], reverse=True)

        rerank_scores: dict[str, float] = {}
        if cfg.rerank and ordered:
            t = time.perf_counter()
            candidates = [chunks[cid] for cid in ordered[: cfg.rerank_candidates]]
            rerank_scores = dict(
                zip((c.chunk_id for c in candidates), self._rerank_scores(query, candidates), strict=True)
            )
            ordered = sorted(rerank_scores, key=lambda cid: rerank_scores[cid], reverse=True)
            timings["rerank"] = round(time.perf_counter() - t, 3)

        hits = [
            SearchHit(
                chunk=chunks[cid],
                score=rerank_scores.get(cid, fused[cid]),
                rank=i,
                dense_rank=best_rank["dense"].get(cid),
                bm25_rank=best_rank["bm25"].get(cid),
                rrf_score=round(fused[cid], 6),
                rerank_score=rerank_scores.get(cid),
            )
            for i, cid in enumerate(self._dedupe(ordered, chunks, cfg)[:top_k], start=1)
        ]
        timings["total"] = round(time.perf_counter() - started, 3)
        log.info(
            "search query=%r filters=%s variants=%d hits=%d timings=%s",
            query,
            filters,
            len(variants),
            len(hits),
            timings,
        )
        return SearchResult(query=query, variants=variants, filters=filters or {}, hits=hits, timings=timings)
