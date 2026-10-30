"""Retrieval agent: turns a field group or a question into ranked, cited evidence via the search engine."""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence

from agents.fields import FieldSpec
from agents.tracing import Step, Tracer
from schemas.chunks import Chunk
from schemas.extraction import Evidence, Source
from search.chunking import count_tokens
from search.hybrid import HybridSearcher
from search.store import Filters, tokenize


def _tokens(evidence: Sequence[Evidence]) -> int:
    return sum(count_tokens(e.text) for e in evidence)


def to_evidence(chunk: Chunk, score: float) -> Evidence:
    return Evidence(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source=Source(file=chunk.file_name, page=chunk.page_start, page_end=chunk.page_end, chunk_id=chunk.chunk_id),
        doc_type=chunk.doc_type.value,
        addendum_number=chunk.addendum_number,
        section=chunk.section,
        score=score,
    )


class RetrievalAgent:
    def __init__(self, searcher: HybridSearcher, tracer: Tracer, per_query_k: int = 4, budget_tokens: int = 1600):
        self.searcher = searcher
        self.tracer = tracer
        self.per_query_k = per_query_k
        self.budget_tokens = budget_tokens

    def _ranked(self, queries: Sequence[str], filters: Filters, k: int, step: Step) -> list[Evidence]:
        """Search each query and fuse the result lists (reciprocal rank), best first."""
        scores: dict[str, float] = {}
        passages: dict[str, Evidence] = {}
        for query in queries:
            started = time.perf_counter()
            result = self.searcher.search(query, filters, top_k=k)
            step.tool(
                "search",
                {"q": query, "filters": filters, "k": k},
                [h.citation for h in result.hits],
                time.perf_counter() - started,
            )
            for h in result.hits:
                cid = h.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + h.rank)
                passages.setdefault(cid, to_evidence(h.chunk, 0.0))
        for cid, ev in passages.items():
            ev.score = round(scores[cid], 5)
        return sorted(passages.values(), key=lambda e: -e.score)

    def _fill(self, ranked_lists: Sequence[Sequence[Evidence]], budget: int) -> list[Evidence]:
        """Take passages round-robin from each list (each field's best, then each field's second best, ...)
        until the token budget is used, so no field is crowded out by another's long passages."""
        chosen: list[tuple[Evidence, set[str], int]] = []
        used = 0
        for round_ in itertools.zip_longest(*ranked_lists):
            for ev in round_:
                if ev is None or any(c.chunk_id == ev.chunk_id for c, _, _ in chosen):
                    continue
                words = set(tokenize(ev.text))
                # Skip a passage mostly contained in one already chosen; a passage that contains chosen ones
                # (a paragraph around a fact line) replaces them, so the context is not lost.
                if any(len(words & other) >= 0.8 * max(1, len(words)) for _, other, _ in chosen):
                    continue
                inside = [c for c in chosen if len(words & c[1]) >= 0.8 * max(1, len(c[1]))]
                tokens = count_tokens(ev.text)
                freed = sum(t for _, _, t in inside)
                if chosen and used - freed + tokens > budget:
                    continue
                chosen = [c for c in chosen if c not in inside] + [(ev, words, tokens)]
                used += tokens - freed
        return [ev for ev, _, _ in chosen]

    def gather(
        self,
        queries: Sequence[str],
        filters: Filters,
        agent_label: str,
        k: int | None = None,
        budget: int | None = None,
    ) -> list[Evidence]:
        """Evidence for one information need: all queries fused into one ranking, then the token budget."""
        with self.tracer.step("retrieval", agent_label, {"queries": list(queries), "filters": filters}) as step:
            ranked = self._ranked(queries, filters, k or self.per_query_k, step)
            chosen = self._fill([ranked], self.budget_tokens if budget is None else budget)
            step.output({"passages": [e.source.label() for e in chosen], "tokens": _tokens(chosen)})
            return chosen

    def for_fields(self, specs: Sequence[FieldSpec], bid_id: str) -> list[Evidence]:
        """Evidence for a group of fields: a ranking per field, merged round-robin. Fields with preferred
        document types (e.g. part numbers -> specification sheets) put the top hits from those first."""
        label = "fields:" + ",".join(s.name for s in specs)
        with self.tracer.step("retrieval", label, {"fields": [s.name for s in specs], "bid_id": bid_id}) as step:
            per_field = []
            for spec in specs:
                ranked = self._ranked(spec.queries, {"bid_id": bid_id}, self.per_query_k, step)
                if spec.doc_types:
                    preferred = self._ranked(
                        spec.queries, {"bid_id": bid_id, "doc_type": list(spec.doc_types)}, self.per_query_k, step
                    )[:2]
                    ranked = preferred + [e for e in ranked if e.chunk_id not in {p.chunk_id for p in preferred}]
                per_field.append(ranked)
            chosen = self._fill(per_field, self.budget_tokens)
            step.output({"passages": [e.source.label() for e in chosen], "tokens": _tokens(chosen)})
            return chosen
