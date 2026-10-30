"""Persistent index: chunk vectors in Chroma, a BM25 view over the same chunks, and parsed documents as JSON.

Indexing is incremental per file: a file whose sha256 is already stored is not re-parsed or re-embedded,
a changed file replaces its old chunks, and a file removed from the bid folder is removed from the index.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from ingestion.pipeline import SUPPORTED_SUFFIXES, ingest_folder
from schemas.chunks import Chunk
from schemas.documents import Document, FileFailure
from search.chunking import chunk_document
from search.embeddings import Embedder, FastEmbedder
from settings import ROOT

log = logging.getLogger(__name__)

# Filters map a metadata field to a value or a list of accepted values, e.g.
# {"bid_id": "Bid1", "doc_type": ["rfp", "addendum"]}.
Filters = dict[str, Any]

# fmt: off
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with",
})
# fmt: on
_WORD = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """BM25 tokens. Compound tokens such as SKUs ("210-blyz") are kept whole and also split into parts."""
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        tokens.append(word)
        parts = re.split(r"[-./]", word)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in _STOPWORDS)
    return tokens


def to_where(filters: Filters | None) -> dict[str, Any] | None:
    clauses = [
        {k: {"$in": list(v)} if isinstance(v, list | tuple | set) else v}
        for k, v in (filters or {}).items()
        if v is not None and v != []
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches(meta: dict[str, Any], filters: Filters | None) -> bool:
    for key, value in (filters or {}).items():
        if value is None or value == []:
            continue
        accepted = value if isinstance(value, list | tuple | set) else [value]
        if meta.get(key) not in accepted:
            return False
    return True


class LuceneBM25(BM25Okapi):
    """BM25 with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)). Okapi's IDF is zero or negative for
    terms in half the corpus or more, which on a small bid (or a filtered subset) hides real matches."""

    def _calc_idf(self, nd: dict[str, int]) -> None:
        self.idf = {w: math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5)) for w, n in nd.items()}


class IndexReport(BaseModel):
    bid_id: str
    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    failures: list[FileFailure] = Field(default_factory=list)
    chunks_added: int = 0
    seconds: float = 0.0


class IndexStore:
    def __init__(self, index_dir: str | Path | None = None, embedder: Embedder | None = None):
        path = Path(index_dir or os.getenv("INDEX_DIR", "index"))
        self.dir = path if path.is_absolute() else ROOT / path
        self.docs_dir = self.dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or FastEmbedder()
        client = chromadb.PersistentClient(path=str(self.dir / "chroma"), settings=Settings(anonymized_telemetry=False))
        self._col = client.get_or_create_collection(
            "chunks", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._version_file = self.dir / "VERSION"
        self._bm25: tuple[str, LuceneBM25, list[Chunk]] | None = None

    # ---- documents -------------------------------------------------------------------------------

    def _doc_path(self, bid_id: str, file_name: str) -> Path:
        return self.docs_dir / bid_id / f"{file_name}.json"

    def bids(self) -> list[str]:
        return sorted(p.name for p in self.docs_dir.iterdir() if p.is_dir() and any(p.glob("*.json")))

    def documents(self, bid_id: str | None = None) -> list[Document]:
        pattern = f"{bid_id}/*.json" if bid_id else "*/*.json"
        return [Document.model_validate_json(p.read_text("utf-8")) for p in sorted(self.docs_dir.glob(pattern))]

    def known_hashes(self, bid_id: str) -> dict[str, str]:
        """file_name -> file_hash of the documents already indexed for a bid."""
        return {d.file_name: d.file_hash for d in self.documents(bid_id)}

    # ---- writes ----------------------------------------------------------------------------------

    def _touch(self) -> None:
        # Other processes (API, CLI) compare this to know their BM25 view is stale.
        self._version_file.write_text(str(time.time_ns()))
        self._bm25 = None

    def add_document(self, doc: Document) -> int:
        """Chunk, embed and store one document, replacing any previous version. Returns the chunk count."""
        chunks = chunk_document(doc)
        vectors = self.embedder.embed_passages([c.indexed_text for c in chunks]) if chunks else []
        self._col.delete(where={"doc_id": doc.doc_id})
        if chunks:
            self._col.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[c.to_metadata() for c in chunks],
            )
        # Written last: a document JSON on disk means its chunks are in the index.
        path = self._doc_path(doc.bid_id, doc.file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(doc.model_dump_json(), "utf-8")
        tmp.replace(path)
        self._touch()
        return len(chunks)

    def remove_document(self, bid_id: str, file_name: str) -> None:
        self._col.delete(where={"doc_id": f"{bid_id}/{file_name}"})
        self._doc_path(bid_id, file_name).unlink(missing_ok=True)
        self._touch()

    def index_folder(self, folder: str | Path, bid_id: str | None = None, force: bool = False) -> IndexReport:
        """force re-parses every file, e.g. after a parser change (file hashes alone would not notice)."""
        started = time.perf_counter()
        folder = Path(folder)
        bid_id = bid_id or folder.name
        known = self.known_hashes(bid_id)
        result = ingest_folder(folder, bid_id, known_files=None if force else known)
        report = IndexReport(
            bid_id=bid_id, unchanged=result.unchanged, skipped=result.skipped, failures=result.failures
        )

        for doc in result.documents:
            try:
                report.chunks_added += self.add_document(doc)
                (report.updated if doc.file_name in known else report.added).append(doc.file_name)
            except Exception as exc:
                report.failures.append(FileFailure(file_name=doc.file_name, error=f"indexing: {exc}"))
                log.error("index failed bid=%s file=%s error=%s", bid_id, doc.file_name, exc)

        present = {p.name for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES}
        for file_name in sorted(set(known) - present):
            self.remove_document(bid_id, file_name)
            report.removed.append(file_name)

        report.seconds = round(time.perf_counter() - started, 1)
        log.info(
            "indexed bid=%s added=%d updated=%d unchanged=%d removed=%d failed=%d chunks=%d secs=%.1f",
            bid_id,
            len(report.added),
            len(report.updated),
            len(report.unchanged),
            len(report.removed),
            len(report.failures),
            report.chunks_added,
            report.seconds,
        )
        return report

    # ---- reads -----------------------------------------------------------------------------------

    def version(self) -> str:
        """Changes whenever the index is written to (by any process)."""
        return self._version_file.read_text() if self._version_file.exists() else ""

    def count(self) -> int:
        return self._col.count()

    def chunks(self, filters: Filters | None = None) -> list[Chunk]:
        got = self._col.get(where=to_where(filters), include=["documents", "metadatas"])
        return [
            Chunk.from_metadata(i, t, m) for i, t, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        ]

    def dense_search(self, query: str, k: int = 20, filters: Filters | None = None) -> list[tuple[Chunk, float]]:
        if not self.count():
            return []
        res = self._col.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where=to_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        return [(Chunk.from_metadata(i, t, m), 1.0 - d) for i, t, m, d in rows]

    def _bm25_index(self) -> tuple[LuceneBM25, list[Chunk]] | None:
        version = self.version()
        if self._bm25 is None or self._bm25[0] != version:
            chunks = self.chunks()
            if not chunks:
                return None
            self._bm25 = (version, LuceneBM25([tokenize(c.indexed_text) for c in chunks]), chunks)
        return self._bm25[1], self._bm25[2]

    def bm25_search(self, query: str, k: int = 20, filters: Filters | None = None) -> list[tuple[Chunk, float]]:
        built = self._bm25_index()
        terms = tokenize(query)
        if built is None or not terms:
            return []
        bm25, chunks = built
        # Scored over the whole corpus (stable IDF), then filtered.
        scored: Iterable[tuple[Chunk, float]] = zip(chunks, bm25.get_scores(terms), strict=True)
        hits = [(c, float(s)) for c, s in scored if s > 0 and matches(c.to_metadata(), filters)]
        return sorted(hits, key=lambda h: h[1], reverse=True)[:k]
