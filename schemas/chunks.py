"""Retrieval units produced by the chunker and stored in the index."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from schemas.documents import DocType


class Chunk(BaseModel):
    chunk_id: str  # "<doc_id>#<n>", stable for a given file hash
    doc_id: str
    bid_id: str
    file_name: str
    file_hash: str
    doc_type: DocType
    addendum_number: int | None = None
    document_date: str | None = None  # ISO date
    title: str | None = None
    section: str | None = None
    page_start: int
    page_end: int
    kind: Literal["text", "table", "fact"] = "text"  # fact: a short date/contact/amount line with its label
    text: str  # what the user and the LLM see

    @property
    def context(self) -> str:
        """Short document context prepended before embedding and BM25, so a chunk that only says
        "due July 9" still matches a query about Addendum 2 of this bid."""
        doc = self.doc_type.value.replace("_", " ")
        if self.addendum_number is not None:
            doc += f" {self.addendum_number}"
        parts = [self.bid_id, doc, self.title, self.section]
        return " | ".join(p for p in parts if p)

    @property
    def indexed_text(self) -> str:
        return f"{self.context}\n{self.text}"

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.file_name}, {pages}"

    def to_metadata(self) -> dict[str, Any]:
        """Flat, None-free metadata for the vector store."""
        meta = self.model_dump(exclude={"chunk_id", "text"})
        meta["doc_type"] = self.doc_type.value
        return {k: v for k, v in meta.items() if v is not None}

    @classmethod
    def from_metadata(cls, chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
        return cls(chunk_id=chunk_id, text=text, **meta)
