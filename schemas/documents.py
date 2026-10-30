"""Normalised document models produced by ingestion and consumed by chunking/indexing."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    BID_PAGE = "bid_page"
    RFP = "rfp"
    ADDENDUM = "addendum"
    SPECS = "specs"
    AFFIDAVIT = "affidavit"
    OTHER = "other"


class Table(BaseModel):
    page_number: int
    rows: list[list[str]]

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        width = max(len(r) for r in self.rows)
        rows = [r + [""] * (width - len(r)) for r in self.rows]
        cell = lambda s: s.replace("|", "/").replace("\n", " ")
        lines = ["| " + " | ".join(cell(c) for c in rows[0]) + " |", "|" + " --- |" * width]
        lines += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows[1:]]
        return "\n".join(lines)


class Page(BaseModel):
    page_number: int  # 1-based physical page; HTML pages are always 1
    text: str
    tables: list[Table] = Field(default_factory=list)
    ocr: bool = False


class Document(BaseModel):
    doc_id: str  # stable id: "<bid_id>/<file_name>"
    bid_id: str
    file_name: str
    file_hash: str  # sha256 of file bytes, used for incremental indexing
    doc_type: DocType
    addendum_number: int | None = None
    title: str | None = None
    document_date: date | None = None
    document_date_source: str | None = None  # "text", "html_field" or "pdf_metadata"
    running_header: str | None = None  # repeated page header removed during cleaning
    fields: dict[str, str] = Field(default_factory=dict)  # label/value pairs from bid pages
    pages: list[Page] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


class FileFailure(BaseModel):
    file_name: str
    error: str


class IngestionResult(BaseModel):
    bid_id: str
    documents: list[Document] = Field(default_factory=list)
    failures: list[FileFailure] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # unsupported file types
    unchanged: list[str] = Field(default_factory=list)  # already indexed with the same hash
