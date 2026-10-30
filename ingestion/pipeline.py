"""Ingest a bid folder: detect file type, parse, clean, attach metadata. Never raises for one bad file."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ingestion import metadata
from ingestion.html_parser import parse_html
from ingestion.pdf_parser import parse_pdf
from schemas.documents import DocType, Document, FileFailure, IngestionResult, Page
from settings import section

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_line(text: str) -> str | None:
    return next((line for line in text.split("\n") if line.strip()), None)


def ingest_file(path: Path, bid_id: str, cfg: dict[str, Any] | None = None, digest: str | None = None) -> Document:
    cfg = cfg if cfg is not None else section("ingestion")
    digest = digest or file_hash(path)
    common = {"doc_id": f"{bid_id}/{path.name}", "bid_id": bid_id, "file_name": path.name, "file_hash": digest}

    if path.suffix.lower() in (".html", ".htm"):
        html = parse_html(path)
        issued = metadata.issue_date_from_fields(html.fields)
        return Document(
            **common,
            doc_type=DocType.BID_PAGE,
            title=html.title,
            document_date=issued,
            document_date_source="html_field" if issued else None,
            fields=html.fields,
            pages=[Page(page_number=1, text=html.text)],
            warnings=html.warnings,
        )

    pdf = parse_pdf(path, cfg)
    first_text = next((p.text for p in pdf.pages if p.text), "")
    opening = "\n".join(filter(None, [pdf.running_header, pdf.title, first_text]))
    doc_type = metadata.classify(path.name, opening)

    issued = metadata.issue_date_from_text("\n".join(p.text for p in pdf.pages[:2]))
    date_source = "text" if issued else None
    if not issued and pdf.metadata_date:
        issued, date_source = pdf.metadata_date, "pdf_metadata"

    return Document(
        **common,
        doc_type=doc_type,
        addendum_number=metadata.addendum_number(path.name, opening) if doc_type == DocType.ADDENDUM else None,
        # The running header is printed on every page, so it is more reliable than PDF metadata titles.
        title=pdf.running_header or pdf.title or _first_line(first_text),
        document_date=issued,
        document_date_source=date_source,
        running_header=pdf.running_header,
        pages=pdf.pages,
        warnings=pdf.warnings,
    )


def ingest_folder(
    folder: str | Path,
    bid_id: str | None = None,
    known_files: Mapping[str, str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> IngestionResult:
    """Parse every supported file in a bid folder.

    known_files: file_name -> sha256 of files already in the index; a file whose name and hash both
    match is reported as unchanged and not re-parsed, which keeps re-indexing a bid cheap.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Bid folder not found: {folder}")
    result = IngestionResult(bid_id=bid_id or folder.name)

    for path in sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            result.skipped.append(path.name)
            log.info("skip unsupported file bid=%s file=%s", result.bid_id, path.name)
            continue
        started = time.perf_counter()
        try:
            digest = file_hash(path)
            if (known_files or {}).get(path.name) == digest:
                result.unchanged.append(path.name)
                continue
            doc = ingest_file(path, result.bid_id, cfg, digest)
            result.documents.append(doc)
            log.info(
                "ingested bid=%s file=%s type=%s pages=%d warnings=%d secs=%.1f",
                doc.bid_id,
                doc.file_name,
                doc.doc_type.value,
                len(doc.pages),
                len(doc.warnings),
                time.perf_counter() - started,
            )
        except Exception as exc:  # a corrupt or unreadable file must not stop the rest of the bid
            result.failures.append(FileFailure(file_name=path.name, error=f"{type(exc).__name__}: {exc}"))
            log.error("failed bid=%s file=%s error=%s", result.bid_id, path.name, exc)
    return result
