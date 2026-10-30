"""Split documents into retrieval chunks that follow section headings and keep tables whole.

Text is packed paragraph by paragraph up to a token budget, so a chunk never cuts a sentence unless
the sentence alone is over budget. A new section starts a new chunk, which keeps citations and the
section label accurate. Tables become their own chunks in Markdown; large tables are split by rows
with the header row repeated, so every piece is readable on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ingestion.cleaning import BULLETS, is_caps_line
from schemas.chunks import Chunk
from schemas.documents import Document, Table
from settings import section as config_section

_TOKEN = re.compile(r"\w+|[^\w\s]")
_RULE = re.compile(r"^[\W_]{8,}$")  # dashed/dotted lines from form layouts
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)+\s+\S")  # "3.2 Delivery Requirements"
_LABELLED_HEADING = re.compile(
    r"^(section|part|article|attachment|exhibit|appendix)\s+[\w.]+\s*[–—:-]\s*\S", re.IGNORECASE
)


def count_tokens(text: str) -> int:
    """Approximate subword count; errs low, so the budget leaves headroom under the 512-token model limit."""
    return len(_TOKEN.findall(text))


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or " | " in line or line.startswith(BULLETS) or len(line) > 90:
        return False
    if line.endswith((".", ",", ";", ":")) or ". " in line or len(line.split()) > 12:
        return False
    return bool(_NUMBERED_HEADING.match(line) or _LABELLED_HEADING.match(line)) or is_caps_line(line)


@dataclass
class _Unit:
    text: str
    page: int
    section: str | None
    tokens: int
    heading: bool = False


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Break an over-budget paragraph at lines, then sentences, then words."""
    if count_tokens(text) <= max_tokens:
        return [text]
    for pattern in (r"\n", r"(?<=[.;!?])\s+"):
        parts = [p for p in re.split(pattern, text) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for part in parts:
                out.extend(_split_long(part, max_tokens))
            return _pack(out, max_tokens)
    words = text.split()
    step = max(1, int(max_tokens * 0.7))  # words run longer than tokens
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


def _pack(parts: list[str], max_tokens: int) -> list[str]:
    out: list[str] = []
    for part in parts:
        if out and count_tokens(out[-1]) + count_tokens(part) <= max_tokens:
            out[-1] = f"{out[-1]}\n{part}"
        else:
            out.append(part)
    return out


def _units(doc: Document, max_tokens: int) -> tuple[list[_Unit], dict[int, str | None]]:
    units: list[_Unit] = []
    page_section: dict[int, str | None] = {}
    section: str | None = None
    prev_heading = False
    para: list[str] = []

    def flush(page_number: int) -> None:
        for piece in _split_long("\n".join(para), max_tokens) if para else []:
            units.append(_Unit(piece, page_number, section, count_tokens(piece)))
        para.clear()

    for page in doc.pages:
        prev_heading = False  # a heading never continues onto the next page
        for line in page.text.split("\n"):
            if _RULE.match(line.strip()):
                continue
            if is_heading(line):
                flush(page.page_number)
                # Headings wrapped over two lines ("SCOPE AND SPECIFICATIONS OF THE" / "PROPOSAL").
                if prev_heading and section and len(f"{section} {line}".split()) <= 16:
                    section = f"{section} {line.strip()}"
                    units[-1].text = units[-1].section = section
                    units[-1].tokens = count_tokens(section)
                else:
                    section = line.strip()
                    units.append(_Unit(section, page.page_number, section, count_tokens(section), heading=True))
                prev_heading = True
                continue
            prev_heading = False
            if line.strip():
                para.append(line)
            else:
                flush(page.page_number)
        flush(page.page_number)
        page_section[page.page_number] = section
    return units, page_section


def _table_chunks(table: Table, max_tokens: int) -> list[str]:
    md = table.to_markdown()
    if count_tokens(md) <= max_tokens:
        return [md]
    lines = md.split("\n")
    header, rows = lines[:2], lines[2:]
    pieces: list[list[str]] = [[]]
    budget = max_tokens - count_tokens("\n".join(header))
    for row in rows:
        if pieces[-1] and count_tokens("\n".join([*pieces[-1], row])) > budget:
            pieces.append([])
        pieces[-1].append(row)
    return ["\n".join(header + piece) for piece in pieces if piece]


_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_FACT_VALUE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"  # 06/10/2024
    rf"|\b\d{{1,2}}-{_MONTH}-\d{{4}}\b"  # 27-JUN-2024
    rf"|\b{_MONTH}\.? \d{{1,2}},? \d{{4}}\b"  # July 9, 2024
    r"|[\w.+-]+@[\w-]+\.[\w.]+"  # email
    r"|\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"  # phone
    r"|\$\s?\d",  # money
    re.IGNORECASE,
)
_FACT_MAX_WORDS = 25  # longer lines are prose; the ordinary chunks cover them
_FACT_MAX_TOKENS = 60


def _fact_windows(doc: Document) -> list[tuple[int, str | None, str]]:
    """Short lines carrying a date, contact or amount, each with the label line above it.

    Header blocks pack a dozen such facts into one chunk, whose embedding then matches none of them
    well. These small extra chunks make "who is the buyer?" or "when are questions due?" retrievable.
    Returns (page, section, text).
    """
    out: list[tuple[int, str | None, str]] = []
    section: str | None = None
    window: list[str] = []

    def flush(page_number: int) -> None:
        if window:
            out.append((page_number, section, "\n".join(window)))
            window.clear()

    for page in doc.pages:
        lines = [line.strip() for line in page.text.split("\n")]
        last_used = -1
        for i, line in enumerate(lines):
            if is_heading(line):
                flush(page.page_number)
                section = line
                continue
            if not line or len(line.split()) > _FACT_MAX_WORDS or not _FACT_VALUE.search(line):
                continue
            # The label is the nearest non-blank line above (HTML fields are separated by blank lines).
            j = i - 1 if i > 0 and lines[i - 1] else i - 2
            prev = lines[j] if j >= 0 else ""
            label = prev if prev and j > last_used and len(prev.split()) <= 12 and not is_heading(prev) else ""
            adjacent = last_used >= i - 2  # continues the previous window
            addition = [x for x in (label, line) if x]
            if not adjacent or count_tokens("\n".join(window + addition)) > _FACT_MAX_TOKENS:
                flush(page.page_number)
            window.extend(addition)
            last_used = i
        flush(page.page_number)
    return out


def chunk_document(doc: Document, cfg: dict[str, Any] | None = None) -> list[Chunk]:
    cfg = cfg if cfg is not None else config_section("chunking")
    max_tokens = int(cfg.get("max_tokens", 350))
    overlap = int(cfg.get("overlap_tokens", 50))
    min_tokens = int(cfg.get("min_tokens", 80))

    base = {
        "doc_id": doc.doc_id,
        "bid_id": doc.bid_id,
        "file_name": doc.file_name,
        "file_hash": doc.file_hash,
        "doc_type": doc.doc_type,
        "addendum_number": doc.addendum_number,
        "document_date": doc.document_date.isoformat() if doc.document_date else None,
        "title": doc.title,
    }
    chunks: list[Chunk] = []

    def emit(units: list[_Unit]) -> None:
        if not units or all(u.heading for u in units):
            return
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                section=units[0].section,
                page_start=min(u.page for u in units),
                page_end=max(u.page for u in units),
                text="\n".join(u.text for u in units),
                **base,
            )
        )

    units, page_section = _units(doc, max_tokens)
    current: list[_Unit] = []
    for unit in units:
        size = sum(u.tokens for u in current)
        new_section = unit.heading and current and unit.section != current[-1].section
        if current and (size + unit.tokens > max_tokens or (new_section and size >= min_tokens)):
            # A heading at the very end belongs to the next chunk, not this one.
            headings: list[_Unit] = []
            while current and current[-1].heading:
                headings.insert(0, current.pop())
            emit(current)
            # Overlap: carry trailing paragraphs into the next chunk, but never across a section change.
            tail: list[_Unit] = []
            if not unit.heading and not headings:
                for u in reversed(current):
                    if u.heading or sum(t.tokens for t in tail) + u.tokens > overlap:
                        break
                    tail.insert(0, u)
            current = tail + headings
        current.append(unit)
    emit(current)

    for page in doc.pages:
        for table in page.tables:
            for md in _table_chunks(table, max_tokens):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{len(chunks)}",
                        section=page_section.get(page.page_number),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        kind="table",
                        text=md,
                        **base,
                    )
                )
    for page_number, section, text in _fact_windows(doc):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                section=section,
                page_start=page_number,
                page_end=page_number,
                kind="fact",
                text=text,
                **base,
            )
        )
    return chunks
