"""PDF parsing: layout-aware text (PyMuPDF), tables (pdfplumber) and OCR fallback (Tesseract)."""

from __future__ import annotations

import io
import itertools
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf

from ingestion.cleaning import BULLETS, clean_lines, normalise_whitespace, strip_repeated_edges
from schemas.documents import Page, Table

log = logging.getLogger(__name__)


@dataclass
class ParsedPdf:
    pages: list[Page]
    title: str | None
    running_header: str | None
    metadata_date: date | None
    warnings: list[str] = field(default_factory=list)


def layout_lines(page: pymupdf.Page, column_gap: float = 18.0) -> list[str]:
    """Rebuild reading lines from word positions.

    Words are grouped by vertical position, so a value printed beside its label (or a SKU beside its
    description in a borderless table) stays on the same line; a wide horizontal gap becomes " | ".
    A blank line marks a vertical gap, which the chunker later uses as a paragraph boundary.
    """
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[dict[str, Any]] = []
    for w in words:
        y_mid, height = (w[1] + w[3]) / 2, w[3] - w[1]
        if rows and abs(y_mid - rows[-1]["y"]) <= max(2.0, 0.4 * height):
            rows[-1]["words"].append(w)
        else:
            rows.append({"y": y_mid, "h": height, "words": [w]})

    lines: list[str] = []
    prev_y: float | None = None
    for row in rows:
        ws = sorted(row["words"], key=lambda w: w[0])
        gaps = [b[0] - a[2] > column_gap and a[4] not in BULLETS for a, b in itertools.pairwise(ws)]
        # Justified text spreads single words across the line ("in  to  submit  a  bid"): when most
        # "columns" would be one word long, it is prose, not columns.
        if len(gaps) >= 3 and sum(gaps) / len(gaps) > 0.6:
            gaps = [False] * len(gaps)
        text = ws[0][4]
        for b, wide_gap in zip(ws[1:], gaps, strict=True):
            text += (" | " if wide_gap else " ") + b[4]
        if prev_y is not None and row["y"] - prev_y > 2.2 * row["h"]:
            lines.append("")
        lines.append(text)
        prev_y = row["y"]
    return lines


def _ocr_lines(page: pymupdf.Page, dpi: int) -> list[str]:
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png")))
    return normalise_whitespace(pytesseract.image_to_string(image)).split("\n")


def clean_table(rows: list[list[str | None]]) -> list[list[str]] | None:
    """Normalise a pdfplumber table; return None for layout boxes that are not real tables."""
    grid = [[normalise_whitespace(c or "").replace("\n", " ") for c in row] for row in rows]
    grid = [row for row in grid if any(row)]
    if not grid:
        return None
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [j for j in range(width) if any(r[j] for r in grid)]
    grid = [[r[j] for j in keep] for r in grid]

    # Form-style grids (merged cells everywhere, even in the first row): column alignment carries no
    # meaning, so drop the empty cells. Real tables keep their grid even when value columns are blank,
    # e.g. an unfilled price schedule.
    if sum(1 for c in grid[0] if not c) / len(grid[0]) > 0.5:
        grid = [[c for c in r if c] for r in grid]
    if len(grid) < 2 or max(len(r) for r in grid) < 2:
        return None
    return grid


def _metadata_date(raw: str | None) -> date | None:
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw or "")
    try:
        return date(int(m[1]), int(m[2]), int(m[3])) if m else None
    except ValueError:
        return None


def parse_pdf(path: Path, cfg: dict[str, Any] | None = None) -> ParsedPdf:
    cfg = cfg or {}
    column_gap = float(cfg.get("column_gap_pt", 18))
    ocr_enabled = bool(cfg.get("ocr_enabled", True))
    ocr_min_chars = int(cfg.get("ocr_min_chars", 50))
    ocr_dpi = int(cfg.get("ocr_dpi", 200))
    warnings: list[str] = []

    doc = pymupdf.open(path)
    if doc.needs_pass:
        raise ValueError("PDF is password-protected")

    raw_pages: list[list[str]] = []
    ocr_pages: set[int] = set()
    textless_pages: set[int] = set()
    for i, page in enumerate(doc):
        lines = layout_lines(page, column_gap)
        if sum(len(line) for line in lines) < ocr_min_chars:
            textless_pages.add(i)
            has_visuals = bool(page.get_images()) or bool(page.get_drawings())
            if ocr_enabled and has_visuals:
                try:
                    ocr = _ocr_lines(page, ocr_dpi)
                    if sum(len(line) for line in ocr) > sum(len(line) for line in lines):
                        lines = ocr
                        ocr_pages.add(i)
                        warnings.append(f"page {i + 1}: no text layer, used OCR")
                except Exception as exc:  # tesseract missing or failed: keep going without this page
                    warnings.append(f"page {i + 1}: OCR failed ({exc})")
            if not any(line.strip() for line in lines):
                warnings.append(f"page {i + 1}: empty page, no text extracted")
        raw_pages.append(lines)

    stripped, header = strip_repeated_edges(raw_pages, float(cfg.get("header_footer_min_ratio", 0.5)))

    tables: dict[int, list[Table]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # No text layer means no text tables, and vector-drawn pages (scanned forms rendered
                # as outlines) take pdfplumber minutes to parse.
                if i in textless_pages:
                    continue
                for raw in page.extract_tables():
                    if rows := clean_table(raw):
                        tables.setdefault(i, []).append(Table(page_number=i + 1, rows=rows))
    except Exception as exc:
        warnings.append(f"table extraction failed ({exc})")

    pages = [
        Page(page_number=i + 1, text=clean_lines(lines), tables=tables.get(i, []), ocr=i in ocr_pages)
        for i, lines in enumerate(stripped)
    ]
    meta = doc.metadata or {}
    title = re.sub(r"^Microsoft (Word|Excel) - ", "", meta.get("title") or "").strip() or None
    for w in warnings:
        log.warning("%s: %s", path.name, w)
    return ParsedPdf(pages, title, header, _metadata_date(meta.get("creationDate")), warnings)
