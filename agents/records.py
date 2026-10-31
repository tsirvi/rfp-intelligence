"""Reading the extracted bid records (outputs/<bid>.json) that the Q&A, comparison and go/no-go agents build on."""

from __future__ import annotations

import json
from pathlib import Path

from schemas.extraction import Source


def record_fields(outputs_dir: Path, bid: str) -> dict | None:
    """The record's fields by name, or None if the bid has not been extracted yet."""
    path = outputs_dir / f"{bid}.json"
    return json.loads(path.read_text("utf-8"))["fields"] if path.exists() else None


def validated(field: dict | None) -> bool:
    """A value the validator passed, with a citation: the only kind used as a fact without the LLM."""
    return bool(field and field.get("status") == "passed" and field.get("value") and field.get("sources"))


def first_source(field: dict) -> Source:
    return Source(**field["sources"][0])


def short_label(source: Source, limit: int = 30) -> str:
    """ "File, p. N" with a long file name shortened, for table cells (full labels go in the source list)."""
    short = source.file if len(source.file) <= limit else source.file[: limit - 1] + "…"
    return source.label().replace(source.file, short, 1)
