"""Document-level metadata: type, addendum number, document date and title.

Rules check the file name first (it is what the issuer chose to call the file), then the opening
text of the document. Nothing here is specific to the sample bids.
"""

from __future__ import annotations

import re
from datetime import date

from dateutil import parser as dateparser

from schemas.documents import DocType

_TYPE_PATTERNS: list[tuple[DocType, re.Pattern[str]]] = [
    (DocType.ADDENDUM, re.compile(r"\b(addend(um|a)|amendment)\b", re.IGNORECASE)),
    (DocType.AFFIDAVIT, re.compile(r"\baffidavits?\b", re.IGNORECASE)),
    (DocType.SPECS, re.compile(r"\b(specs?|specifications?|datasheet|data sheet|bill of materials)\b", re.IGNORECASE)),
    (
        DocType.RFP,
        re.compile(
            r"\b(rfp|rfq|rfi|ifb|itb|porfp|solicitation|tender)\b"
            r"|request for (proposals?|quot(es?|ations?)|information|bids?)|invitation (to|for) (bid|tender)",
            re.IGNORECASE,
        ),
    ),
]
_ADDENDUM_NUMBER = re.compile(
    r"\b(?:addend(?:um|a)|amendment)\s*(?:no\.?|number|#)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    w: i for i, w in enumerate(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"], 1)
}
_DATE_TOKEN = r"(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\.? \d{1,2},? \d{4})"
_ISSUE_DATE = re.compile(r"(issue date|date issued|issued on|dated)\b[^\n]{0,60}?" + _DATE_TOKEN, re.IGNORECASE)
_ISSUE_FIELD = re.compile(r"publication|published|issue|posted", re.IGNORECASE)


def _normalise_name(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem)


def classify(file_name: str, opening_text: str) -> DocType:
    """opening_text: title, running header and the start of the first page."""
    if file_name.lower().endswith((".html", ".htm")):
        return DocType.BID_PAGE
    for doc_type, pattern in _TYPE_PATTERNS:
        if pattern.search(_normalise_name(file_name)):
            return doc_type
    # In body text, RFP wording outranks "specifications": RFP cover pages often mention both.
    text_order = [DocType.ADDENDUM, DocType.AFFIDAVIT, DocType.RFP, DocType.SPECS]
    patterns = dict(_TYPE_PATTERNS)
    for doc_type in text_order:
        if patterns[doc_type].search(opening_text[:400]):
            return doc_type
    return DocType.OTHER


def addendum_number(file_name: str, opening_text: str) -> int | None:
    for source in (_normalise_name(file_name), opening_text[:400]):
        if m := _ADDENDUM_NUMBER.search(source):
            token = m[1].lower()
            return int(token) if token.isdigit() else _WORD_NUMBERS[token]
    return None


def parse_date(text: str) -> date | None:
    try:
        return dateparser.parse(text, fuzzy=True).date()
    except (ValueError, OverflowError, TypeError):
        return None


def issue_date_from_text(text: str) -> date | None:
    m = _ISSUE_DATE.search(text)
    return parse_date(m[2]) if m else None


def issue_date_from_fields(fields: dict[str, str]) -> date | None:
    for label, value in fields.items():
        if _ISSUE_FIELD.search(label) and (d := parse_date(value)):
            return d
    return None
