"""HTML bid-page parsing: strip portal boilerplate, keep label/value fields in reading order."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from ingestion.cleaning import clean_lines, normalise_whitespace

# Removed outright. Hidden inputs and meta tags also carry session tokens, which must not be indexed.
_JUNK_TAGS = [
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "template",
    "button",
    "input",
    "select",
    "textarea",
    "nav",
    "header",
    "footer",
    "link",
    "meta",
    "img",
]
# Matched against each class token and the id, anchored, so "side-bar" does not match "side-bar-view-content".
_JUNK_TOKEN = re.compile(
    r"^(nav|navbar|navigation|.*menus?|cookie.*|share.*|.*actions|breadcrumbs?|.*dialog.*|modal|hidden"
    r"|side-?bar|.*throbber.*|.*ajax.*|skip.*|read-?more|disclaimer)$",
    re.IGNORECASE,
)
_LABEL_TOKEN = re.compile(r"^(?!no-).*label$", re.IGNORECASE)  # "field-label", not "no-label" wrappers
_HEADING = re.compile(r"^h[1-6]$")


@dataclass
class ParsedHtml:
    text: str
    title: str | None
    fields: dict[str, str]
    warnings: list[str] = field(default_factory=list)


def _is_junk(tag: Tag) -> bool:
    if tag.attrs is None:
        return False
    tokens = list(tag.get("class") or []) + [tag.get("id") or ""]
    style = (tag.get("style") or "").replace(" ", "").lower()
    return any(t and _JUNK_TOKEN.match(t) for t in tokens) or "display:none" in style


def _text(tag: Tag) -> str:
    return normalise_whitespace(tag.get_text(" ", strip=True))


def _value_for(label: Tag) -> Tag | None:
    if label.name == "dt":
        return label.find_next_sibling("dd")
    if label.name == "th":
        return label.find_next_sibling("td")
    if label.name == "label" and label.get("for"):
        return label.find_next(id=label["for"])
    return label.find_next_sibling()


def _nearest_heading(tag: Tag) -> str:
    heading = tag.find_previous(_HEADING)
    return _text(heading) if heading else ""


def parse_html(path: Path) -> ParsedHtml:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    title_tag = soup.find("title")
    page_title = _text(title_tag) if title_tag else None

    for tag in soup.find_all(_JUNK_TAGS):
        tag.decompose()
    for tag in [t for t in soup.find_all(True) if _is_junk(t)]:
        if not tag.decomposed:
            tag.decompose()

    root = soup.find("main") or soup.body or soup
    h1 = root.find("h1")
    title = _text(h1) if h1 else page_title

    # Show the real target of links whose visible text is a (possibly truncated) URL.
    for a in root.find_all("a", href=True):
        if a.get_text(strip=True).lower().startswith("http") and a["href"].startswith("http"):
            a.string = a["href"]

    fields: dict[str, str] = {}
    labels = [
        t
        for t in root.find_all(True)
        if t.name in ("dt", "th", "label") or any(_LABEL_TOKEN.match(c) for c in t.get("class") or [])
    ]
    for label in labels:
        if label.decomposed:
            continue
        value = _value_for(label)
        if value is None or value.decomposed:
            continue
        value_text = _text(value)
        if not value_text:
            continue
        # Blank labels (e.g. contact lines) take their meaning from the section heading above them.
        key = _text(label).rstrip(":") or _nearest_heading(label) or "Field"
        fields[key] = f"{fields[key]}; {value_text}" if key in fields else value_text
        label.replace_with(NavigableString(f"\n{key}: {value_text}\n"))
        value.decompose()

    lines = normalise_whitespace(root.get_text("\n")).split("\n")
    deduped = [line for i, line in enumerate(lines) if i == 0 or line != lines[i - 1]]
    text = clean_lines(deduped)
    warnings = [] if text else ["no text extracted from HTML"]
    return ParsedHtml(text=text, title=title, fields=fields, warnings=warnings)
