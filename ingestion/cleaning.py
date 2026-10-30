"""Text clean-up shared by the PDF and HTML parsers."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

BULLETS = ("•", "▪", "○", "◦", "", "●", "■")  # : Word/Symbol bullet
_SENTENCE_END = (".", ":", ";", "!", "?")


def normalise_whitespace(text: str) -> str:
    """NFKC-normalise (ligatures, non-breaking spaces), collapse runs of spaces and blank lines."""
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_caps_line(line: str) -> bool:
    """Mostly upper-case text, as used for section headings ("PRICING INFORMATION")."""
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) >= 0.8


def merge_lone_bullets(lines: list[str]) -> list[str]:
    """Join a bullet that sits on its own line with the line that follows it."""
    out: list[str] = []
    for line in lines:
        if out and out[-1].strip() in BULLETS and line.strip():
            out[-1] = f"{out[-1].strip()} {line.strip()}"
        else:
            out.append(line)
    return out


def unwrap_lines(lines: list[str]) -> list[str]:
    """Rejoin lines broken mid-sentence and words hyphenated across a line break.

    Hyphens are kept ("pre-\\nproposal" -> "pre-proposal"): RFPs are rarely auto-hyphenated, so a
    line-final hyphen is far more often a real compound than a split word. Column rows (" | ") are
    never joined, so table-like lines stay one row per line.
    """
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else ""
        joinable = (
            prev
            and line
            and " | " not in prev
            and " | " not in line
            and line[0].islower()
            and not prev.endswith(_SENTENCE_END)
            and not is_caps_line(prev)  # a heading, not the start of a wrapped sentence
            and not line.startswith(BULLETS)
        )
        if joinable:
            out[-1] = prev + line if re.search(r"[A-Za-z]-$", prev) else f"{prev} {line}"
        else:
            out.append(line)
    return out


_PAGE_MARKER = re.compile(r"\bpage\b|\bpg\b|^\W*\d+\W*$|\d+\s*(of|/|\|)\s*\d+", re.IGNORECASE)


def _line_key(line: str) -> str:
    # Page-number lines vary between pages ("Page 2 of 40"), so compare them with digits masked.
    # Other lines must repeat exactly: body lines that differ only by a number are not footers.
    key = line.lower().strip()
    return re.sub(r"\d+", "#", key) if _PAGE_MARKER.search(key) else key


def strip_repeated_edges(
    pages: list[list[str]], min_ratio: float = 0.5, edge: int = 3
) -> tuple[list[list[str]], str | None]:
    """Remove running headers/footers: lines near the top or bottom that repeat across pages.

    Returns the cleaned pages and the most frequent removed header line (kept as document metadata,
    so the context it carries, such as "ADDENDUM No. 1", is not lost).
    """
    if len(pages) < 3:
        return pages, None

    def edges(lines: list[str]) -> tuple[list[int], list[int]]:
        idx = [i for i, line in enumerate(lines) if line.strip()]
        # On short pages the window shrinks to a third, so it never covers the body.
        n = max(1, min(edge, len(idx) // 3))
        return idx[:n], idx[-n:]

    counts: Counter[str] = Counter()
    for lines in pages:
        top, bottom = edges(lines)
        counts.update({_line_key(lines[i]) for i in top + bottom})
    threshold = max(2, int(min_ratio * len(pages) + 0.999))
    repeated = {k for k, n in counts.items() if n >= threshold and k}

    header_counts: Counter[str] = Counter()
    cleaned: list[list[str]] = []
    for lines in pages:
        top, bottom = edges(lines)
        top_drop = [i for i in top if _line_key(lines[i]) in repeated]
        drop = set(top_drop) | {i for i in bottom if _line_key(lines[i]) in repeated}
        if top_drop:  # a header can span several lines ("ADDENDUM No. 1" / "RFP JA-207652 ...")
            header_counts[" ".join(lines[i].strip() for i in top_drop)] += 1
        cleaned.append([line for i, line in enumerate(lines) if i not in drop])
    header = header_counts.most_common(1)[0][0] if header_counts else None
    return cleaned, header


def clean_lines(lines: list[str]) -> str:
    return normalise_whitespace("\n".join(unwrap_lines(merge_lone_bullets(lines))))
