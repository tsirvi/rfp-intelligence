"""Query expansion with procurement synonyms.

Bids say the same thing many ways ("due date", "closing date", "solicitation due"). Each matched phrase
yields extra query variants with a synonym swapped in; all variants are retrieved and fused, so a
question phrased one way still finds a document phrased another. Deterministic and offline; an LLM
expander can be passed to the searcher instead.
"""

from __future__ import annotations

import itertools
import re

SYNONYMS: list[list[str]] = [
    ["due date", "deadline", "closing date", "submission deadline", "solicitation due", "proposal due"],
    ["pre-bid", "pre-proposal", "prebid conference", "pre-bid meeting", "site visit"],
    ["questions due", "deadline for questions", "inquiries", "clarification questions"],
    ["bid bond", "bid security", "proposal guarantee"],
    ["performance bond", "payment bond", "surety"],
    ["insurance", "liability coverage", "certificate of insurance"],
    ["contract term", "length of contract", "contract duration", "renewal", "initial term"],
    ["payment terms", "invoice", "net 30", "invoicing instructions"],
    ["contact", "buyer", "procurement officer", "point of contact", "purchasing agent"],
    ["delivery", "shipping", "lead time", "ship date"],
    ["quantity", "qty", "number of units", "target quantity"],
    ["price", "pricing", "cost", "price sheet", "unit price"],
    ["m/wbe", "mbe", "minority business", "small business", "sbr", "hub"],
    ["submit", "submission method", "submitted online", "electronic submission", "sealed bid"],
    ["addendum", "amendment", "addenda"],
    ["specifications", "requirements", "minimum requirements", "scope of work"],
    ["evaluation criteria", "scoring", "evaluation process", "basis of award", "best value"],
    ["warranty", "extended warranty", "guarantee"],
]


def _pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w/-]){re.escape(phrase)}(?![\w/-])", re.IGNORECASE)


def expand_query(query: str, max_variants: int = 3) -> list[str]:
    """The original query first, then up to max_variants rewrites with one synonym substituted.

    Rewrites are taken from each matched synonym group in turn, so one group cannot use up the budget.
    """
    per_group: list[list[str]] = []
    for group in SYNONYMS:
        # Longest phrase first, so "submission deadline" wins over "deadline".
        for phrase in sorted(group, key=len, reverse=True):
            pattern = _pattern(phrase)
            if pattern.search(query):
                per_group.append([pattern.sub(alt, query, count=1) for alt in group if alt != phrase])
                break
    variants = [query]
    for round_robin in itertools.zip_longest(*per_group):
        for rewrite in round_robin:
            if rewrite and rewrite not in variants:
                variants.append(rewrite)
    return variants[: max_variants + 1]
