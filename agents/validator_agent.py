"""Validator / critic: checks every field against its cited evidence, without calling the LLM.

Checks
- citation:    a value must cite at least one passage.
- grounding:   every date and number in the value must appear in the cited passages, and most of its
               words too; this is the hallucination check.
- format:      date fields must contain a parseable date (the due date also a time).
- consistency: a field changed by an addendum must carry the addendum's value; the due date cannot be
               earlier than the solicitation's issue date.
- not_found:   fields almost every solicitation states (see FieldSpec.expected) are retried once.
Failed fields go back to the orchestrator for re-retrieval and re-extraction.
"""

from __future__ import annotations

import re
from datetime import date

from dateutil import parser as dateparser

from agents.fields import BY_NAME
from agents.tracing import Tracer
from schemas.extraction import AddendumChange, Evidence, FieldResult, ValidationIssue, ValidationSummary

_DATE_TEXT = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}-[a-z]{3}-\d{4}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4}\b",
    re.IGNORECASE,
)
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z]{4,}")
_STOP = frozenset(
    [
        "with",
        "from",
        "that",
        "this",
        "have",
        "will",
        "must",
        "shall",
        "such",
        "each",
        "other",
        "their",
        "there",
        "which",
        "into",
        "only",
        "also",
    ]
)


def _dates(text: str) -> set[date]:
    found = set()
    for m in _DATE_TEXT.finditer(text):
        try:
            found.add(dateparser.parse(m.group(0), fuzzy=True).date())
        except (ValueError, OverflowError):
            continue
    return found


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "").replace("|", " ")).lower()


_HEDGE = re.compile(r"\b(implied|implies|likely|probably|presumably|assumed|inferred|may be required)\b", re.IGNORECASE)
_NEGATIVE_VALUE = re.compile(r"^(none|no|not required|not applicable|no .{0,40} required)\.?$", re.IGNORECASE)
_NEGATION_IN_TEXT = re.compile(
    r"\b(not required|none|n/a|not applicable|waived|no (bid )?bond|no pre-?(bid|proposal))\b", re.IGNORECASE
)


def grounding_problems(value: str, evidence_text: str) -> list[str]:
    problems = []
    if _HEDGE.search(value):
        problems.append("value is hedged or inferred, not stated in the documents")
    if _NEGATIVE_VALUE.match(value.strip()) and not _NEGATION_IN_TEXT.search(evidence_text):
        problems.append(f"'{value}' is not stated in the cited passages; use null if the documents are silent")
    value_dates = _dates(value)
    missing = value_dates - _dates(evidence_text)
    if missing:
        problems.append(f"date(s) {', '.join(sorted(d.isoformat() for d in missing))} not in the cited passages")

    rest = _TIME.sub(" ", _DATE_TEXT.sub(" ", value))
    haystack = _norm(evidence_text)
    numbers = {n.replace(",", "") for n in _NUMBER.findall(rest)}
    absent = sorted(n for n in numbers if len(n) > 1 and n not in haystack)
    if absent:
        problems.append(f"number(s) {', '.join(absent[:5])} not in the cited passages")

    words = [w for w in _WORD.findall(rest.lower()) if w not in _STOP]
    if len(words) >= 3:
        found = sum(w in haystack for w in words)
        if found / len(words) < 0.4:
            problems.append(f"only {found}/{len(words)} words appear in the cited passages")
    return problems


def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower() or (bool(_dates(a)) and _dates(a) == _dates(b))


def support(value: str, text: str) -> float:
    """How well one passage supports a value: 0 if it fails grounding, else the share of the value's
    words and numbers that appear in it."""
    if grounding_problems(value, text):
        return 0.0
    haystack = _norm(text)
    terms = [t for t in re.findall(r"[a-z]{3,}|\d+", _norm(value)) if t not in _STOP]
    return sum(t in haystack for t in terms) / len(terms) if terms else 0.5


def _cross_field(f: FieldResult, fields: dict[str, FieldResult]) -> list[ValidationIssue]:
    """Consistency between fields."""
    due = fields.get("Due Date")
    if f.name == "Delivery Date" and f.value and due and due.value:
        delivery_dates, due_dates = _dates(f.value), _dates(due.value)
        if delivery_dates and delivery_dates <= due_dates:
            return [
                ValidationIssue(
                    field=f.name,
                    check="consistency",
                    message="delivery date equals the proposal due date; that is a submission date, not a delivery "
                    "requirement (look for a delivery window such as 'within N days of award')",
                )
            ]
    return []


class ValidatorAgent:
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def validate(
        self,
        fields: dict[str, FieldResult],
        evidence: dict[str, Evidence],
        changes: list[AddendumChange],
        issue_date: date | None = None,
    ) -> ValidationSummary:
        summary = ValidationSummary()
        with self.tracer.step("validator", "validate", {"fields": len(fields)}) as step:
            for name, f in fields.items():
                spec = BY_NAME[name]
                if spec.kind == "summary":
                    continue
                issues = self._check(f, spec.kind, spec.expected, evidence, changes, issue_date)
                issues += _cross_field(f, fields)
                if f.value is None and not any(i.check != "not_found" for i in issues):
                    f.status = "not_found"
                    summary.not_found += 1
                    summary.issues += issues  # an expected field not found is reported, and retried
                elif issues:
                    f.status = "failed"
                    summary.failed += 1
                    summary.issues += issues
                else:
                    f.status = "passed"
                    summary.passed += 1
            step.output(
                {
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "not_found": summary.not_found,
                    "issues": [i.model_dump() for i in summary.issues],
                }
            )
        return summary

    @staticmethod
    def _check(f, kind, expected, evidence, changes, issue_date) -> list[ValidationIssue]:
        name = f.name
        if f.value is None:
            if f.notes.startswith("Extraction failed"):
                return [ValidationIssue(field=name, check="error", message=f.notes)]
            return (
                [ValidationIssue(field=name, check="not_found", message="expected field not found")] if expected else []
            )

        issues = []
        cited = [evidence[s.chunk_id].text for s in f.sources if s.chunk_id in evidence]
        if not f.sources:
            issues.append(ValidationIssue(field=name, check="citation", message="value has no citation"))
        elif cited:
            for problem in grounding_problems(f.value, "\n".join(cited)):
                issues.append(ValidationIssue(field=name, check="grounding", message=problem))

        if kind == "date" and f.value.strip().lower() != "none":
            value_dates = _dates(f.value)
            if not value_dates:
                issues.append(ValidationIssue(field=name, check="format", message="no parseable date"))
            elif name == "Due Date" and not _TIME.search(f.value):
                issues.append(ValidationIssue(field=name, check="format", message="due date has no time"))
            elif name == "Due Date" and issue_date and max(value_dates) < issue_date:
                issues.append(
                    ValidationIssue(
                        field=name, check="consistency", message=f"due date is before issue date {issue_date}"
                    )
                )

        for c in changes:
            if (
                c.field == name
                and c.change_type == "modifies"
                and c.applied
                and c.new_value
                and BY_NAME[name].kind in ("text", "date")
            ):
                latest = max(
                    (x for x in changes if x.field == name and x.change_type == "modifies" and x.new_value),
                    key=lambda x: x.addendum_number or 0,
                )
                if not _same(f.value, latest.new_value):
                    issues.append(
                        ValidationIssue(
                            field=name,
                            check="consistency",
                            message=f"does not reflect {latest.addendum}: {latest.new_value}",
                        )
                    )
                break
        return issues
