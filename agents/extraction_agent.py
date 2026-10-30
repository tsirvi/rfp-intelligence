"""Extraction agent: fills a group of fields from retrieved evidence only, with a citation for each value."""

from __future__ import annotations

import re
from collections.abc import Sequence

from agents.fields import FieldSpec
from agents.llm import LLM, LLMError
from agents.tracing import Tracer
from agents.validator_agent import support
from schemas.extraction import NOT_FOUND, Evidence, ExtractionOutput, FieldResult

_EVIDENCE_REF = re.compile(r"\bE\d+\b")

# Strings models use for "no value"; these are treated as null, not as a value.
_EMPTY = re.compile(
    r"^(null|none found|n/?a|unknown|not (specified|stated|mentioned|provided|available|found.*))\.?$", re.IGNORECASE
)


def _clean_value(raw: str | None) -> str | None:
    value = _INLINE_IDS.sub("", raw or "").replace("**", "").strip().strip(";,").strip()
    return None if not value or _EMPTY.match(value) else value


_INLINE_IDS = re.compile(r"\s*[\[(]\s*E\d+(\s*,\s*E\d+)*\s*[\])]")


def prune_citations(value: str, cited: list[Evidence]) -> list[Evidence]:
    """Keep the passages that support the value by themselves. Models tend to cite every passage they
    were shown, which would let the validator find a stray "24 months" in an unrelated clause. If no single
    passage supports the value, keep the best one, so the validator judges the value against it."""
    if len(cited) <= 1:
        return cited
    scored = [(support(value, ev.text), ev) for ev in cited]
    strong = [ev for score, ev in scored if score >= 0.5]
    return strong[:3] if strong else [max(scored, key=lambda x: x[0])[1]]


SYSTEM = f"""You extract fields from procurement bid documents (RFPs, addenda, bid pages, specifications).
Rules:
- Use ONLY the evidence passages given. Never use outside knowledge and never guess.
- For each field give: the value, the ids of the passages that state it (e.g. ["E2"]), a confidence from 0 to 1, \
and a short note.
- If the passages do not state a field, set value to null and notes to "{NOT_FOUND}".
- If passages conflict, an addendum overrides the base RFP; the higher addendum number wins. Mention it in notes.
- Do not infer or embellish: report requirements exactly as stated (e.g. "encouraged" is not "mandatory").
- Keep notes to 20 words or fewer.
- Copy numbers, dates, names, part numbers and model numbers exactly as written.
- Dates: write "YYYY-MM-DD HH:MM TZ" (24-hour clock) when the time and zone are given.
- Lists (documents, products, part numbers, specifications): one string, items separated by "; ".
"""


def format_evidence(evidence: Sequence[Evidence]) -> tuple[str, dict[str, Evidence]]:
    ids: dict[str, Evidence] = {}
    blocks = []
    for i, ev in enumerate(evidence, start=1):
        eid = f"E{i}"
        ids[eid] = ev
        kind = ev.doc_type.replace("_", " ") + (f" {ev.addendum_number}" if ev.addendum_number is not None else "")
        blocks.append(f"[{eid}] ({ev.source.label()}; {kind})\n{ev.text}")
    return "\n\n".join(blocks), ids


class ExtractionAgent:
    def __init__(self, llm: LLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer

    def extract(
        self,
        group: str,
        specs: Sequence[FieldSpec],
        evidence: Sequence[Evidence],
        feedback: dict[str, str] | None = None,
        attempt: int = 1,
    ) -> dict[str, FieldResult]:
        names = [s.name for s in specs]
        if not evidence:  # nothing retrieved: answer without calling the model
            return {n: FieldResult(name=n, notes=NOT_FOUND, attempts=attempt, status="not_found") for n in names}

        evidence_text, ids = format_evidence(evidence)
        field_lines = []
        for s in specs:
            line = f"- {s.name}: {s.description}"
            if feedback and s.name in feedback:
                line += f"\n  A previous answer was rejected: {feedback[s.name]} Look again carefully."
            field_lines.append(line)
        prompt = f"Fields to extract:\n{chr(10).join(field_lines)}\n\nEvidence:\n{evidence_text}"

        with self.tracer.step("extraction", f"group:{group}", {"fields": names, "attempt": attempt}) as step:
            try:
                out = self.llm.structured(SYSTEM, prompt, ExtractionOutput, step)
            except LLMError as exc:  # recorded and handed to the validator, which schedules a retry
                step.output({"error": str(exc)})
                return {
                    n: FieldResult(name=n, notes=f"Extraction failed: {exc}", attempts=attempt, status="failed")
                    for n in names
                }

            results: dict[str, FieldResult] = {}
            by_name = {f.name: f for f in out.fields}
            for name in names:
                f = by_name.get(name)
                if f is None:
                    results[name] = FieldResult(name=name, notes=NOT_FOUND, attempts=attempt, status="not_found")
                    continue
                cited_ids = f.evidence_ids or _EVIDENCE_REF.findall(f.notes)  # ids sometimes land in the notes
                cited = [ids[e] for e in dict.fromkeys(cited_ids) if e in ids]
                value = _clean_value(f.value)
                cited = prune_citations(value, cited) if value else []
                results[name] = FieldResult(
                    name=name,
                    value=value,
                    sources=[ev.source for ev in cited],
                    confidence=round(f.confidence, 2),
                    notes=f.notes.strip() or ("" if value else NOT_FOUND),
                    attempts=attempt,
                    status="draft" if value else "not_found",
                )
            step.output({n: {"value": r.value, "sources": [s.label() for s in r.sources]} for n, r in results.items()})
            return results
