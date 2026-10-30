"""Addendum reconciliation agent: finds what each addendum changes and applies the latest change per field."""

from __future__ import annotations

import re

from agents.extraction_agent import format_evidence
from agents.fields import BY_NAME, FIELD_NAMES, FIELDS
from agents.llm import LLM, LLMError
from agents.retrieval_agent import RetrievalAgent
from agents.tracing import Tracer
from agents.validator_agent import _dates
from schemas.documents import DocType, Document
from schemas.extraction import AddendumChange, ChangeLogOutput, Evidence, FieldResult

SYSTEM = """You compare an addendum with the original solicitation it amends.
List each change the addendum makes that affects one of the listed fields. Use ONLY the passages given.
- change_type "modifies": the addendum replaces a stated value (e.g. a new due date). Give new_value exactly as \
written, and old_value if the addendum states it.
- change_type "clarifies": the addendum answers a question or adds detail without replacing a value.
- field: one of the listed field names, or "Other" for changes that affect none of them.
- Report only what the addendum actually states. Never list a field the addendum does not change or answer \
(no entries like "does not mention"). Return an empty list if nothing relevant changes.
- One entry per change; do not repeat the same change.
"""


class AddendumAgent:
    def __init__(self, llm: LLM, retrieval: RetrievalAgent, tracer: Tracer):
        self.llm = llm
        self.retrieval = retrieval
        self.tracer = tracer

    def change_log(self, addenda: list[Document]) -> tuple[list[AddendumChange], dict[str, Evidence]]:
        """Returns the changes and the passages they cite (the validator checks values against these)."""
        changes: list[AddendumChange] = []
        pool: dict[str, Evidence] = {}
        field_list = "\n".join(f"- {f.name}: {f.description}" for f in FIELDS if f.kind != "summary")
        queries = [q for f in FIELDS for q in f.queries]
        for doc in sorted(addenda, key=lambda d: d.addendum_number or 0):
            # The search engine is the tool: only passages of this addendum that touch the fields.
            evidence = self.retrieval.gather(queries, {"doc_id": doc.doc_id}, f"addendum:{doc.file_name}", k=2)
            if not evidence:
                continue
            pool.update({e.chunk_id: e for e in evidence})
            text, ids = format_evidence(evidence)
            prompt = f"Fields:\n{field_list}\n\nAddendum passages ({doc.file_name}):\n{text}"
            with self.tracer.step("addendum", f"changes:{doc.file_name}", {"passages": len(evidence)}) as step:
                try:
                    out = self.llm.structured(SYSTEM, prompt, ChangeLogOutput, step)
                except LLMError as exc:
                    step.output({"error": str(exc)})  # this addendum is skipped; others still run
                    continue
                seen: set[tuple[str, str]] = set()
                for c in out.changes:
                    name = _canonical(c.field)
                    key = (name or "Other", c.description.strip().lower())
                    if _NO_CHANGE.search(c.description) or key in seen:
                        continue  # models still emit "the addendum does not change X" and duplicates
                    seen.add(key)
                    # A "modification" without a new value is a clarification.
                    change_type = c.change_type if c.new_value and c.new_value.strip() else "clarifies"
                    changes.append(
                        AddendumChange(
                            addendum=doc.file_name,
                            addendum_number=doc.addendum_number,
                            field=name or "Other",
                            change_type=change_type,
                            description=c.description.strip(),
                            old_value=c.old_value,
                            new_value=c.new_value,
                            sources=[ids[e].source for e in dict.fromkeys(c.evidence_ids) if e in ids],
                        )
                    )
                step.output([c.model_dump(exclude={"sources"}) for c in changes if c.addendum == doc.file_name])
        return changes, pool


_NO_CHANGE = re.compile(
    r"\b(does not|doesn't|did not|no) (explicitly )?(change|mention|state|specify|modify|affect)", re.IGNORECASE
)


def _canonical(name: str) -> str | None:
    lowered = name.strip().lower()
    return next((n for n in FIELD_NAMES if n.lower() == lowered), None)


def apply_changes(fields: dict[str, FieldResult], changes: list[AddendumChange]) -> None:
    """Apply changes in addendum order, so the latest addendum wins. Idempotent: safe to call after a retry.

    Only single-value fields are replaced; for list fields a change is recorded as a clarification, so one
    Q&A answer cannot wipe out a whole specification list.
    """
    for change in sorted(changes, key=lambda c: c.addendum_number or 0):
        current = fields.get(change.field)
        if current is None or not change.sources:
            continue
        label = f"Addendum {change.addendum_number}" if change.addendum_number is not None else change.addendum
        replaces = (
            change.change_type == "modifies" and change.new_value and BY_NAME[change.field].kind in ("text", "date")
        )
        if replaces:
            if same_value(current.value, change.new_value):
                # Already extracted (e.g. normalised "2024-07-09 14:00 CST"): credit the addendum.
                current.sources = change.sources + [s for s in current.sources if s not in change.sources]
                if label not in current.notes:
                    current.notes = f"Per {label}" + (f" (was: {change.old_value})" if change.old_value else "")
            else:
                previous = current.value
                current.value = change.new_value
                current.sources = change.sources + [s for s in current.sources if s not in change.sources]
                current.confidence = max(current.confidence, 0.85)
                current.status = "draft"
                was = change.old_value or previous
                current.notes = f"Changed by {label}" + (f" (was: {was})" if was else "")
            change.applied = True
        # Clarifications stay in the change log only; copying them into every field's notes buried the
        # extracted values under Q&A text.


def same_value(a: str | None, b: str | None) -> bool:
    """Equal as text, or the same dates written differently ("2024-07-09 14:00" vs "July 9, 2024 2:00 PM")."""
    if not a or not b:
        return False
    if a.strip().lower() == b.strip().lower():
        return True
    da, db = _dates(a), _dates(b)
    return bool(da) and da == db


def addenda_of(documents: list[Document]) -> list[Document]:
    return [d for d in documents if d.doc_type == DocType.ADDENDUM]
