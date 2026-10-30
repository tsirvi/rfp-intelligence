"""Report agent: writes the bid summary from validated fields, then the JSON record and a readable report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agents.fields import FIELD_NAMES
from agents.llm import LLM, LLMError
from agents.tracing import Tracer
from schemas.extraction import (
    NOT_FOUND,
    AddendumChange,
    BidRecord,
    FieldOut,
    FieldResult,
    SummaryOutput,
    ValidationSummary,
)

SUMMARY_SYSTEM = """Write a concise summary (3-6 sentences) of a procurement bid for a sales team, using ONLY the \
extracted fields given. Cover: who is buying what, the deadline, how to submit, and notable requirements or \
addendum changes. Do not add facts that are not in the fields. Plain prose, no lists."""

SUMMARY_INPUTS = [
    "Title", "company_name", "Bid Number", "Product", "Due Date", "Bid Submission Type", "Term of Bid",
    "Delivery Date", "Pre Bid Meeting", "Bid Bond Requirement", "Installation", "Contract or Cooperative to use",
]  # fmt: skip


class ReportAgent:
    def __init__(self, llm: LLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer

    def summarise(self, bid_id: str, fields: dict[str, FieldResult], changes: list[AddendumChange]) -> FieldResult:
        # Only validated fields: a rejected value must not resurface as a fact in the summary.
        known = [fields[n] for n in SUMMARY_INPUTS if n in fields and fields[n].value and fields[n].status == "passed"]
        if not known:
            return FieldResult(name="Bid Summary", notes=NOT_FOUND, status="not_found")
        lines = [f"- {f.name}: {f.value}" for f in known]
        lines += [f"- {c.addendum}: {c.description}" for c in changes if c.applied and c.field != "Other"][:5]
        with self.tracer.step("report", "summary", {"bid_id": bid_id, "fields": [f.name for f in known]}) as step:
            try:
                out = self.llm.structured(SUMMARY_SYSTEM, f"Bid {bid_id}:\n" + "\n".join(lines), SummaryOutput, step)
            except LLMError as exc:
                step.output({"error": str(exc)})
                return FieldResult(name="Bid Summary", notes=f"Summary failed: {exc}", status="failed")
            step.output(out.summary)
        sources = list({s.label(): s for f in known for s in f.sources[:1]}.values())
        return FieldResult(
            name="Bid Summary",
            value=out.summary.strip(),
            sources=sources,
            confidence=round(min(f.confidence for f in known), 2),
            notes="Generated from the extracted fields above",
            status="passed",
        )

    @staticmethod
    def record(
        bid_id: str,
        model: str,
        fields: dict[str, FieldResult],
        changes: list[AddendumChange],
        validation: ValidationSummary,
        run: dict,
    ) -> BidRecord:
        out: dict[str, FieldOut] = {}
        for name in FIELD_NAMES:
            f = fields.get(name) or FieldResult(name=name, notes=NOT_FOUND, status="not_found")
            confidence = f.confidence
            notes = f.notes or (NOT_FOUND if f.value is None else "")
            if f.status == "failed":
                # Kept for review rather than dropped, but flagged and down-weighted.
                confidence = min(confidence, 0.3)
                problems = [i.message for i in validation.issues if i.field == name]
                notes = f"{notes}; validation failed: {'; '.join(problems)}".strip("; ")
            elif f.value is None and not f.sources:
                confidence = max(confidence, 0.5)  # confidence that it is absent, not in a value
            out[name] = FieldOut(
                value=f.value,
                sources=[
                    {"file": s.file, "page": s.page}
                    | ({"page_end": s.page_end} if s.page_end and s.page_end != s.page else {})
                    for s in f.sources
                ],
                confidence=round(confidence, 2),
                notes=notes,
                status="not_found" if f.value is None else ("failed" if f.status == "failed" else "passed"),
            )
        return BidRecord(
            bid_id=bid_id,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            model=model,
            fields=out,
            addendum_changes=changes,
            validation=validation,
            run=run,
        )

    @staticmethod
    def write(record: BidRecord, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{record.bid_id}.json"
        json_path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False), "utf-8")
        md_path = out_dir / f"{record.bid_id}.md"
        md_path.write_text(to_markdown(record), "utf-8")
        return json_path, md_path


def to_markdown(record: BidRecord) -> str:
    v = record.validation
    lines = [
        f"# {record.bid_id}",
        "",
        record.fields["Bid Summary"].value or "",
        "",
        f"Validation: {v.passed} passed, {v.failed} failed, {v.not_found} not found. Model: {record.model}.",
        "",
        "| Field | Value | Sources | Confidence | Notes |",
        "|---|---|---|---|---|",
    ]
    for name, f in record.fields.items():
        if name == "Bid Summary":
            continue
        cell = lambda s: (s or "").replace("|", "/").replace("\n", " ")
        src = "; ".join(f"{s['file']} p.{s['page']}" for s in f.sources)
        lines.append(f"| {name} | {cell(f.value) or '—'} | {cell(src)} | {f.confidence:.2f} | {cell(f.notes)} |")
    if record.addendum_changes:
        lines += ["", "## Addendum changes", ""]
        for c in record.addendum_changes:
            lines.append(f"- **{c.addendum}** ({c.change_type}, {c.field}): {c.description}")
    return "\n".join(lines) + "\n"
