"""Bid comparison agent: a side-by-side report of several bids from their extracted records.

The table and the key differences are computed from validated values only, so they are exact and cited.
The LLM adds a short narrative on top, citing table cells by id; without a valid citation it is dropped.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from dateutil import parser as dateparser
from pydantic import BaseModel, Field

from agents.fields import FIELD_NAMES
from agents.llm import LLM, LLMError
from agents.qa_agent import _INLINE_REF
from agents.records import first_source, record_fields, short_label, validated
from agents.tracing import Tracer
from schemas.extraction import NOT_FOUND, AnswerOutput, Source

SYSTEM = """You compare procurement bids for a sales team, using ONLY the table cells given (each has an id).
- Write 3-5 short bullet points (under 120 words in total) on the differences that matter for deciding which
  bid to pursue:
  deadlines, submission method, bonds, delivery, contract terms, products.
- Cite the ids of the cells you use in evidence_ids. Never add facts that are not in the cells.
- Do not repeat the table; point out what differs and why it matters.
"""

UNVALIDATED = "⚠️ extracted value not validated"


class Cell(BaseModel):
    value: str | None
    status: str  # passed | failed | not_found
    source: Source | None = None


class Comparison(BaseModel):
    bids: list[str]
    generated_at: str
    fields: dict[str, dict[str, Cell]]  # field -> bid -> cell
    differences: list[str] = Field(default_factory=list)  # computed, exact
    summary: str | None = None  # written by the LLM, cited
    summary_sources: list[str] = Field(default_factory=list)
    run: dict = Field(default_factory=dict)


def _cell(field: dict | None) -> Cell:
    if validated(field):
        return Cell(value=field["value"], status="passed", source=first_source(field))
    if field and field.get("status") == "failed":
        return Cell(value=field.get("value"), status="failed")
    return Cell(value=None, status="not_found")


def _date(value: str) -> datetime | None:
    try:
        return dateparser.parse(re.sub(r"\b[A-Z]{2,4}$", "", value.strip()), fuzzy=True)
    except (ValueError, OverflowError):
        return None


def differences(fields: dict[str, dict[str, Cell]], bids: list[str]) -> list[str]:
    """Plain, exact statements about what differs, from validated values only."""
    out = []
    due = {b: (c, _date(c.value)) for b, c in fields["Due Date"].items() if c.status == "passed" and c.value}
    dated = sorted(((d, b, c) for b, (c, d) in due.items() if d), key=lambda x: x[0])
    if len(dated) > 1:
        order = ", then ".join(f"{b} ({c.value})" for _, b, c in dated)
        out.append(f"Deadlines: {order}.")
    for name, label in (
        ("Bid Bond Requirement", "Bid bond"),
        ("Bid Submission Type", "Submission"),
        ("Delivery Date", "Delivery"),
        ("Term of Bid", "Contract term"),
        ("Contract or Cooperative to use", "Purchasing vehicle"),
    ):
        stated = {b: c.value for b, c in fields[name].items() if c.status == "passed"}
        if stated:
            silent = [b for b in bids if b not in stated]
            text = "; ".join(f"{b}: {_short(v)}" for b, v in stated.items())
            out.append(f"{label}: {text}" + (f"; not stated for {', '.join(silent)}." if silent else "."))
    unchecked = {b: [n for n in fields if fields[n][b].status == "failed"] for b in bids}
    for bid, names in unchecked.items():
        if names:
            out.append(f"Check by hand for {bid}: {', '.join(names)} (the validator did not confirm these).")
    return out


def _short(value: str, limit: int = 120) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


class ComparisonAgent:
    def __init__(self, llm: LLM | None, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer

    def compare(self, bids: list[str], outputs_dir: Path, summary: bool = True) -> Comparison:
        with self.tracer.step("comparison", "table", {"bids": bids}) as step:
            records = {b: record_fields(outputs_dir, b) for b in bids}
            missing = [b for b, r in records.items() if r is None]
            if missing:
                raise ValueError(f"No extracted record for {', '.join(missing)}; run the extraction first")
            names = [n for n in FIELD_NAMES if n != "Bid Summary"]
            fields = {n: {b: _cell(records[b].get(n)) for b in bids} for n in names}
            result = Comparison(
                bids=bids,
                generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                fields=fields,
                differences=differences(fields, bids),
            )
            step.output({"fields": len(fields), "differences": len(result.differences)})
        if summary and self.llm is not None:
            self._summarise(result)
        return result

    def _summarise(self, result: Comparison) -> None:
        cells, ids = [], {}
        for name, row in result.fields.items():
            for bid, cell in row.items():
                if cell.status == "passed":
                    eid = f"E{len(ids) + 1}"
                    ids[eid] = cell
                    cells.append(f"[{eid}] {bid} | {name}: {_short(cell.value, 200)}")
        with self.tracer.step("comparison", "summary", {"bids": result.bids, "cells": len(cells)}) as step:
            try:
                out = self.llm.structured(
                    SYSTEM, "Table cells:\n" + "\n".join(cells), AnswerOutput, step, max_tokens=400
                )
            except LLMError as exc:
                step.output({"error": str(exc)})
                return
            cited = [e for e in dict.fromkeys(out.evidence_ids) if e in ids]
            if not cited or out.answer.strip().rstrip(".") == NOT_FOUND:
                step.output({"dropped": "summary cited no table cell"})
                return
            # Inline ids become readable citations; the model saw only cell ids.
            result.summary = _INLINE_REF.sub(lambda m: _cite(m[1] or m[2], ids), out.answer).strip()
            result.summary_sources = sorted({ids[e].source.label() for e in cited})
            step.output({"summary": result.summary, "sources": result.summary_sources})


def _cite(eid: str, ids: dict[str, Cell]) -> str:
    return f"[{short_label(ids[eid].source)}]" if eid in ids else ""


def to_markdown(c: Comparison) -> str:
    lines = [
        f"# Bid comparison: {' vs '.join(c.bids)}",
        "",
        f"Generated {c.generated_at} from the extracted records.",
        "",
    ]
    if c.summary:
        lines += ["## Summary (written by the model from the table below)", "", c.summary, ""]
    if c.differences:
        lines += ["## Key differences", "", *[f"- {d}" for d in c.differences], ""]
    lines += ["## Side by side", "", "| Field | " + " | ".join(c.bids) + " |", "|---|" + "---|" * len(c.bids)]
    for name, row in c.fields.items():
        cells = []
        for bid in c.bids:
            cell = row[bid]
            if cell.status == "passed":
                cells.append(f"{_short(cell.value, 160).replace('|', '/')} [{short_label(cell.source)}]")
            else:
                cells.append(UNVALIDATED if cell.status == "failed" else NOT_FOUND)
        lines.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
    sources = sorted({cell.source.label() for row in c.fields.values() for cell in row.values() if cell.source})
    lines += ["", "## Sources", "", *[f"- {s}" for s in sources], ""]
    return "\n".join(lines)


def write(c: Comparison, outputs_dir: Path) -> tuple[Path, Path]:
    out = outputs_dir / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    stem = "_vs_".join(c.bids)
    json_path, md_path = out / f"{stem}.json", out / f"{stem}.md"
    json_path.write_text(json.dumps(c.model_dump(mode="json"), indent=2, ensure_ascii=False), "utf-8")
    md_path.write_text(to_markdown(c), "utf-8")
    return json_path, md_path
