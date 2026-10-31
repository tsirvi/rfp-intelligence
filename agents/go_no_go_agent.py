"""Go / no-go agent: recommends whether to bid, from the extracted record and a company capability profile.

Rule-based on purpose: each check states the field, the value and its citation, so the recommendation can be
explained and challenged line by line. Values the validator did not confirm are never grounds for "go"; they
turn the check into "review".

  GO      every check passed
  REVIEW  nothing rules the bid out, but a person must confirm something (listed)
  NO-GO   at least one capability or deadline check fails
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import yaml
from dateutil import parser as dateparser
from pydantic import BaseModel, Field

from agents.records import first_source, record_fields, validated
from agents.tracing import Tracer
from settings import ROOT

Outcome = Literal["pass", "review", "fail", "skip"]


class Check(BaseModel):
    name: str
    outcome: Outcome
    reason: str
    field: str | None = None
    value: str | None = None
    source: str | None = None


class Decision(BaseModel):
    bid_id: str
    decision: Literal["GO", "REVIEW", "NO-GO"]
    company: str
    evaluation_date: str
    generated_at: str
    checks: list[Check]
    summary: str = ""
    counts: dict[str, int] = Field(default_factory=dict)


def load_profile(path: str | Path | None = None) -> dict:
    path = Path(path or os.getenv("COMPANY_PROFILE") or ROOT / "company_profile.yaml")
    return yaml.safe_load(path.read_text("utf-8")) or {}


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_DAYS = re.compile(r"(\d+)\s*(?:\)\s*)?(?:calendar\s+|business\s+|working\s+)?days?", re.IGNORECASE)
_NOT_REQUIRED = re.compile(r"\b(not required|none|no bid bond|not applicable|n/a)\b", re.IGNORECASE)
_ELECTRONIC = re.compile(r"electronic|online|portal|e-?procurement|emma|isupplier|upload|web", re.IGNORECASE)
_PAPER = re.compile(r"hard cop|sealed|by hand|courier|hand[- ]deliver|mail(ed)? to", re.IGNORECASE)
_GENERIC_MFG = re.compile(
    r"\b(the|a|an|or|and|of|by|must|be|manufacturers?|authori[sz]ed|resellers?|dealers?|distributors?|"
    r"registration|registered|approved|equal)\b|[^\w\s]",
    re.IGNORECASE,
)
_EQUAL = re.compile(r"approved equal|or equal|equivalent", re.IGNORECASE)


def _due(value: str) -> date | None:
    try:
        return dateparser.parse(re.sub(r"\s[A-Z]{2,4}$", "", value.strip()), fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


def _contract_years(value: str) -> int | None:
    """'Total max 5 years', or a base term plus renewals ('2 years ... renew for 2 additional 1-year periods')."""
    if m := re.search(r"(?:total|max\w*)\D{0,20}(\d+)\s*years?", value, re.IGNORECASE):
        return int(m[1])
    base = re.search(r"(\d+)\s*(?:\)\s*)?-?\s*years?", value, re.IGNORECASE)
    if not base:
        return None
    renewals = sum(
        int(n) * int(y) for n, y in re.findall(r"(\d+)\s+additional\s+(\d+)\s*-?\s*years?", value, re.IGNORECASE)
    )
    renewals += sum(int(y) for y in re.findall(r"renewal\s*\d*\s*:\s*(\d+)\s*years?", value, re.IGNORECASE))
    return int(base[1]) + renewals


class GoNoGoAgent:
    def __init__(self, tracer: Tracer, profile: dict | None = None):
        self.tracer = tracer
        self.profile = profile if profile is not None else load_profile()

    def assess(self, bid_id: str, outputs_dir: Path) -> Decision:
        with self.tracer.step("go_no_go", "assess", {"bid_id": bid_id, "company": self.profile.get("company")}) as step:
            fields = record_fields(outputs_dir, bid_id)
            if fields is None:
                raise ValueError(f"No extracted record for {bid_id}; run the extraction first")
            on = self.profile.get("evaluation_date")
            today = on if isinstance(on, date) else (dateparser.parse(str(on)).date() if on else date.today())
            checks = [
                self._deadline(fields, today),
                self._products(fields),
                self._manufacturer(fields),
                self._bid_bond(fields),
                self._installation(fields),
                self._submission(fields),
                self._delivery(fields),
                self._contract(fields),
                self._documents(fields),
            ]
            checks = [c for c in checks if c is not None]
            counts = {o: sum(c.outcome == o for c in checks) for o in ("pass", "review", "fail", "skip")}
            decision = "NO-GO" if counts["fail"] else "REVIEW" if counts["review"] else "GO"
            result = Decision(
                bid_id=bid_id,
                decision=decision,
                company=self.profile.get("company", "the company"),
                evaluation_date=today.isoformat(),
                generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                checks=checks,
                counts=counts,
                summary=_summary(decision, checks),
            )
            step.output({"decision": decision, "counts": counts})
            return result

    # ---- checks: each returns None when the profile does not configure it ---------------------------------

    def _deadline(self, fields: dict, today: date) -> Check:
        f, name = fields.get("Due Date"), "Deadline"
        due = _due(f["value"]) if f and f.get("value") else None
        if due is None:
            return Check(name=name, outcome="review", reason="No due date in the record; find it before deciding.")
        days = (due - today).days
        minimum = int(self.profile.get("min_days_to_prepare", 0))
        if not validated(f):
            return _check(name, "review", f"Due date {due} not confirmed by the validator.", "Due Date", f)
        if days < 0:
            return _check(name, "fail", f"Closed {-days} days before {today}.", "Due Date", f)
        if days < minimum:
            return _check(name, "fail", f"Only {days} days left; we need at least {minimum}.", "Due Date", f)
        return _check(name, "pass", f"{days} days left to prepare (minimum {minimum}).", "Due Date", f)

    def _products(self, fields: dict) -> Check | None:
        words = [w.lower() for w in self.profile.get("product_keywords") or []]
        if not words:
            return None
        names = ("Product", "Title", "Bid Summary")
        text = " ".join(str((fields.get(n) or {}).get("value") or "") for n in names).lower()
        hits = sorted({w for w in words if w in text})
        f = fields.get("Product") or fields.get("Title")
        if hits:
            return _check("Product fit", "pass", f"Within our product lines: {', '.join(hits)}.", "Product", f)
        return _check("Product fit", "fail", "None of our product lines appear in the bid.", "Product", f)

    def _manufacturer(self, fields: dict) -> Check | None:
        brands = self.profile.get("authorized_manufacturers") or []
        f = fields.get("MFG for Registration")
        if not brands or not (f and f.get("value")):
            return None
        value = f["value"]
        ours = [b for b in brands if re.search(rf"\b{re.escape(b)}\b", value, re.IGNORECASE)]
        if ours:
            return _check(
                "Manufacturer", "pass", f"We are authorised for {', '.join(ours)}.", "MFG for Registration", f
            )
        if not _GENERIC_MFG.sub("", value).strip():  # "Manufacturer or Authorized Reseller": no brand named
            reason = "Manufacturer authorisation required, but no manufacturer is named; confirm which."
            return _check("Manufacturer", "review", reason, "MFG for Registration", f)
        why = "an approved equal would have to be qualified" if _EQUAL.search(value) else "not in our list"
        return _check(
            "Manufacturer", "review", f"Requires '{value}'; {why}.", "MFG for Registration", f
        )  # fmt: skip

    def _bid_bond(self, fields: dict) -> Check | None:
        limit = self.profile.get("max_bid_bond_percent")
        f, name = fields.get("Bid Bond Requirement"), "Bid bond"
        if limit is None:
            return None
        if not (f and f.get("value")):
            return Check(name=name, outcome="pass", reason="No bid bond stated in the documents.")
        if not validated(f):
            return _check(name, "review", "Bid bond value not confirmed by the validator.", "Bid Bond Requirement", f)
        if _NOT_REQUIRED.search(f["value"]):
            return _check(name, "pass", "Not required.", "Bid Bond Requirement", f)
        if m := _PERCENT.search(f["value"]):
            pct = float(m[1])
            outcome = "pass" if pct <= float(limit) else "fail"
            return _check(name, outcome, f"{pct:g}% required; our limit is {limit}%.", "Bid Bond Requirement", f)
        return _check(name, "review", "Bid bond required; amount not a percentage.", "Bid Bond Requirement", f)

    def _installation(self, fields: dict) -> Check | None:
        provides = self.profile.get("provides_installation")
        f = fields.get("Installation")
        if provides is None or not (f and f.get("value")):
            return None
        required = bool(re.match(r"\s*(yes|required)", f["value"], re.IGNORECASE))
        if not required:
            return _check("Installation", "pass", "No installation work required.", "Installation", f)
        if provides:
            return _check("Installation", "pass", "Installation required; we provide it.", "Installation", f)
        return _check("Installation", "fail", "Installation required; we do not provide it.", "Installation", f)

    def _submission(self, fields: dict) -> Check | None:
        methods = " ".join(self.profile.get("submission_methods") or []).lower()
        f = fields.get("Bid Submission Type")
        if not methods or not (f and f.get("value")):
            return None
        value, name = f["value"], "Submission"
        needs = [k for k, rx in (("electronic", _ELECTRONIC), ("paper", _PAPER)) if rx.search(value)]
        if not needs:
            return _check(
                name, "review", "Submission method unclear; read it before deciding.", "Bid Submission Type", f
            )
        can = {"electronic": bool(_ELECTRONIC.search(methods)), "paper": bool(_PAPER.search(methods))}
        if any(can[n] for n in needs):
            return _check(name, "pass", f"We can submit this way ({', '.join(needs)}).", "Bid Submission Type", f)
        return _check(name, "fail", f"Requires {' or '.join(needs)} submission.", "Bid Submission Type", f)

    def _delivery(self, fields: dict) -> Check | None:
        minimum = self.profile.get("min_delivery_days")
        f = fields.get("Delivery Date")
        if minimum is None or not (f and f.get("value")):
            return None
        if not validated(f):
            return _check(
                "Delivery", "review", "Delivery requirement not confirmed by the validator.", "Delivery Date", f
            )
        if not (m := _DAYS.search(f["value"])):
            return _check(
                "Delivery", "review", "Delivery is a date, not a window; check lead times.", "Delivery Date", f
            )
        days = int(m[1])
        outcome = "pass" if days >= int(minimum) else "review"
        return _check("Delivery", outcome, f"{days} days to deliver; we need {minimum}.", "Delivery Date", f)

    def _contract(self, fields: dict) -> Check | None:
        limit = self.profile.get("max_contract_years")
        f = fields.get("Term of Bid")
        if limit is None or not (f and f.get("value")):
            return None
        years = _contract_years(f["value"])
        if years is None:
            return None
        outcome = "pass" if years <= int(limit) else "review"
        return _check("Contract term", outcome, f"Up to {years} years; we commit to {limit} at most.", "Term of Bid", f)

    def _documents(self, fields: dict) -> Check | None:
        known = [w.lower() for w in self.profile.get("documents_we_provide") or []]
        f = fields.get("Any Additional Documentation Required")
        if not known or not (f and f.get("value")):
            return None
        items = [i.strip() for i in re.split(r";|\n", f["value"]) if i.strip()]
        missing = [i for i in items if not any(k in i.lower() for k in known)]
        name, field = "Required documents", "Any Additional Documentation Required"
        if missing:
            return _check(name, "review", f"Not in our standard set: {'; '.join(missing)}.", field, f)
        return _check(name, "pass", f"We can supply all {len(items)} required documents.", field, f)


def _check(name: str, outcome: Outcome, reason: str, field: str, f: dict | None) -> Check:
    source = first_source(f).label() if f and f.get("sources") else None
    return Check(name=name, outcome=outcome, reason=reason, field=field, value=(f or {}).get("value"), source=source)


def _summary(decision: str, checks: list[Check]) -> str:
    fails = [c for c in checks if c.outcome == "fail"]
    reviews = [c for c in checks if c.outcome == "review"]
    if decision == "NO-GO":
        return "Do not bid: " + " ".join(f"{c.name}: {c.reason}" for c in fails)
    if decision == "REVIEW":
        return "No blocker found; confirm before bidding: " + "; ".join(c.name for c in reviews) + "."
    return "Bid: every check passed."


ICON = {"pass": "✅", "review": "⚠️", "fail": "❌", "skip": "—"}


def to_markdown(d: Decision) -> str:
    lines = [
        f"# Go / no-go: {d.bid_id} — **{d.decision}**",
        "",
        d.summary,
        "",
        (
            f"Assessed for {d.company} as of {d.evaluation_date}. Rule-based, from the extracted record; "
            "informational only, not a substitute for reading the solicitation."
        ),
        "",
        "| Check | Outcome | Reason | Value | Source |",
        "|---|---|---|---|---|",
    ]
    for c in d.checks:
        value = (c.value or "").replace("|", "/").replace("\n", " ")
        value = value if len(value) <= 100 else value[:99] + "…"
        lines.append(f"| {c.name} | {ICON[c.outcome]} {c.outcome} | {c.reason} | {value} | {c.source or ''} |")
    return "\n".join(lines) + "\n"


def write(d: Decision, outputs_dir: Path) -> tuple[Path, Path]:
    out = outputs_dir / "go_no_go"
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / f"{d.bid_id}.json", out / f"{d.bid_id}.md"
    json_path.write_text(json.dumps(d.model_dump(mode="json"), indent=2, ensure_ascii=False), "utf-8")
    md_path.write_text(to_markdown(d), "utf-8")
    return json_path, md_path
