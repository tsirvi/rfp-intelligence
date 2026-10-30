"""The 20 fields of the bid record (Part D): meaning, specialist group and retrieval queries.

Queries use generic procurement wording, not anything from the sample bids, so a new bid folder works
without code changes. Groups are extracted by separate specialists in parallel; keeping each group small
keeps every LLM prompt short, which matters on a CPU-only model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["text", "date", "list", "summary"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: str
    description: str
    queries: tuple[str, ...]
    kind: Kind = "text"
    expected: bool = False  # nearly every solicitation states it, so "not found" is retried
    doc_types: tuple[str, ...] = field(default=())  # document types searched first, when the bid has them


FIELDS: tuple[FieldSpec, ...] = (
    # Identity
    FieldSpec(
        "Bid Number",
        "identity",
        "Official solicitation / RFP identifier (e.g. JA-207652).",
        ("solicitation number RFP number bid number", "reference number project number"),
        expected=True,
    ),
    FieldSpec(
        "Title",
        "identity",
        "Full title of the bid or RFP.",
        ("title of the request for proposal solicitation",),
        expected=True,
    ),
    FieldSpec(
        "company_name",
        "identity",
        "Issuing organization / agency name.",
        ("issuing organization agency name", "purchasing department of the district or agency"),
        expected=True,
    ),
    FieldSpec(
        "contact_info",
        "identity",
        "Procurement contact: name, email, phone.",
        ("procurement contact buyer name email phone", "point of contact procurement officer"),
        expected=True,
    ),
    # Dates & logistics
    FieldSpec(
        "Due Date",
        "dates",
        "Final submission deadline, including time and time zone, after any addendum changes. "
        "Format: YYYY-MM-DD HH:MM TZ.",
        ("proposal due date and time submission deadline", "closing date", "addendum new due date extended"),
        kind="date",
        expected=True,
    ),
    FieldSpec(
        "Pre Bid Meeting",
        "dates",
        'Date, time, location and whether attendance is mandatory; "None" if the documents say there is none.',
        ("pre-bid pre-proposal conference meeting date time location", "mandatory site visit attendance"),
    ),
    FieldSpec(
        "Delivery Date",
        "dates",
        "Required delivery date or delivery window after award (not the proposal due date).",
        ("delivery date delivery within days of award", "lead time ship date delivery schedule"),
    ),
    # Submission & terms
    FieldSpec(
        "Bid Submission Type",
        "terms",
        "How the bid is submitted (electronic portal, sealed paper, email, etc.).",
        ("how to submit proposals electronic portal online", "sealed bids hand delivery email not accepted"),
        expected=True,
    ),
    FieldSpec(
        "Term of Bid",
        "terms",
        "Contract duration and any renewal options.",
        ("contract term length of contract renewal options", "initial term extension years"),
    ),
    FieldSpec(
        "Payment Terms",
        "terms",
        "Terms on which the issuing agency pays the vendor (e.g. Net 30) and invoicing conditions; "
        "not a vendor's own terms, nor payments between prime contractors and subcontractors.",
        ("payment terms net 30 days payment", "invoice invoicing instructions"),
    ),
    # Compliance & commercial
    FieldSpec(
        "Bid Bond Requirement",
        "compliance",
        "Bid bond or bid security required to submit a proposal, with amount or percentage; "
        '"Not required" only if the documents say so. Not performance, payment or joint-venture bonds.',
        ("bid bond bid security proposal guarantee amount percentage", "bonds insurance requirements"),
    ),
    FieldSpec(
        "Any Additional Documentation Required",
        "compliance",
        "Affidavits, forms, certificates, insurance, W-9, etc. the bidder must submit.",
        ("required forms affidavits certificates to submit with the proposal", "W-9 insurance certificate form"),
        kind="list",
    ),
    FieldSpec(
        "Contract or Cooperative to use",
        "compliance",
        "Purchasing cooperative, interlocal agreement or state master contract the purchase is made under "
        "or open to, if any.",
        ("cooperative purchasing interlocal agreement", "master contract state contract vehicle eligible"),
    ),
    FieldSpec(
        "MFG for Registration",
        "compliance",
        "Manufacturer(s) the bidder must be registered / authorized with.",
        ("authorized reseller of the manufacturer", "manufacturer name registration"),
    ),
    # Product
    FieldSpec(
        "Product",
        "product",
        "Every product or line item requested, with quantities; include all categories, not just the first.",
        ("products to purchase description quantity", "line items quantity unit"),
        kind="list",
        expected=True,
    ),
    FieldSpec(
        "Model_no",
        "product",
        "Model number(s) of the requested product(s), e.g. a laptop model; not component names such as the CPU.",
        ("model number product model",),
        kind="list",
        doc_types=("specs",),
    ),
    FieldSpec(
        "Part_no",
        "product",
        "Part / SKU number(s) of the requested product(s).",
        ("part number SKU", "description SKU list"),
        kind="list",
        doc_types=("specs",),
    ),
    FieldSpec(
        "Product Specification",
        "specs",
        "Key technical specifications (CPU, RAM, storage, display, warranty, etc.).",
        ("technical specifications processor memory storage display", "warranty requirements years"),
        kind="list",
        doc_types=("specs",),
    ),
    FieldSpec(
        "Installation",
        "specs",
        "Whether installation / deployment / imaging services are required.",
        ("installation deployment imaging setup services", "asset tagging etching delivery to locations"),
    ),
    # Generated by the report agent from the other fields
    FieldSpec("Bid Summary", "summary", "A concise (3-6 sentence) summary of the bid.", (), kind="summary"),
)

FIELD_NAMES = [f.name for f in FIELDS]
BY_NAME = {f.name: f for f in FIELDS}
GROUPS: dict[str, list[FieldSpec]] = {}
for _spec in FIELDS:
    if _spec.kind != "summary":
        GROUPS.setdefault(_spec.group, []).append(_spec)

# Question wording that a field answers directly, for the Q&A agent's instant path (answer from the validated
# record, no LLM). Deliberately narrow and generic: a miss only costs time (the question goes to the LLM),
# a false match gives a wrong answer, so ambiguous wording is left out.
ASK_PATTERNS: dict[str, str] = {
    "Bid Number": r"\b(rfp|bid|solicitation|sourcing|reference|project)\s*(number|no\.?|#|id)\b",
    "Title": r"\btitle\b",
    "company_name": r"\bwho (issued|is issuing|is the issuer)\b|\bissu(ing|er) (organi[sz]ation|agency)\b",
    "contact_info": r"\b(contact (person|details|info\w*|for)|(procurement|purchasing) contact|point of contact"
    r"|who (do|should) (i|we) contact|buyer|procurement officer|purchasing agent)\b",
    "Due Date": r"^(?!.*\bquestions?\b).*\b(due|deadline|closing date)\b",
    "Pre Bid Meeting": r"\bpre[- ]?(bid|proposal|submission)\b|\bsite visit\b|\bwalk[- ]?through\b",
    "Delivery Date": r"^(?!.*\b(address|location|site)s?\b).*\bdeliver(y|ed)\b",
    "Bid Submission Type": r"\bhow (do|can|should|to|must) (i |we |vendors? |bidders? )?submit\b|\bsubmission (method|type)\b",
    "Term of Bid": r"\bcontract (term|length|duration|period)\b|\bhow long is the contract\b|\brenewals?\b",
    "Payment Terms": r"\bpayment terms?\b|\bnet \d+\b|\bhow (will|are) (we|vendors?) (be )?paid\b",
    "Bid Bond Requirement": r"\bbid (bond|security|guarantee)\b|\bproposal guarantee\b",
    "Any Additional Documentation Required": r"\baffidavits?\b|\b(required|additional) (documents|documentation|forms)\b|\bw-?9\b",
    "Contract or Cooperative to use": r"\bco-?operative\b|\bco-?op\b|\binterlocal\b|\bpiggyback\b|\bmaster contract\b",
    "MFG for Registration": r"\bmanufacturers?\b|\boem\b|\bauthori[sz]ed reseller\b",
    "Product": r"\bwhat (products?|items?|equipment|devices?)\b|\bline items?\b|\bquantit(y|ies)\b",
    "Model_no": r"\bmodel (number|no\.?|#)s?\b|\b(which|what) models?\b",
    "Part_no": r"\bpart (number|no\.?|#)s?\b|\bskus?\b",
    "Product Specification": r"\bspec(ification)?s\b|\b(ram|memory|storage|processor|cpu)\b",
    "Installation": r"\binstall(ation|ed)?\b|\bimaging\b|\basset tag",
}
