"""Q&A agent: answers a question over one or more bids with citations, or says "Not found in documents"."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agents.extraction_agent import _EVIDENCE_REF, format_evidence
from agents.fields import ASK_PATTERNS
from agents.llm import LLM, LLMError, LLMTimeout
from agents.records import record_fields, short_label, validated
from agents.tracing import Tracer
from schemas.documents import DocType
from schemas.extraction import NOT_FOUND, AnswerOutput, Evidence, Source
from search.store import IndexStore
from settings import section

SYSTEM = f"""You answer questions about procurement bids using ONLY the evidence passages given.
- Cite the ids of every passage you rely on in evidence_ids (e.g. ["E1", "E4"]).
- If the evidence does not answer the question, answer exactly "{NOT_FOUND}". Never guess.
- An addendum overrides the base RFP; the higher addendum number wins. Say so when it matters.
- When comparing bids, answer for each bid separately and name the bid. Only say bids are the same or
  different when the evidence states both values. Leave out details the question did not ask about.
- Be concise: at most 3 sentences. Quote dates, amounts and part numbers exactly.
"""

_STOP = frozenset(
    [
        "what",
        "which",
        "when",
        "where",
        "does",
        "with",
        "from",
        "that",
        "this",
        "have",
        "will",
        "must",
        "bid",
        "bids",
        "the",
        "for",
        "and",
        "are",
        "about",
        "there",
        "how",
        "much",
        "many",
        "who",
        "why",
        "any",
        "can",
        "should",
        "need",
        "your",
        "their",
    ]
)
_CHANGES = re.compile(r"\b(addend\w*|chang\w*|clarif\w*|amend\w*|revis\w*)", re.IGNORECASE)
# Across bids: "both bids", "all the bids", "compare the bids", "how do the bids differ". Not "Addendum 2
# compared to the original RFP", which compares documents within one bid.
_COMPARE = re.compile(
    r"\b(both|each|all|two|three|across)\b[^.?]*\bbids?\b|\b(compare|comparison)\b[^.?]*\bbids\b"
    r"|\bbids\b[^.?]*\b(compare|differ\w*|versus|vs)\b",
    re.IGNORECASE,
)
_GENERIC = frozenset(
    [
        "rfp",
        "rfq",
        "ifb",
        "request",
        "proposal",
        "solicitation",
        "addendum",
        "addenda",
        "original",
        "document",
        "sample",
        "contract",
        "purchase",
        "district",
        "order",
    ]
)
_ADDENDUM_NO = re.compile(r"\baddend\w*\s*(?:no\.?\s*|number\s*|#\s*)?(\d+)", re.IGNORECASE)


# "[E3]", "(E3)" or a bare "E3" in the answer text: the prompt's numbering, replaced by readable citations.
_INLINE_REF = re.compile(r"[\[(]\s*(E\d+)\s*[\])]|\b(E\d+)\b")


def _cite(match: re.Match, ids: dict[str, Evidence]) -> str:
    eid = match.group(1) or match.group(2)
    return f"[{ids[eid].source.label()}]" if eid in ids else ""


class QAPlan(BaseModel):
    question: str
    bids: list[str]
    compare: bool = False
    reason: str = ""
    title_words: int = 0  # how many question words matched the chosen bid's title


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Source] = Field(default_factory=list)
    bids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    run_id: str | None = None
    # record: instant answer from the validated extraction record; llm: written by the model from passages;
    # passages: the model ran out of time or failed, so the most relevant cited sentences are shown instead;
    # chat: a help text for greetings
    mode: Literal["record", "llm", "passages", "chat"] = "llm"
    cached: bool = False  # served from the semantic cache (a stored answer to an equivalent question)
    cache_similarity: float | None = None
    usage: dict = Field(default_factory=dict)  # this run's ledger entry: seconds, tokens, LLM calls, est. cost


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def plan_question(question: str, store: IndexStore) -> QAPlan:
    """Decide which bids a question is about: explicit ids first, then words from the bid titles."""
    bids = store.bids()
    # "bid 1" means Bid1; ids must match whole words, so "Bid1" is not found inside "Bid10"
    text = re.sub(r"(?<=[a-z])\s+(?=\d)", "", question.lower())
    named = [b for b in bids if re.search(r"\b" + re.escape("".join(b.lower().split())) + r"\b", text)]
    compare = len(named) > 1 or bool(_COMPARE.search(question))
    if named:
        return QAPlan(question=question, bids=named, compare=compare or len(named) > 1, reason="bid id in question")
    if compare or len(bids) <= 1:
        return QAPlan(question=question, bids=bids, compare=compare, reason="all bids")

    # Words every solicitation shares say nothing about which bid is meant.
    words = {_singular(w) for w in re.findall(r"[a-z0-9]{3,}", question.lower()) if w not in _STOP} - _GENERIC
    scores = {}
    for bid in bids:
        titles = " ".join(
            filter(None, (d.title for d in store.documents(bid) if d.doc_type in (DocType.RFP, DocType.BID_PAGE)))
        )
        scores[bid] = len(words & {_singular(w) for w in re.findall(r"[a-z0-9]{3,}", titles.lower())})
    best = max(scores.values())
    if best == 0:
        return QAPlan(question=question, bids=bids, reason="no bid named; searching all")
    chosen = [b for b, s in scores.items() if s == best]
    return QAPlan(question=question, bids=chosen, reason=f"matched bid title words ({best})", title_words=best)


def document_evidence(bid_id: str, store: IndexStore) -> list[Evidence]:
    """One line per document (type, title, pages), so "what are these files about?" has an answer."""
    out = []
    for d in store.documents(bid_id):
        kind = d.doc_type.value + (f" {d.addendum_number}" if d.addendum_number else "")
        title = (d.title or "").replace("**", "").strip()
        out.append(
            Evidence(
                chunk_id=f"doc:{bid_id}:{d.file_name}",
                text=f"{bid_id} document: {d.file_name} ({kind}, {len(d.pages)} page{'s' if len(d.pages) != 1 else ''})"
                + (f". Title: {title}" if title else ""),
                source=Source(file=d.file_name, page=1),
                doc_type=d.doc_type.value,
                addendum_number=d.addendum_number,
            )
        )
    return out


_ASK = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in ASK_PATTERNS.items()}
_WORD = re.compile(r"[a-z0-9][a-z0-9-]{2,}")
_OVERVIEW = re.compile(
    r"\b(files?|documents?|folders?|bids?|rfps?|tenders?|this|it)\b.*\babout\b|\bwhat (files|documents)\b"
    r"|\b(overview|summary|summari[sz]e|gist)\b|\btell me about\b|\b(details|information|info) (of|on|about)\b"
    r"|\b(major|main|key|important|basic|general|overall)\s+(data|details?|info\w*|points?|facts?|highlights?)\b",
    re.IGNORECASE,
)
_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|hiya|thanks?|thank you|ok(ay)?|cool|great|good (morning|afternoon|evening))\b[\s!.?]*$",
    re.IGNORECASE,
)
HELP = (
    "Hi! Ask me about the indexed bids, for example their due dates, required documents, bid bonds, "
    "products or what an addendum changed. Every answer cites the document and page it comes from."
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def match_fields(question: str, limit: int = 3) -> list[str]:
    """Record fields the question asks for directly (see fields.ASK_PATTERNS); [] if none or too many."""
    names = [name for name, pattern in _ASK.items() if pattern.search(question)]
    return names if len(names) <= limit else []


def record_answer(plan: QAPlan, outputs_dir: Path) -> Answer | None:
    """Tier 1: answer from the validated record, without the LLM.

    Only when every bid in scope has a validator-passed value for every field asked about; otherwise the
    question goes to the LLM, so a rejected or missing extraction never becomes an instant answer.
    """
    names = match_fields(plan.question)
    if not names or not plan.bids:
        return None
    lines, sources, confidences = [], {}, []
    for bid in plan.bids:
        fields = record_fields(outputs_dir, bid)
        if fields is None:
            return None
        for name in names:
            f = fields.get(name)
            if not validated(f):
                return None
            cited = [Source(**s) for s in f["sources"][:3]]
            sources.update({c.label(): c for c in cited})
            confidences.append(f["confidence"])
            notes = f.get("notes") or ""
            # Extraction notes that refer to prompt evidence ids ("E5 lists…") are internal, not for users.
            note = f" ({notes})" if notes and len(notes) <= 120 and not _INLINE_REF.search(notes) else ""
            label = f"{bid} · {name}"
            lines.append(
                f"**{label.replace('_', ' ')}:** {f['value']}{note} " + " ".join(f"[{c.label()}]" for c in cited)
            )
    text = lines[0] if len(lines) == 1 else "\n".join(f"- {line}" for line in lines)
    return Answer(
        question=plan.question,
        answer=text,
        citations=list(sources.values()),
        bids=plan.bids,
        confidence=round(min(confidences), 2),
        mode="record",
    )


def small_talk_answer(question: str) -> Answer | None:
    """Greetings and thanks get a short help text instead of a document search."""
    return Answer(question=question, answer=HELP, mode="chat") if _SMALL_TALK.match(question) else None


def overview_answer(plan: QAPlan, outputs_dir: Path, store: IndexStore) -> Answer | None:
    """ "What are the bids about?": title, issuer and due date from the validated record, plus the documents."""
    if not _OVERVIEW.search(plan.question) or not plan.bids:
        return None
    parts, citations = [], {}
    for bid in plan.bids:
        fields = record_fields(outputs_dir, bid) or {}
        facts = []
        for name, label in (("Title", ""), ("company_name", "issued by "), ("Due Date", "due ")):
            f = fields.get(name)
            if validated(f):
                source = Source(**f["sources"][0])
                citations[source.label()] = source
                facts.append(f"{label}{f['value']} [{source.label()}]")
        docs = document_evidence(bid, store)
        citations.update({e.source.label(): e.source for e in docs})
        listing = "\n".join(f"  - {e.text.split(' document: ', 1)[-1]}" for e in docs)
        parts.append(f"**{bid}**: " + ("; ".join(facts) if facts else "no validated record yet") + f"\n{listing}")
    return Answer(
        question=plan.question,
        answer="\n\n".join(parts),
        citations=list(citations.values()),
        bids=plan.bids,
        confidence=0.9,
        mode="record",
    )


# Fields shown when two or more bids are compared in general ("compare the bids"), in this order.
COMPARE_FIELDS = (
    "Title",
    "company_name",
    "Due Date",
    "Bid Submission Type",
    "Pre Bid Meeting",
    "Term of Bid",
    "Delivery Date",
    "Payment Terms",
    "Bid Bond Requirement",
    "Contract or Cooperative to use",
    "Product",
    "Any Additional Documentation Required",
)
_COMPARE_WORDS = frozenset(
    [
        "compare",
        "comparison",
        "both",
        "differ",
        "difference",
        "differences",
        "different",
        "between",
        "versus",
        "each",
        "all",
        "across",
        "them",
        "side",
        "two",
        "give",
        "show",
        "tell",
        "please",
    ]
)


def _cell(value: str, limit: int = 140) -> str:
    value = " ".join(value.split()).replace("|", "/")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def compare_answer(plan: QAPlan, outputs_dir: Path) -> Answer | None:
    """ "Compare the bids" with no particular topic: the key fields side by side from the validated records."""
    if not plan.compare or len(plan.bids) < 2:
        return None
    topic = {_singular(w) for w in _WORD.findall(plan.question.lower())} - _STOP - _COMPARE_WORDS
    if topic - {_singular(b.lower()) for b in plan.bids}:
        return None  # "compare the warranty…": a specific topic, answered elsewhere
    records = {bid: record_fields(outputs_dir, bid) for bid in plan.bids}
    if any(fields is None for fields in records.values()):
        return None
    rows, citations = [], {}
    for name in COMPARE_FIELDS:
        cells = []
        for bid in plan.bids:
            f = records[bid].get(name) or {}
            if validated(f):
                source = Source(**f["sources"][0])
                citations[source.label()] = source
                cells.append(f"{_cell(f['value'])} [{short_label(source)}]")
            elif f.get("status") == "failed":
                cells.append("⚠️ extracted value not validated")
            else:
                cells.append(NOT_FOUND)
        rows.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
    header = "| Field | " + " | ".join(plan.bids) + " |\n|---|" + "---|" * len(plan.bids)
    return Answer(
        question=plan.question,
        answer=header + "\n" + "\n".join(rows),
        citations=list(citations.values()),
        bids=plan.bids,
        confidence=0.9,
        mode="record",
    )


def passages_answer(plan: QAPlan, evidence: list[Evidence], limit: int = 3, timed_out: bool = True) -> Answer:
    """Tier 3: quote the sentences that best match the question, with citations; no LLM involved."""
    words = {w for w in _WORD.findall(plan.question.lower()) if w not in _STOP}
    picks = []
    for rank, e in enumerate(evidence):
        if e.chunk_id.startswith(("doc:", "record:")):
            continue
        # Whole sentences only: headings and table rows match the words but say nothing.
        sentences = [s.strip() for s in _SENTENCE.split(e.text) if len(s.split()) >= 6 and "|" not in s]
        scored = [(len(words & set(_WORD.findall(s.lower()))), s) for s in sentences]
        score, sentence = max(scored, default=(0, ""))
        if score:
            picks.append((score, -rank, sentence if len(sentence) <= 300 else sentence[:297] + "…", e))
    picks = sorted(picks, key=lambda p: (-p[0], -p[1]))[:limit]
    if not picks:  # nothing matched the question's words: that is not evidence the answer is absent
        return Answer(
            question=plan.question,
            answer=f"{'The model did not finish in time' if timed_out else 'The model could not answer'} and no "
            "passage matched the question's wording. Try a more specific question, e.g. about the due date, "
            "warranty or required documents.",
            bids=plan.bids,
            mode="passages",
        )
    body = "\n".join(f"- “{sentence}” [{e.source.label()}]" for _, _, sentence, e in picks)
    return Answer(
        question=plan.question,
        answer=f"{'The model did not finish in time' if timed_out else 'The model could not answer'}, "
        f"so here are the most relevant passages:\n{body}",
        citations=list({e.source.label(): e.source for *_, e in picks}.values()),
        bids=plan.bids,
        confidence=0.3,
        mode="passages",
    )


class QAAgent:
    def __init__(self, llm: LLM, tracer: Tracer, budget_s: float | None = None):
        self.llm = llm
        self.tracer = tracer
        cfg = section("qa")
        self.budget_s = float(budget_s if budget_s is not None else cfg.get("answer_budget_s", 90))
        self.deadline = time.perf_counter() + self.budget_s  # the agent is built per question

    def _remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def answer(self, plan: QAPlan, evidence: list[Evidence]) -> Answer:
        if not evidence:
            return Answer(question=plan.question, answer=NOT_FOUND, bids=plan.bids)
        text, ids = format_evidence(evidence)
        # Evidence first, question last: Ollama reuses an already-read prompt prefix, so a follow-up question
        # about the same bid only pays for what changed.
        prompt = f"Evidence:\n{text}\n\nBids in scope: {', '.join(plan.bids)}\nQuestion: {plan.question}"
        with self.tracer.step("qa", "answer", {"question": plan.question, "bids": plan.bids}) as step:
            try:
                if self._remaining() < 3:
                    raise LLMTimeout("no time left for the model")
                out = self.llm.structured(
                    SYSTEM, prompt, AnswerOutput, step, timeout_s=self._remaining(), max_tokens=400
                )
                cited = [e for e in dict.fromkeys(out.evidence_ids or _EVIDENCE_REF.findall(out.answer)) if e in ids]
                cited = [e for e in cited if not ids[e].chunk_id.startswith("doc:")] or cited
                if out.answer.strip().rstrip(".") != NOT_FOUND and not cited:
                    # Guardrail: an answer without a citation is not accepted; ask once more if time allows.
                    if self._remaining() < 3:
                        raise LLMTimeout("no time left to ask for citations")
                    out = self.llm.structured(
                        SYSTEM,
                        prompt + "\n\nYour previous answer cited no passage ids. Cite them.",
                        AnswerOutput,
                        step,
                        timeout_s=self._remaining(),
                        max_tokens=400,
                    )
                    cited = [e for e in dict.fromkeys(out.evidence_ids) if e in ids]
            except LLMError as exc:  # timeout, unusable output or Ollama down: quote the passages instead
                result = passages_answer(plan, evidence, timed_out=isinstance(exc, LLMTimeout))
                result.run_id = self.tracer.run_id
                step.output({"error": str(exc), "fallback": result.model_dump(exclude={"question"})})
                return result
            answer = out.answer.strip()
            if answer.rstrip(".").lower() == NOT_FOUND.lower():
                answer, cited, confidence = NOT_FOUND, [], 0.0  # "not found" needs no sources
            elif not cited:
                answer, confidence = NOT_FOUND, 0.0
            else:
                confidence = round(out.confidence, 2)
            # Inline "[E3]" markers refer to the prompt's numbering; replace them with readable citations.
            answer = _INLINE_REF.sub(lambda m: _cite(m, ids), answer)
            result = Answer(
                question=plan.question,
                answer=re.sub(r"\s+([.,;])", r"\1", answer).strip(),
                citations=list({ids[e].source.label(): ids[e].source for e in cited}.values()),
                bids=plan.bids,
                confidence=confidence,
                run_id=self.tracer.run_id,
            )
            step.output(result.model_dump(exclude={"question"}))
            return result
