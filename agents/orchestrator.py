"""Orchestrator: the LangGraph graphs that coordinate the agents.

Extraction mode
    ingest -> plan -> extract_group (one per field group, in parallel) -> reconcile -> validate
        -> [rejected fields left and retries remaining] retry_field (parallel) -> reconcile -> validate ...
        -> report
Question-answering mode
    plan -> retrieve (one per bid, in parallel) -> answer

Shared state is the typed dict below; parallel branches merge into it through reducers. Every agent step
is traced (agents/tracing.py), and a failing step degrades its fields instead of stopping the run.
"""

from __future__ import annotations

import logging
import operator
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.addendum_agent import AddendumAgent, addenda_of, apply_changes
from agents.extraction_agent import ExtractionAgent
from agents.fields import BY_NAME, GROUPS
from agents.ingestion_agent import IngestionAgent
from agents.llm import LLM, OllamaLLM
from agents.qa_agent import (
    _ADDENDUM_NO,
    _CHANGES,
    Answer,
    QAAgent,
    compare_answer,
    document_evidence,
    overview_answer,
    plan_question,
    record_answer,
    small_talk_answer,
)
from agents.report_agent import ReportAgent
from agents.retrieval_agent import RetrievalAgent
from agents.tracing import Tracer
from agents.validator_agent import ValidatorAgent
from schemas.documents import DocType, Document
from schemas.extraction import AddendumChange, BidRecord, Evidence, FieldResult, ValidationSummary
from search.hybrid import HybridSearcher
from search.store import IndexStore
from settings import ROOT, section

log = logging.getLogger(__name__)


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _failed_group(names: list[str], exc: Exception, attempts: int = 1) -> dict:
    note = f"Extraction failed: {type(exc).__name__}: {exc}"
    return {
        "fields": {n: FieldResult(name=n, notes=note, attempts=attempts, status="failed") for n in names},
        "errors": [f"{', '.join(names)}: {note}"],
    }


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class ExtractionState(TypedDict, total=False):
    bid_id: str
    folder: str | None
    documents: list[Document]
    plan: list[str]  # field groups to extract
    evidence: Annotated[dict[str, Evidence], _merge]  # chunk_id -> passage, from every retrieval
    fields: Annotated[dict[str, FieldResult], _merge]
    changes: list[AddendumChange]
    validation: ValidationSummary
    retry: list[str]  # fields sent back by the validator
    feedback: dict[str, str]
    rounds: int  # validation rounds so far
    errors: Annotated[list[str], operator.add]
    record: BidRecord


class QAState(TypedDict, total=False):
    question: str
    bids: list[str] | None
    plan: Any
    evidence: Annotated[list[Evidence], operator.add]
    answer: Answer | None


class Orchestrator:
    def __init__(
        self,
        store: IndexStore | None = None,
        llm: LLM | None = None,
        searcher: HybridSearcher | None = None,
        outputs_dir: Path | None = None,
        trace: bool = True,
    ):
        cfg = section("agents")
        self.max_retries = int(cfg.get("max_retries", 2))
        self.max_parallel = int(cfg.get("max_parallel", 3))
        self.store = store or IndexStore()
        self.searcher = searcher or HybridSearcher(self.store)
        self.llm = llm or OllamaLLM()
        self.outputs_dir = outputs_dir or ROOT / "outputs"
        self.trace = trace
        self.qa_passage_tokens = int(section("qa").get("passage_tokens", 600))

    # ---- agents for one run (each run gets its own tracer) -----------------------------------------

    def _agents(self, tracer: Tracer) -> dict[str, Any]:
        retrieval = RetrievalAgent(self.searcher, tracer)
        return {
            "ingestion": IngestionAgent(self.store, tracer),
            "retrieval": retrieval,
            "extraction": ExtractionAgent(self.llm, tracer),
            "addendum": AddendumAgent(self.llm, retrieval, tracer),
            "validator": ValidatorAgent(tracer),
            "report": ReportAgent(self.llm, tracer),
            "qa": QAAgent(self.llm, tracer),
        }

    # ---- extraction mode -------------------------------------------------------------------------

    def _extraction_graph(self, tracer: Tracer):
        a = self._agents(tracer)

        def ingest(state: ExtractionState) -> dict:
            documents, report = a["ingestion"].run(state["bid_id"], state.get("folder"))
            errors = [f"{f.file_name}: {f.error}" for f in report.failures] if report else []
            if not documents:
                errors.append(f"no documents indexed for {state['bid_id']}")
            return {"documents": documents, "errors": errors}

        def plan(state: ExtractionState) -> dict:
            with tracer.step("orchestrator", "plan", {"bid_id": state["bid_id"]}) as step:
                groups = list(GROUPS) if state["documents"] else []
                step.output(
                    {
                        "groups": {g: [s.name for s in GROUPS[g]] for g in groups},
                        "then": [
                            "reconcile addenda",
                            "validate",
                            f"retry rejected fields (max {self.max_retries})",
                            "report",
                        ],
                    }
                )
            return {"plan": groups, "rounds": 0}

        def fan_out_groups(state: ExtractionState) -> list[Send] | str:
            if not state["plan"]:
                return "report"
            return [Send("extract_group", {"bid_id": state["bid_id"], "group": g}) for g in state["plan"]]

        def extract_group(payload: dict) -> dict:
            specs = GROUPS[payload["group"]]
            try:
                evidence = a["retrieval"].for_fields(specs, payload["bid_id"])
                fields = a["extraction"].extract(payload["group"], specs, evidence)
            except Exception as exc:  # e.g. an index error: this group fails, the other groups carry on
                log.exception("group %s failed", payload["group"])
                return _failed_group([s.name for s in specs], exc)
            return {"evidence": {e.chunk_id: e for e in evidence}, "fields": fields}

        def reconcile(state: ExtractionState) -> dict:
            changes, pool, errors = state.get("changes"), {}, []
            if changes is None:  # computed once; later rounds only re-apply it
                try:
                    changes, pool = a["addendum"].change_log(addenda_of(state["documents"]))
                except Exception as exc:  # fields keep their extracted values; the error is reported
                    log.exception("addendum reconciliation failed")
                    changes, errors = [], [f"addendum reconciliation failed: {type(exc).__name__}: {exc}"]
            fields = {n: f.model_copy(deep=True) for n, f in state["fields"].items()}
            with tracer.step("addendum", "apply", {"changes": len(changes)}) as step:
                apply_changes(fields, changes)
                step.output([f"{c.addendum}: {c.field} ({c.change_type})" for c in changes if c.applied])
            return {"changes": changes, "evidence": pool, "fields": fields, "errors": errors}

        def validate(state: ExtractionState) -> dict:
            fields = {n: f.model_copy(deep=True) for n, f in state["fields"].items()}
            rfp = next((d for d in state["documents"] if d.doc_type == DocType.RFP), None)
            summary = a["validator"].validate(
                fields, state["evidence"], state["changes"], rfp.document_date if rfp else None
            )
            feedback: dict[str, str] = {}
            for issue in summary.issues:
                f = fields[issue.field]
                if f.attempts <= self.max_retries and (issue.check != "not_found" or f.attempts == 1):
                    feedback[issue.field] = f"{feedback.get(issue.field, '')} {issue.message}.".strip()
            with tracer.step("orchestrator", "route", {"round": state["rounds"] + 1}) as step:
                step.output({"retry": sorted(feedback), "done": not feedback})
            return {
                "fields": fields,
                "validation": summary,
                "retry": sorted(feedback),
                "feedback": feedback,
                "rounds": state["rounds"] + 1,
            }

        def fan_out_retries(state: ExtractionState) -> list[Send] | str:
            if not state["retry"]:
                return "report"
            return [
                Send(
                    "retry_field",
                    {"bid_id": state["bid_id"], "field": state["fields"][n], "feedback": state["feedback"][n]},
                )
                for n in state["retry"]
            ]

        def retry_field(payload: dict) -> dict:
            try:
                return _retry_field(payload)
            except Exception as exc:
                log.exception("retry of %s failed", payload["field"].name)
                return _failed_group([payload["field"].name], exc, attempts=payload["field"].attempts + 1)

        def _retry_field(payload: dict) -> dict:
            prev: FieldResult = payload["field"]
            spec = BY_NAME[prev.name]
            # Re-retrieve with the field description as an extra query and more results per query.
            evidence = a["retrieval"].gather(
                [f"{spec.name}: {spec.description}", *spec.queries],
                {"bid_id": payload["bid_id"]},
                f"retry:{spec.name}",
                k=6,
            )
            rejected = f"'{prev.value}' was rejected: {payload['feedback']}" if prev.value else payload["feedback"]
            fields = a["extraction"].extract(
                f"retry:{spec.name}", [spec], evidence, {spec.name: rejected}, prev.attempts + 1
            )
            return {"evidence": {e.chunk_id: e for e in evidence}, "fields": fields}

        def report(state: ExtractionState) -> dict:
            fields = dict(state.get("fields", {}))
            fields["Bid Summary"] = a["report"].summarise(state["bid_id"], fields, state.get("changes", []))
            run = {
                "run_id": tracer.run_id,
                **tracer.totals,
                "validation_rounds": state.get("rounds", 0),
                "errors": state.get("errors", []),
            }
            record = a["report"].record(
                state["bid_id"],
                self.llm.model,
                fields,
                state.get("changes", []),
                state.get("validation") or ValidationSummary(),
                run,
            )
            return {"record": record}

        g = StateGraph(ExtractionState)
        for name, fn in [
            ("ingest", ingest),
            ("plan", plan),
            ("extract_group", extract_group),
            ("reconcile", reconcile),
            ("validate", validate),
            ("retry_field", retry_field),
            ("report", report),
        ]:
            g.add_node(name, fn)
        g.add_edge(START, "ingest")
        g.add_edge("ingest", "plan")
        g.add_conditional_edges("plan", fan_out_groups, ["extract_group", "report"])
        g.add_edge("extract_group", "reconcile")
        g.add_edge("reconcile", "validate")
        g.add_conditional_edges("validate", fan_out_retries, ["retry_field", "report"])
        g.add_edge("retry_field", "reconcile")
        g.add_edge("report", END)
        return g.compile()

    def _warm_up(self) -> None:
        # Load the embedding model once, before parallel branches would race to load it.
        if self.store.count():
            self.store.embedder.embed_query("warm-up")

    def preload(self) -> None:
        """Load the embedding model and the LLM ahead of the first question (called when the API starts)."""
        self._warm_up()
        if hasattr(self.llm, "preload"):
            self.llm.preload()

    def extract_bid(self, bid_id: str, folder: str | Path | None = None, write: bool = True) -> BidRecord:
        self._warm_up()
        tracer = Tracer(out_dir=self.outputs_dir / "traces", enabled=self.trace)
        started = time.perf_counter()
        state = self._extraction_graph(tracer).invoke(
            {"bid_id": bid_id, "folder": str(folder) if folder else None, "fields": {}, "evidence": {}, "errors": []},
            {"max_concurrency": self.max_parallel, "recursion_limit": 50},
        )
        record: BidRecord = state["record"]
        record.run["seconds"] = round(time.perf_counter() - started, 1)
        record.run["trace"] = _display(tracer.path) if self.trace else None
        if write:
            json_path, md_path = ReportAgent.write(record, self.outputs_dir)
            log.info("wrote %s and %s (trace %s)", json_path, md_path, tracer.path)
        return record

    # ---- question-answering mode -------------------------------------------------------------------

    def _qa_graph(self, tracer: Tracer):
        a = self._agents(tracer)

        def plan(state: QAState) -> dict:
            with tracer.step("orchestrator", "plan_question", {"question": state["question"]}) as step:
                p = plan_question(state["question"], self.store)
                # The selected bids (e.g. the UI's bid picker) apply unless the question itself names a bid
                # or asks for a comparison: "the Dell laptop bid" or "both bids" beats the picker.
                # One shared title word ("warranty") is too weak to override the picker; two are not.
                question_scoped = (
                    p.compare or p.reason == "bid id in question" or (p.title_words >= 2 and len(p.bids) == 1)
                )
                selected = state.get("bids") or []
                # "both bids" with several bids selected means those; with one selected, all bids.
                compare_selected = p.compare and p.reason != "bid id in question" and len(selected) > 1
                if selected and (not question_scoped or compare_selected):
                    p.bids, p.reason = selected, "selected bids"
                step.output(p.model_dump())
            return {"plan": p}

        def lookup(state: QAState) -> dict:
            # Tier 1: a question about a field that every bid in scope has a validated value for is answered
            # from the record instantly; anything else goes on to retrieval and the LLM.
            with tracer.step("qa", "record_lookup", {"question": state["question"]}) as step:
                plan = state["plan"]
                found = (
                    small_talk_answer(plan.question)
                    or overview_answer(plan, self.outputs_dir, self.store)
                    or compare_answer(plan, self.outputs_dir)
                    or record_answer(plan, self.outputs_dir)
                )
                if found is not None:
                    found.run_id = tracer.run_id
                step.output(found.model_dump(exclude={"question"}) if found else {"answered": False})
            return {"answer": found} if found else {}

        def fan_out(state: QAState) -> list[Send] | str:
            if state.get("answer") is not None:
                return END
            if not state["plan"].bids:
                return "answer"  # nothing indexed: the answer agent replies "Not found in documents"
            bids = state["plan"].bids
            return [Send("retrieve", {"question": state["question"], "bid": b, "n_bids": len(bids)}) for b in bids]

        def retrieve(payload: dict) -> dict:
            bid, question = payload["bid"], payload["question"]
            budget = max(250, self.qa_passage_tokens // payload["n_bids"])  # keeps the whole prompt small
            try:
                passages = []
                if _CHANGES.search(question):  # "what changed in Addendum 2": look in that addendum first
                    addenda = {"bid_id": bid, "doc_type": ["addendum"]}
                    if number := _ADDENDUM_NO.search(question):
                        addenda["addendum_number"] = [int(number[1])]
                    passages = a["retrieval"].gather([question], addenda, f"qa:{bid}", k=3, budget=budget)
                passages = passages or a["retrieval"].gather(
                    [question], {"bid_id": bid}, f"qa:{bid}", k=3, budget=budget
                )
            except Exception:  # answer from the other bids instead of failing
                log.exception("retrieval for %s failed", bid)
                passages = []
            return {"evidence": document_evidence(bid, self.store) + passages}

        def answer(state: QAState) -> dict:
            return {"answer": a["qa"].answer(state["plan"], state.get("evidence", []))}

        g = StateGraph(QAState)
        g.add_node("plan", plan)
        g.add_node("lookup", lookup)
        g.add_node("retrieve", retrieve)
        g.add_node("answer", answer)
        g.add_edge(START, "plan")
        g.add_edge("plan", "lookup")
        g.add_conditional_edges("lookup", fan_out, ["retrieve", "answer", END])
        g.add_edge("retrieve", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def ask(self, question: str, bids: list[str] | None = None) -> Answer:
        self._warm_up()
        tracer = Tracer(out_dir=self.outputs_dir / "traces", enabled=self.trace)
        state = self._qa_graph(tracer).invoke(
            {"question": question, "bids": bids, "evidence": []}, {"max_concurrency": self.max_parallel}
        )
        return state["answer"]
