"""Agent tests with a scripted LLM, so the orchestration logic is tested without a model."""

import json
import re

import pytest

from agents.addendum_agent import apply_changes
from agents.extraction_agent import ExtractionAgent
from agents.fields import BY_NAME, FIELD_NAMES
from agents.llm import LLMError, LLMTimeout
from agents.orchestrator import Orchestrator
from agents.qa_agent import (
    QAAgent,
    QAPlan,
    compare_answer,
    document_evidence,
    match_fields,
    plan_question,
    record_answer,
)
from agents.tracing import Tracer
from agents.validator_agent import ValidatorAgent, grounding_problems
from schemas.extraction import (
    NOT_FOUND,
    AddendumChange,
    AnswerOutput,
    ChangeLogOutput,
    Evidence,
    ExtractedField,
    ExtractionOutput,
    FieldResult,
    ProposedChange,
    Source,
    SummaryOutput,
)
from search.hybrid import HybridSearcher, SearchConfig
from search.store import IndexStore
from tests.test_pdf_parser import NO_OCR, make_pdf
from tests.test_store import HashEmbedder


class ScriptedLLM:
    model = "scripted"

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def structured(self, system, prompt, schema, step=None, timeout_s=None, max_tokens=None):
        self.calls.append((schema.__name__, prompt))
        if step is not None:
            step.tracer.count_llm_call()
            step.usage(100, 10)
        return self.handler(schema, prompt)


def evidence_ids(prompt: str) -> dict[str, str]:
    """[E1] (...)\\ntext blocks of a prompt -> id: text"""
    return {m[0]: m[1] for m in re.findall(r"\[(E\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[E|\Z)", prompt, re.DOTALL)}


def ev(chunk_id: str, text: str, file: str = "RFP.pdf", page: int = 1) -> Evidence:
    return Evidence(
        chunk_id=chunk_id, text=text, source=Source(file=file, page=page, chunk_id=chunk_id), doc_type="rfp"
    )


# ---- validator ----------------------------------------------------------------------------------------


def test_grounding_accepts_reformatted_dates_and_rejects_invented_ones():
    text = "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST."
    assert grounding_problems("2024-07-09 14:00 CST", text) == []
    assert grounding_problems("2024-07-10 14:00 CST", text) == ["date(s) 2024-07-10 not in the cited passages"]
    assert "number(s) 45" in grounding_problems("Within 45 days of award", "Delivery within 30 days of award")[0]


def test_validator_flags_missing_citation_format_and_stale_addendum_value():
    evidence = {"c1": ev("c1", "Proposals due June 27, 2024 at 2:00 PM CST.")}
    fields = {
        "Due Date": FieldResult(name="Due Date", value="2024-06-27", sources=[evidence["c1"].source], confidence=0.9),
        "Title": FieldResult(name="Title", value="Laptops", confidence=0.9),  # no citation
        "Delivery Date": FieldResult(name="Delivery Date", notes=NOT_FOUND),
        "Bid Number": FieldResult(name="Bid Number", notes=NOT_FOUND),  # expected field
    }
    change = AddendumChange(
        addendum="Add 1.pdf",
        addendum_number=1,
        field="Due Date",
        change_type="modifies",
        description="new date",
        new_value="2024-07-09 14:00 CST",
        applied=True,
    )
    summary = ValidatorAgent(Tracer(enabled=False)).validate(fields, evidence, [change])
    checks = {(i.field, i.check) for i in summary.issues}
    assert ("Title", "citation") in checks
    assert ("Due Date", "format") in checks and ("Due Date", "consistency") in checks
    assert ("Bid Number", "not_found") in checks
    assert fields["Delivery Date"].status == "not_found" and summary.not_found == 2 and summary.failed == 2


# ---- addendum reconciliation --------------------------------------------------------------------------


def test_apply_changes_latest_addendum_wins_and_lists_are_only_annotated():
    src = Source(file="Add.pdf", page=1, chunk_id="a")
    fields = {
        "Due Date": FieldResult(name="Due Date", value="2024-06-27 14:00 CST", confidence=0.9),
        "Product Specification": FieldResult(name="Product Specification", value="16GB RAM; 256GB SSD"),
    }
    changes = [
        AddendumChange(
            addendum="Add 2.pdf",
            addendum_number=2,
            field="Due Date",
            change_type="modifies",
            description="extended",
            new_value="2024-07-09 14:00 CST",
            sources=[src],
        ),
        AddendumChange(
            addendum="Add 1.pdf",
            addendum_number=1,
            field="Due Date",
            change_type="modifies",
            description="extended",
            new_value="2024-07-01 14:00 CST",
            sources=[src],
        ),
        AddendumChange(
            addendum="Add 1.pdf",
            addendum_number=1,
            field="Product Specification",
            change_type="modifies",
            description="one-year warranty for Chromebooks",
            new_value="1-year warranty",
            sources=[src],
        ),
    ]
    apply_changes(fields, changes)
    apply_changes(fields, changes)  # idempotent
    assert fields["Due Date"].value == "2024-07-09 14:00 CST"
    assert fields["Due Date"].notes.startswith("Changed by Addendum 2")
    assert fields["Product Specification"].value == "16GB RAM; 256GB SSD"  # list fields are never replaced
    assert fields["Product Specification"].notes == ""  # the clarification stays in the change log only


def test_apply_changes_keeps_normalised_value_when_addendum_states_the_same_date():
    src = Source(file="Add 2.pdf", page=1, chunk_id="a")
    fields = {"Due Date": FieldResult(name="Due Date", value="2024-07-09 14:00 CST", confidence=0.9)}
    change = AddendumChange(
        addendum="Add 2.pdf",
        addendum_number=2,
        field="Due Date",
        change_type="modifies",
        description="extended",
        new_value="July 9, 2024 at 2:00 PM CST",
        sources=[src],
    )
    apply_changes(fields, [change])
    assert fields["Due Date"].value == "2024-07-09 14:00 CST"
    assert fields["Due Date"].sources[0].file == "Add 2.pdf" and fields["Due Date"].notes.startswith("Per Addendum 2")


def test_citations_are_pruned_to_passages_that_support_the_value():
    from agents.extraction_agent import prune_citations

    passages = [ev("a", "Contract term: three years."), ev("b", "Affidavit valid for 24 months."), ev("c", "Laptops.")]
    assert [e.chunk_id for e in prune_citations("24 months", passages)] == ["b"]
    # Nothing supports the value alone: keep the best one, so the validator rejects it against that.
    assert len(prune_citations("Net 45 days", passages)) == 1


def test_delivery_date_equal_to_due_date_is_flagged():
    evidence = {"c": ev("c", "Laptop | 30 | 06/10/2024. Closing Date: 06/10/2024 02:00 PM EDT")}
    src = [evidence["c"].source]
    fields = {
        "Due Date": FieldResult(name="Due Date", value="2024-06-10 14:00 EDT", sources=src, confidence=0.9),
        "Delivery Date": FieldResult(name="Delivery Date", value="2024-06-10", sources=src, confidence=0.9),
    }
    summary = ValidatorAgent(Tracer(enabled=False)).validate(fields, evidence, [])
    assert [(i.field, i.check) for i in summary.issues] == [("Delivery Date", "consistency")]


# ---- extraction agent ---------------------------------------------------------------------------------


def test_extraction_uses_ids_from_notes_drops_unknown_ids_and_survives_llm_errors():
    def handler(schema, prompt):
        return ExtractionOutput(
            fields=[
                ExtractedField(name="Title", evidence_ids=[], value="Laptops", confidence=0.8, notes="see E1"),
                ExtractedField(name="Bid Number", evidence_ids=["E9"], value="X-1", confidence=0.8, notes=""),
                ExtractedField(name="company_name", evidence_ids=[], value=NOT_FOUND, confidence=0.1, notes=""),
            ]
        )

    agent = ExtractionAgent(ScriptedLLM(handler), Tracer(enabled=False))
    specs = [BY_NAME[n] for n in ("Title", "Bid Number", "company_name", "contact_info")]
    out = agent.extract("identity", specs, [ev("c1", "Laptops RFP")])
    assert out["Title"].sources[0].chunk_id == "c1"
    assert out["Bid Number"].sources == []  # E9 does not exist; the validator will reject the value
    assert out["company_name"].value is None and out["company_name"].status == "not_found"
    assert out["contact_info"].status == "not_found"  # omitted by the model

    def failing(schema, prompt):
        raise LLMError("timeout")

    failed = ExtractionAgent(ScriptedLLM(failing), Tracer(enabled=False)).extract("identity", specs, [ev("c1", "x")])
    assert all(f.status == "failed" and f.notes.startswith("Extraction failed") for f in failed.values())


# ---- full extraction graph ----------------------------------------------------------------------------


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.pipeline.section", lambda name: NO_OCR)
    folder = tmp_path / "BidT"
    folder.mkdir()
    rfp_lines = [
        (72, 100, "Request for Proposal: proposals due June 27, 2024 at 2:00 PM CST."),
        (72, 140, "Delivery Date: delivery within 30 days of award."),
    ]
    make_pdf(folder / "RFP Widgets.pdf", [rfp_lines])
    make_pdf(
        folder / "Addendum 1.pdf", [[(72, 100, "Addendum No. 1: the new due date is July 9, 2024 at 2:00 PM CST.")]]
    )
    store = IndexStore(tmp_path / "index", embedder=HashEmbedder())
    return store, folder


def scripted_bid(schema, prompt):
    blocks = evidence_ids(prompt)
    find = lambda needle: next((i for i, t in blocks.items() if needle in t), None)
    if schema is ExtractionOutput:
        names = re.findall(r"^- ([^:\n]+):", prompt.split("\n\nEvidence:")[0], re.MULTILINE)
        fields = []
        for name in names:
            if name == "Due Date" and find("June 27"):
                value, eid = "2024-06-27 14:00 CST", find("June 27")
                fields.append(ExtractedField(name=name, evidence_ids=[eid], value=value, confidence=0.9, notes=""))
            elif name == "Delivery Date" and find("30 days"):
                # First attempt invents a number; after the validator's feedback it copies the real one.
                value = "Within 30 days of award" if "rejected" in prompt else "Within 45 days of award"
                fields.append(
                    ExtractedField(name=name, evidence_ids=[find("30 days")], value=value, confidence=0.9, notes="")
                )
            else:
                fields.append(ExtractedField(name=name, evidence_ids=[], value=None, confidence=0.0, notes=NOT_FOUND))
        return ExtractionOutput(fields=fields)
    if schema is ChangeLogOutput:
        eid = find("July 9")
        return ChangeLogOutput(
            changes=[
                ProposedChange(
                    field="Due Date",
                    change_type="modifies",
                    description="Due date extended",
                    new_value="2024-07-09 14:00 CST",
                    evidence_ids=[eid],
                )
            ]
            if eid
            else []
        )
    if schema is SummaryOutput:
        return SummaryOutput(summary="A bid for widgets due 9 July 2024.")
    raise AssertionError(schema)


def test_extraction_graph_retries_rejected_field_and_applies_addendum(indexed, tmp_path):
    store, folder = indexed
    llm = ScriptedLLM(scripted_bid)
    orch = Orchestrator(
        store=store, llm=llm, searcher=HybridSearcher(store, config=SearchConfig()), outputs_dir=tmp_path / "out"
    )
    record = orch.extract_bid("BidT", folder)

    due = record.fields["Due Date"]
    assert due.value == "2024-07-09 14:00 CST"
    assert due.notes.startswith("Changed by Addendum 1")
    assert due.sources[0]["file"] == "Addendum 1.pdf"
    assert [c.field for c in record.addendum_changes] == ["Due Date"] and record.addendum_changes[0].applied
    # The invented number was rejected by the validator and re-extracted once, with the feedback.
    retries = [p for name, p in llm.calls if name == "ExtractionOutput" and "rejected" in p]
    delivery = [p for p in retries if p.startswith("Fields to extract:\n- Delivery Date")]
    assert len(delivery) == 1 and "'Within 45 days of award' was rejected" in delivery[0]
    assert record.fields["Delivery Date"].value == "Within 30 days of award"
    # Fields nearly every bid states (here: missing from the tiny test bid) get exactly one more try.
    expected = {n for n in FIELD_NAMES if BY_NAME[n].expected and n != "Due Date"}
    assert {re.match(r"Fields to extract:\n- ([^:]+):", p)[1] for p in retries} - {"Delivery Date"} == expected
    assert len(retries) == 1 + len(expected)
    assert set(record.fields) == set(FIELD_NAMES)
    assert record.fields["Bid Summary"].value == "A bid for widgets due 9 July 2024."
    assert record.validation.failed == 0

    saved = json.loads((tmp_path / "out" / "BidT.json").read_text())
    assert saved["fields"]["Due Date"]["value"] == "2024-07-09 14:00 CST"
    trace = [
        json.loads(line)
        for line in (tmp_path / "out" / "traces" / f"{record.run['run_id']}.jsonl").read_text().splitlines()
    ]
    agents = {t["agent"] for t in trace}
    assert {"ingestion", "orchestrator", "retrieval", "extraction", "addendum", "validator", "report"} <= agents
    assert all("latency_s" in t and "prompt_tokens" in t for t in trace)


def test_extraction_graph_handles_empty_bid(tmp_path):
    store = IndexStore(tmp_path / "index", embedder=HashEmbedder())
    orch = Orchestrator(store=store, llm=ScriptedLLM(scripted_bid), outputs_dir=tmp_path / "out")
    record = orch.extract_bid("Nothing", write=False)
    assert record.run["errors"] == ["no documents indexed for Nothing"]
    assert all(f.value is None for f in record.fields.values())


# ---- question answering -------------------------------------------------------------------------------


def test_question_naming_a_bid_overrides_the_selected_bid(indexed, tmp_path):
    store, folder = indexed
    store.index_folder(folder)
    other = tmp_path / "BidU"
    other.mkdir()
    make_pdf(other / "RFP Gadgets.pdf", [[(72, 100, "Request for Proposal: gadget repairs, due August 1, 2024.")]])
    store.index_folder(other)
    llm = ScriptedLLM(lambda schema, prompt: AnswerOutput(evidence_ids=[], answer=NOT_FOUND, confidence=0.0))
    orch = Orchestrator(store=store, llm=llm, searcher=HybridSearcher(store), outputs_dir=tmp_path / "out")
    assert orch.ask("Compare the due dates of both bids", bids=["BidT"]).bids == ["BidT", "BidU"]
    assert orch.ask("When is BidU due?", bids=["BidT"]).bids == ["BidU"]  # named in the question
    assert orch.ask("What is the gadget price?", bids=["BidT"]).bids == ["BidT"]  # one title word: picker wins
    assert orch.ask("When are the gadget repairs due?", bids=["BidT"]).bids == ["BidU"]  # clear title match


def test_plan_question_finds_bids_by_id_title_or_comparison(indexed):
    store, folder = indexed
    store.index_folder(folder)
    store.index_folder(folder, bid_id="BidT2")
    assert plan_question("When is BidT due?", store).bids == ["BidT"]  # not BidT2 as well
    assert plan_question("When is BidT2 due?", store).bids == ["BidT2"]
    store.remove_document("BidT2", "RFP Widgets.pdf")
    store.remove_document("BidT2", "Addendum 1.pdf")
    assert plan_question("When is BidT due?", store).bids == ["BidT"]
    assert plan_question("Compare the warranty of both bids", store).compare
    assert plan_question("When are the widgets proposals due?", store).bids == ["BidT"]


def test_qa_evidence_lists_documents(indexed):
    store, folder = indexed
    store.index_folder(folder)
    docs = document_evidence("BidT", store)
    assert sorted(e.source.file for e in docs) == sorted(p.name for p in folder.iterdir())
    assert all(e.text.startswith("BidT document: ") for e in docs)


def test_comparison_needs_several_bids_not_just_the_word_compare(indexed):
    store, folder = indexed
    store.index_folder(folder)
    store.index_folder(folder, bid_id="BidT2")
    assert plan_question("Compare the warranty of both bids", store).compare
    assert plan_question("How do the bids differ?", store).compare
    assert plan_question("Compare BidT and BidT2", store).compare
    assert not plan_question("What changed in Addendum 1 compared to the original RFP?", store).compare
    assert not plan_question("What are the bids about?", store).compare


def test_answer_cites_passages_rather_than_the_document_list():
    llm = ScriptedLLM(lambda schema, prompt: AnswerOutput(evidence_ids=["E1", "E2"], answer="No.", confidence=0.9))
    evidence = [ev("doc:B:RFP.pdf", "B document: RFP.pdf (rfp, 9 pages)"), ev("c1", "No email bids.", page=9)]
    answer = QAAgent(llm, Tracer(enabled=False)).answer(QAPlan(question="Email?", bids=["B"]), evidence)
    assert [c.label() for c in answer.citations] == ["RFP.pdf, p. 9"]


def test_match_fields_is_narrow():
    assert match_fields("What is the submission deadline for Bid1 after all addendums?") == ["Due Date"]
    assert match_fields("When is Bid2 due?") == ["Due Date"]
    assert match_fields("Is a bid bond required, and if so, how much?") == ["Bid Bond Requirement"]
    assert match_fields("Which affidavits are required?") == ["Any Additional Documentation Required"]
    assert match_fields("What is the part number of the laptop?") == ["Part_no"]
    assert match_fields("Who is the procurement contact for Bid2?") == ["contact_info"]
    assert match_fields("What are the due date and the payment terms?") == ["Due Date", "Payment Terms"]
    # Near misses go to the LLM instead.
    assert match_fields("What is the deadline to submit questions?") == []
    assert match_fields("What is the delivery address?") == []
    assert match_fields("Compare the warranty requirements of both bids.") == []
    assert match_fields("What changed in Addendum 2 compared to the original RFP?") == []


def test_record_answer_uses_only_validated_values(tmp_path):
    source = [{"file": "Addendum 1.pdf", "page": 1}]
    due = {"value": "2024-07-09 14:00 CST", "sources": source, "notes": "Per Addendum 1", "confidence": 0.9}
    bond = {"value": "Not required", "sources": source, "notes": "", "confidence": 0.3, "status": "failed"}
    record = {"fields": {"Due Date": due | {"status": "passed"}, "Bid Bond Requirement": bond}}
    (tmp_path / "BidT.json").write_text(json.dumps(record))

    found = record_answer(QAPlan(question="When is it due?", bids=["BidT"]), tmp_path)
    assert found.mode == "record"
    assert found.answer == "**BidT · Due Date:** 2024-07-09 14:00 CST (Per Addendum 1) [Addendum 1.pdf, p. 1]"
    assert [c.label() for c in found.citations] == ["Addendum 1.pdf, p. 1"]
    assert record_answer(QAPlan(question="Is a bid bond required?", bids=["BidT"]), tmp_path) is None
    assert record_answer(QAPlan(question="When is it due?", bids=["BidT", "NoRecord"]), tmp_path) is None


def test_general_comparison_is_a_table_of_validated_fields(tmp_path):
    source = [{"file": "RFP.pdf", "page": 2}]
    passed = {"status": "passed", "sources": source, "notes": "", "confidence": 0.9}
    (tmp_path / "A.json").write_text(json.dumps({"fields": {"Due Date": passed | {"value": "2024-07-09"}}}))
    (tmp_path / "B.json").write_text(
        json.dumps({"fields": {"Due Date": {"status": "failed", "value": "x", "sources": source}}})
    )
    table = compare_answer(QAPlan(question="compare both the bids", bids=["A", "B"], compare=True), tmp_path)
    assert table.mode == "record"
    assert "| Due Date | 2024-07-09 [RFP.pdf, p. 2] | ⚠️ extracted value not validated |" in table.answer
    assert f"| Term of Bid | {NOT_FOUND} | {NOT_FOUND} |" in table.answer
    topic = QAPlan(question="Compare the warranty of both bids", bids=["A", "B"], compare=True)
    assert compare_answer(topic, tmp_path) is None  # a specific topic goes to retrieval


def test_ask_answers_field_questions_instantly_and_others_with_the_llm(indexed, tmp_path):
    store, folder = indexed

    def handler(schema, prompt):
        if schema is AnswerOutput:
            return AnswerOutput(evidence_ids=["E1"], answer="See [E1].", confidence=0.8)
        return scripted_bid(schema, prompt)

    llm = ScriptedLLM(handler)
    orch = Orchestrator(
        store=store, llm=llm, searcher=HybridSearcher(store, config=SearchConfig()), outputs_dir=tmp_path / "out"
    )
    orch.extract_bid("BidT", folder)
    calls = len(llm.calls)

    instant = orch.ask("When is BidT due?")
    assert instant.mode == "record" and "2024-07-09 14:00 CST" in instant.answer
    assert len(llm.calls) == calls  # no LLM call

    hello = orch.ask("hi")
    assert hello.mode == "chat" and len(llm.calls) == calls

    assert orch.ask("what is the major data of BidT?").mode == "record"
    assert orch.ask("tell me about BidT").mode == "record"
    overview = orch.ask("What are the bids about?")
    assert overview.mode == "record" and overview.answer.startswith("**BidT**: ") and len(llm.calls) == calls
    assert "RFP Widgets.pdf (rfp, 1 page)" in overview.answer

    written = orch.ask("What colour are the widgets in BidT?")
    assert written.mode == "llm" and [c[0] for c in llm.calls[calls:]] == ["AnswerOutput"]
    assert "Evidence:" in llm.calls[-1][1] and llm.calls[-1][1].rstrip().endswith("widgets in BidT?")

    calls = len(llm.calls)
    again = orch.ask("What colour are the widgets in BidT?")  # repeated: served by the semantic cache
    assert again.cached and again.answer == written.answer and len(llm.calls) == calls
    assert again.usage["cached"] and again.usage["llm_calls"] == 0
    assert written.usage["mode"] == "llm" and written.usage["llm_calls"] == 1


def test_answer_falls_back_to_cited_passages_when_the_model_is_too_slow():
    class SlowLLM(ScriptedLLM):
        def structured(self, system, prompt, schema, step=None, timeout_s=None, max_tokens=None):
            assert timeout_s is not None and timeout_s <= 90
            raise LLMTimeout("timeout after 30s")

    evidence = [
        ev("c0", "Proposals Due | Date | Time"),  # a table row: matches the words, says nothing
        ev("c1", "General terms apply. Proposals are due June 27, 2024 at 2:00 PM.", page=3),
        ev("c2", "Unrelated text about parking."),
    ]
    answer = QAAgent(SlowLLM(None), Tracer(enabled=False)).answer(
        QAPlan(question="When are proposals due?", bids=["B"]), evidence
    )
    assert answer.mode == "passages" and answer.confidence == 0.3
    assert "“Proposals are due June 27, 2024 at 2:00 PM.” [RFP.pdf, p. 3]" in answer.answer
    assert [c.label() for c in answer.citations] == ["RFP.pdf, p. 3"]


def test_answer_without_citation_becomes_not_found():
    llm = ScriptedLLM(lambda schema, prompt: AnswerOutput(evidence_ids=[], answer="June 27", confidence=0.9))
    answer = QAAgent(llm, Tracer(enabled=False)).answer(QAPlan(question="Due?", bids=["B"]), [ev("c1", "due June 27")])
    assert answer.answer == NOT_FOUND and len(llm.calls) == 2  # asked once more to cite, then refused


def test_answer_accepts_percent_confidence_and_normalises_not_found():
    llm = ScriptedLLM(
        lambda schema, prompt: AnswerOutput.model_validate(
            {"evidence_ids": ["E1"], "answer": "Not found in documents.", "confidence": 100}
        )
    )
    answer = QAAgent(llm, Tracer(enabled=False)).answer(QAPlan(question="Bond?", bids=["B"]), [ev("c1", "x")])
    assert answer.answer == NOT_FOUND and answer.citations == [] and answer.confidence == 0.0
    assert AnswerOutput.model_validate({"evidence_ids": [], "answer": "a", "confidence": 85}).confidence == 0.85


def test_answer_falls_back_to_passages_when_the_model_returns_unusable_output():
    class BrokenLLM(ScriptedLLM):
        def structured(self, system, prompt, schema, step=None, timeout_s=None, max_tokens=None):
            raise LLMError("AnswerOutput: bad JSON")

    evidence = [ev("c1", "Proposals are due June 27, 2024 at 2:00 PM CST.")]
    answer = QAAgent(BrokenLLM(None), Tracer(enabled=False)).answer(
        QAPlan(question="When are proposals due?", bids=["B"]), evidence
    )
    assert answer.mode == "passages" and answer.answer.startswith("The model could not answer")


def test_answer_replaces_inline_ids_with_citations():
    llm = ScriptedLLM(
        lambda schema, prompt: AnswerOutput(evidence_ids=["E1"], answer="Due June 27 [E1].", confidence=0.9)
    )
    answer = QAAgent(llm, Tracer(enabled=False)).answer(QAPlan(question="Due?", bids=["B"]), [ev("c1", "due June 27")])
    assert answer.answer == "Due June 27 [RFP.pdf, p. 1]."
    assert [c.label() for c in answer.citations] == ["RFP.pdf, p. 1"]


def test_validator_rejects_hedged_and_unsupported_negative_values():
    assert grounding_problems("W-9 (implied by tax forms)", "W-9 tax forms") == [
        "value is hedged or inferred, not stated in the documents"
    ]
    assert grounding_problems("Not required", "Insurance and bonds as noted elsewhere.")
    assert grounding_problems("Not required", "A bid bond is not required for this solicitation.") == []


def test_extraction_treats_placeholder_strings_as_missing_and_strips_ids():
    from agents.extraction_agent import _clean_value

    assert [_clean_value(v) for v in ("null", "Not specified.", "N/A", "Not found in documents")] == [None] * 4
    assert _clean_value("Mercury Affidavit (E5), Contract Affidavit [E6]") == "Mercury Affidavit, Contract Affidavit"
    assert _clean_value("None") == "None"  # a real answer for Pre Bid Meeting, checked by the validator


def test_summary_ignores_fields_the_validator_rejected():
    from agents.report_agent import ReportAgent

    seen = []

    def handler(schema, prompt):
        seen.append(prompt)
        return SummaryOutput(summary="Summary.")

    fields = {
        "Title": FieldResult(name="Title", value="Laptops", status="passed", confidence=0.9),
        "Bid Bond Requirement": FieldResult(name="Bid Bond Requirement", value="Not required", status="failed"),
    }
    ReportAgent(ScriptedLLM(handler), Tracer(enabled=False)).summarise("B", fields, [])
    assert "Laptops" in seen[0] and "Not required" not in seen[0]


def test_a_crashing_search_fails_its_fields_without_stopping_the_run(indexed, tmp_path):
    store, folder = indexed
    store.index_folder(folder)

    class BrokenSearcher(HybridSearcher):
        def search(self, query, filters=None, top_k=None, **overrides):
            raise RuntimeError("index unavailable")

    orch = Orchestrator(
        store=store, llm=ScriptedLLM(scripted_bid), searcher=BrokenSearcher(store), outputs_dir=tmp_path / "out"
    )
    record = orch.extract_bid("BidT", write=False)
    assert record.fields["Due Date"].value is None
    assert "index unavailable" in record.fields["Due Date"].notes
    assert any("index unavailable" in e for e in record.run["errors"])


def test_llm_check_explains_missing_ollama():
    from agents.llm import OllamaLLM

    with pytest.raises(LLMError, match="not reachable"):
        OllamaLLM(base_url="http://127.0.0.1:9").check()


def test_evidence_keeps_the_paragraph_around_a_fact_line():
    from agents.retrieval_agent import RetrievalAgent

    fact = ev("f1", "The new due date is July 9, 2024 at 2:00 PM CST.")
    paragraph = ev("p1", "Addendum 2 extends the due date. The new due date is July 9, 2024 at 2:00 PM CST. Sign it.")
    other = ev("o1", "Delivery within 30 days of award.")
    chosen = RetrievalAgent(None, Tracer(enabled=False))._fill([[fact, paragraph], [other]], budget=200)
    assert [e.chunk_id for e in chosen] == ["o1", "p1"]  # the paragraph replaced its fact line
    assert [e.chunk_id for e in RetrievalAgent(None, Tracer(enabled=False))._fill([[paragraph, fact]], 200)] == ["p1"]
