"""Comparison report, go / no-go recommendation, semantic Q&A cache and the cost/latency ledger."""

import json

import pytest

from agents.comparison_agent import ComparisonAgent, to_markdown
from agents.go_no_go_agent import GoNoGoAgent, _contract_years
from agents.qa_cache import SemanticCache
from agents.tracing import Tracer
from agents.usage import record_run, summary
from schemas.extraction import AnswerOutput
from tests.test_agents import ScriptedLLM
from tests.test_store import HashEmbedder

SOURCE = [{"file": "RFP.pdf", "page": 2}]


def passed(value: str) -> dict:
    return {"value": value, "status": "passed", "sources": SOURCE, "confidence": 0.9, "notes": ""}


@pytest.fixture
def records(tmp_path):
    a = {
        "Due Date": passed("2024-07-09 14:00 CST"),
        "Product": passed("Dell Latitude laptops (300)"),
        "MFG for Registration": passed("Dell"),
        "Bid Bond Requirement": passed("5% of the bid amount"),
        "Installation": passed("Yes: imaging and asset tagging"),
        "Bid Submission Type": passed("Electronic submission through the e-procurement portal"),
        "Delivery Date": passed("Within 45 days of award"),
        "Term of Bid": passed("Initial term 3 years; Renewal 1: 1 year"),
        "Any Additional Documentation Required": passed("W-9; Certificate of insurance"),
    }
    b = {
        "Due Date": passed("2024-06-10 14:00 EDT"),
        "Product": passed("Office chairs"),
        "Bid Bond Requirement": {"value": "10%", "status": "failed", "sources": SOURCE},
    }
    for bid, fields in (("A", a), ("B", b)):
        (tmp_path / f"{bid}.json").write_text(json.dumps({"fields": fields}))
    return tmp_path


PROFILE = {
    "company": "Test Co",
    "evaluation_date": "2024-06-01",
    "min_days_to_prepare": 7,
    "product_keywords": ["laptop"],
    "authorized_manufacturers": ["Dell"],
    "max_bid_bond_percent": 5,
    "provides_installation": True,
    "submission_methods": ["portal"],
    "min_delivery_days": 30,
    "max_contract_years": 5,
    "documents_we_provide": ["W-9", "insurance"],
}


def test_comparison_table_differences_and_cited_summary(records):
    def handler(schema, prompt):
        assert schema is AnswerOutput and "[E1] A | Due Date" in prompt
        return AnswerOutput(evidence_ids=["E1"], answer="A closes later (E1).", confidence=0.8)

    result = ComparisonAgent(ScriptedLLM(handler), Tracer(enabled=False)).compare(["A", "B"], records)
    assert result.fields["Due Date"]["B"].value == "2024-06-10 14:00 EDT"
    assert result.fields["Bid Bond Requirement"]["B"].status == "failed"
    assert result.differences[0] == "Deadlines: B (2024-06-10 14:00 EDT), then A (2024-07-09 14:00 CST)."
    assert "Check by hand for B: Bid Bond Requirement" in " ".join(result.differences)
    assert result.summary == "A closes later [RFP.pdf, p. 2]." and result.summary_sources == ["RFP.pdf, p. 2"]
    assert "| Bid Bond Requirement | 5% of the bid amount [RFP.pdf, p. 2] | ⚠️ extracted value not validated |" in (
        to_markdown(result)
    )


def test_comparison_drops_an_uncited_summary_and_needs_records(records):
    llm = ScriptedLLM(lambda schema, prompt: AnswerOutput(evidence_ids=[], answer="A is better.", confidence=0.9))
    assert ComparisonAgent(llm, Tracer(enabled=False)).compare(["A", "B"], records).summary is None
    with pytest.raises(ValueError, match="No extracted record for C"):
        ComparisonAgent(None, Tracer(enabled=False)).compare(["A", "C"], records)


def test_go_when_every_capability_check_passes(records):
    decision = GoNoGoAgent(Tracer(enabled=False), PROFILE).assess("A", records)
    assert decision.decision == "GO", [c for c in decision.checks if c.outcome != "pass"]
    assert {c.name for c in decision.checks} >= {"Deadline", "Product fit", "Bid bond", "Contract term"}
    assert all(c.source == "RFP.pdf, p. 2" for c in decision.checks)


def test_no_go_and_review_reasons(records):
    assert GoNoGoAgent(Tracer(enabled=False), PROFILE | {"max_bid_bond_percent": 2}).assess("A", records).decision == (
        "NO-GO"
    )
    late = GoNoGoAgent(Tracer(enabled=False), PROFILE | {"evaluation_date": "2024-07-05"}).assess("A", records)
    assert late.decision == "NO-GO" and "Only 4 days left" in late.summary
    other = GoNoGoAgent(Tracer(enabled=False), PROFILE).assess("B", records)
    outcomes = {c.name: c.outcome for c in other.checks}
    assert outcomes["Product fit"] == "fail" and outcomes["Bid bond"] == "review"  # unvalidated value: review
    no_install = GoNoGoAgent(Tracer(enabled=False), PROFILE | {"provides_installation": False}).assess("A", records)
    assert {c.name: c.outcome for c in no_install.checks}["Installation"] == "fail"


def test_contract_years_reads_totals_and_renewals():
    assert _contract_years("Initial Term: 3 years; Renewal 1: 1 year; Renewal 2: 1 year; Total max 5 years") == 5
    assert _contract_years("2 years from award, with option to renew for 2 additional 1-year periods") == 4
    assert _contract_years("Initial term 3 years; Renewal 1: 1 year") == 4
    assert _contract_years("one-off purchase") is None


def test_semantic_cache_matches_same_question_and_bids_only(tmp_path):
    cache = SemanticCache(tmp_path / "cache.json", HashEmbedder(), threshold=0.94)
    cache.store("When is Bid1 due?", ["Bid1"], "v1", {"answer": "9 July", "mode": "llm"})
    assert cache.lookup("When is Bid1 due?", ["Bid1"], "v1")[0]["answer"] == "9 July"
    assert cache.lookup("When is Bid1 due?", ["Bid2"], "v1") is None  # other bids
    assert cache.lookup("When is Bid1 due?", ["Bid1"], "v2") is None  # index or record changed
    assert cache.lookup("What colour are the chairs?", ["Bid1"], "v1") is None
    cache.store("What changed in Addendum 1?", ["Bid1"], "v1", {"answer": "x", "mode": "llm"})
    assert cache.lookup("What changed in Addendum 2?", ["Bid1"], "v1") is None  # numbers must match
    reloaded = SemanticCache(tmp_path / "cache.json", HashEmbedder(), threshold=0.94)
    assert reloaded.lookup("When is Bid1 due?", ["Bid1"], "v1") is not None  # persisted


def test_usage_ledger_summarises_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.usage.section", lambda name: {"prompt_per_1k": 1.0, "output_per_1k": 2.0})
    tracer = Tracer(enabled=False)
    tracer.totals.update(llm_calls=2, prompt_tokens=1000, output_tokens=500)
    entry = record_run(tmp_path, "ask", tracer, ["Bid1"], 12.5, mode="llm", cached=False)
    assert entry["est_cost"] == 2.0
    record_run(tmp_path, "ask", Tracer(enabled=False), ["Bid1"], 0.1, mode="llm", cached=True)
    totals = summary(tmp_path)
    assert totals["runs"] == 2 and totals["ask_cache_hit_rate"] == 0.5
    assert totals["by_kind"]["ask"]["prompt_tokens"] == 1000 and totals["by_kind"]["ask"]["avg_seconds"] == 6.3
