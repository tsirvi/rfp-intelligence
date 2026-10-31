import json

import pytest
from fastapi.testclient import TestClient

import api.app as api_app
from search.hybrid import HybridSearcher, SearchConfig
from search.store import IndexStore
from tests.test_pdf_parser import NO_OCR, make_pdf
from tests.test_search import KeywordReranker
from tests.test_store import HashEmbedder


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.pipeline.section", lambda name: NO_OCR)
    state = api_app.State.__new__(api_app.State)
    state.store = IndexStore(tmp_path / "index", embedder=HashEmbedder())
    state.searcher = HybridSearcher(state.store, reranker=KeywordReranker(), config=SearchConfig(top_k=3))
    state.jobs = {}
    state.index_lock = api_app.threading.Lock()
    state.extract_jobs, state.extract_lock, state._orchestrator = {}, api_app.threading.Lock(), None
    monkeypatch.setattr(api_app, "state", state)
    monkeypatch.setattr(api_app, "index_roots", lambda: [tmp_path.resolve()])
    return TestClient(api_app.app)  # not used as a context manager, so startup indexing does not run


@pytest.fixture
def bid(tmp_path):
    folder = tmp_path / "BidQ"
    folder.mkdir()
    make_pdf(folder / "RFP.pdf", [[(72, 100, "Proposals are due on June 27, 2024 at 2:00 PM.")]])
    make_pdf(folder / "Addendum 1.pdf", [[(72, 100, "Addendum 1: the closing date moves to July 9, 2024.")]])
    return folder


def test_index_then_search_with_citations(client, bid):
    job = client.post("/index", json={"path": str(bid)})
    assert job.status_code == 202
    # TestClient runs background tasks before returning, so the job is already finished.
    status = client.get(f"/index/{job.json()['job_id']}").json()
    assert status["status"] == "done" and sorted(status["report"]["added"]) == ["Addendum 1.pdf", "RFP.pdf"]

    res = client.get("/search", params={"q": "closing date", "bid_id": "BidQ", "doc_type": ["addendum"]}).json()
    assert [h["file_name"] for h in res["hits"]] == ["Addendum 1.pdf"]
    hit = res["hits"][0]
    assert hit["citation"] == "Addendum 1.pdf, p. 1" and hit["addendum_number"] == 1
    assert res["filters"] == {"bid_id": ["BidQ"], "doc_type": ["addendum"]}

    bids = client.get("/bids").json()
    assert bids[0]["bid_id"] == "BidQ" and len(bids[0]["documents"]) == 2
    assert client.get("/health").json()["bids"] == ["BidQ"]


def test_index_rejects_missing_folder(client, tmp_path):
    assert client.post("/index", json={"path": str(tmp_path / "nope")}).status_code == 400


def test_unknown_job_and_validation(client):
    assert client.get("/index/doesnotexist").status_code == 404
    assert client.get("/search", params={"q": "x"}).status_code == 422


def test_search_on_empty_index_returns_no_hits(client):
    assert client.get("/search", params={"q": "due date"}).json()["hits"] == []


def test_ask_and_extract_validate_input(client, bid):
    assert client.post("/ask", json={"question": "When is it due?"}).status_code == 409  # nothing indexed
    client.post("/index", json={"path": str(bid)})
    assert client.post("/ask", json={"question": "When is it due?", "bids": ["Nope"]}).status_code == 404
    assert client.post("/extract", json={"bid_id": "Nope"}).status_code == 404
    assert client.get("/extract/doesnotexist").status_code == 404


def test_search_filters_by_addendum_number_and_accepts_top_k(client, bid):
    client.post("/index", json={"path": str(bid)})
    res = client.get("/search", params={"q": "date", "bid_id": "BidQ", "addendum_number": 1, "top_k": 5}).json()
    assert res["hits"] and all(h["addendum_number"] == 1 for h in res["hits"])
    assert res["filters"]["addendum_number"] == [1]
    assert len(client.get("/search", params={"q": "date", "top_k": 1}).json()["hits"]) == 1


def test_extract_does_not_queue_a_second_run_of_the_same_bid(client, bid, monkeypatch):
    client.post("/index", json={"path": str(bid)})
    monkeypatch.setattr(api_app, "run_extract_job", lambda job: None)  # leave the job queued
    first = client.post("/extract", json={"bid_id": "BidQ"}).json()
    again = client.post("/extract", json={"bid_id": "BidQ"}).json()
    assert again["job_id"] == first["job_id"] and len(api_app.state.extract_jobs) == 1


def test_index_refuses_folders_outside_the_allowed_roots(client, tmp_path):
    outside = tmp_path.parent / "elsewhere"
    assert client.post("/index", json={"path": str(outside)}).status_code == 403
    assert client.post("/index", json={"path": "/etc"}).status_code == 403


def test_compare_go_no_go_and_usage(client, bid, tmp_path, monkeypatch):
    from agents.orchestrator import Orchestrator

    client.post("/index", json={"path": str(bid)})
    out = tmp_path / "out"
    out.mkdir()
    source = [{"file": "RFP.pdf", "page": 1}]
    fields = {"Due Date": {"value": "2024-06-27 14:00 CST", "status": "passed", "sources": source}}
    for b in ("BidQ", "BidR"):
        (out / f"{b}.json").write_text(json.dumps({"fields": fields}))
    state = api_app.state
    state._orchestrator = Orchestrator(
        store=state.store, llm=None, searcher=state.searcher, outputs_dir=out, trace=False
    )

    assert client.post("/compare", json={"bids": ["BidQ", "Nope"]}).status_code == 404
    report = client.post("/compare", json={"bids": ["BidQ", "BidR"], "summary": False}).json()
    assert report["fields"]["Due Date"]["BidQ"]["value"] == "2024-06-27 14:00 CST"
    assert "| Due Date |" in report["markdown"]

    decision = client.get("/go-no-go/BidQ").json()
    assert decision["decision"] in ("GO", "REVIEW", "NO-GO") and decision["checks"]
    usage = client.get("/usage").json()
    assert usage["by_kind"]["compare"]["runs"] == 1 and usage["by_kind"]["go_no_go"]["runs"] == 1
