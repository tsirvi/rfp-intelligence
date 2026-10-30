import pytest

from eval.run_eval import is_match, normalise, score_question
from search.expansion import expand_query
from search.hybrid import HybridSearcher, SearchConfig, rrf_fuse
from search.store import IndexStore
from tests.test_pdf_parser import NO_OCR, make_pdf
from tests.test_store import HashEmbedder


class KeywordReranker:
    """Scores a passage by how many query words it contains; records calls."""

    def __init__(self):
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        words = set(query.lower().split())
        return [float(sum(w in t.lower() for w in words)) for t in texts]


@pytest.fixture
def searcher(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.pipeline.section", lambda name: NO_OCR)
    store = IndexStore(tmp_path / "index", embedder=HashEmbedder())
    for bid, pages in {
        "BidA": {
            "RFP.pdf": "Solicitation due 27-JUN-2024 at 2:00 PM.",
            "Addendum 2.pdf": "The new closing date is July 9, 2024.",
            "Specs.pdf": "Latitude 5550 SKU 210-BLYZ with 16GB memory.",
        },
        "BidB": {"RFP.pdf": "Proposals are due June 10, 2024. Warranty three years."},
    }.items():
        folder = tmp_path / bid
        folder.mkdir()
        for name, text in pages.items():
            make_pdf(folder / name, [[(72, 100, text)]])
        store.index_folder(folder)
    config = SearchConfig(top_k_dense=5, top_k_bm25=5, rerank_candidates=10, top_k=3)
    return HybridSearcher(store, reranker=KeywordReranker(), config=config)


def test_rrf_rewards_agreement_between_rankings():
    scores = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == scores["b"] > scores["c"] == scores["d"]
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)


def test_expansion_prefers_longest_phrase_and_keeps_original_first():
    variants = expand_query("What is the submission deadline?")
    assert variants[0] == "What is the submission deadline?"
    assert "What is the due date?" in variants
    assert expand_query("Laptop SKU 210-BLYZ") == ["Laptop SKU 210-BLYZ"]


def test_expansion_finds_differently_worded_document(searcher):
    # "closing date" appears only in the addendum; the query says "deadline".
    plain = searcher.search("deadline", {"bid_id": "BidA"}, expand=False, rerank=False)
    expanded = searcher.search("deadline", {"bid_id": "BidA"}, rerank=False)
    assert len(expanded.variants) > 1
    assert not any(h.chunk.file_name == "Addendum 2.pdf" and h.bm25_rank for h in plain.hits)
    assert any(h.chunk.file_name == "Addendum 2.pdf" and h.bm25_rank for h in expanded.hits)


def test_filters_restrict_to_one_bid(searcher):
    hits = searcher.search("due date warranty", {"bid_id": "BidB"}).hits
    assert hits and all(h.chunk.bid_id == "BidB" for h in hits)


def test_rerank_orders_by_reranker_score_and_is_cached(searcher):
    result = searcher.search("SKU 210-BLYZ memory", {"bid_id": "BidA"}, rerank=True)
    assert result.hits[0].chunk.file_name == "Specs.pdf"
    assert result.hits[0].rerank_score == max(h.rerank_score for h in result.hits)
    calls = searcher.reranker.calls
    searcher.search("SKU 210-BLYZ memory", {"bid_id": "BidA"}, rerank=True)
    assert searcher.reranker.calls == calls


def test_near_duplicates_from_one_document_are_collapsed(searcher):
    # The RFP yields a text chunk and a fact chunk with the same sentence; only one is returned.
    hits = searcher.search("solicitation due", {"bid_id": "BidA"}, top_k=5).hits
    assert [h.chunk.file_name for h in hits].count("RFP.pdf") == 1
    with_dupes = searcher.search("solicitation due", {"bid_id": "BidA"}, top_k=5, dedupe=False).hits
    assert [h.chunk.file_name for h in with_dupes].count("RFP.pdf") == 2


def test_single_retriever_configs(searcher):
    bm25_only = searcher.search("210-BLYZ", dense=False, expand=False, rerank=False)
    assert all(h.dense_rank is None for h in bm25_only.hits)
    dense_only = searcher.search("210-BLYZ", bm25=False, expand=False, rerank=False)
    assert all(h.bm25_rank is None for h in dense_only.hits)


def test_eval_matching_and_scoring(searcher):
    chunks = [h.chunk for h in searcher.search("closing date", {"bid_id": "BidA"}, top_k=5).hits]
    passage = {"file": "addendum", "contains": "new  closing date | is July 9"}
    assert normalise("A | b\n c") == "a b c"
    assert any(is_match(c, passage) for c in chunks)
    scored = score_question(chunks, [passage, {"file": "nope", "contains": "x"}])
    assert scored["recall@5"] == 0.5 and scored["rr"] > 0
