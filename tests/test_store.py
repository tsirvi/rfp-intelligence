"""Index store tests with a tiny deterministic embedder, so they run without downloading a model."""

import hashlib
import math

import pytest

from search.store import IndexStore, matches, to_where, tokenize
from tests.test_pdf_parser import NO_OCR, make_pdf


class HashEmbedder:
    """Bag-of-words hashed into 64 dimensions: texts sharing words get similar vectors."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 64
        for word in tokenize(text):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.pipeline.section", lambda name: NO_OCR)
    return IndexStore(tmp_path / "index", embedder=HashEmbedder())


@pytest.fixture
def bid(tmp_path):
    folder = tmp_path / "BidZ"
    folder.mkdir()
    make_pdf(folder / "RFP.pdf", [[(72, 100, "Proposals are due on June 27, 2024 at 2:00 PM.")]])
    make_pdf(folder / "Specs.pdf", [[(72, 100, "Dell Latitude 5550 SKU 210-BLYZ with 16GB memory.")]])
    return folder


def test_tokenize_keeps_skus_whole_and_split():
    assert tokenize("SKU 210-BLYZ, the Wi-Fi") == ["sku", "210-blyz", "210", "blyz", "wi-fi", "wi", "fi"]


def test_filters():
    assert to_where({"bid_id": "B1", "doc_type": None}) == {"bid_id": "B1"}
    assert to_where({"bid_id": "B1", "doc_type": ["rfp"]}) == {
        "$and": [{"bid_id": "B1"}, {"doc_type": {"$in": ["rfp"]}}]
    }
    assert matches({"bid_id": "B1", "doc_type": "rfp"}, {"doc_type": ["rfp", "addendum"]})
    assert not matches({"bid_id": "B1"}, {"bid_id": "B2"})


def test_index_and_search(store, bid):
    report = store.index_folder(bid)
    # One text chunk per file, plus a fact chunk for the RFP's dated line.
    assert sorted(report.added) == ["RFP.pdf", "Specs.pdf"] and report.chunks_added == 3
    assert store.bids() == ["BidZ"]

    hits = store.bm25_search("210-BLYZ")
    assert hits[0][0].file_name == "Specs.pdf"
    assert store.dense_search("when are proposals due", k=1)[0][0].file_name == "RFP.pdf"
    assert [c.file_name for c, _ in store.bm25_search("proposals", filters={"doc_type": "specs"})] == []


def test_reindex_is_incremental(store, bid):
    store.index_folder(bid)
    assert store.index_folder(bid).unchanged == ["RFP.pdf", "Specs.pdf"]

    make_pdf(bid / "RFP.pdf", [[(72, 100, "Proposals are now due on July 9, 2024.")]])
    (bid / "Specs.pdf").unlink()
    report = store.index_folder(bid)
    assert report.updated == ["RFP.pdf"] and report.removed == ["Specs.pdf"]
    chunks = store.chunks({"bid_id": "BidZ"})
    assert {c.file_name for c in chunks} == {"RFP.pdf"} and all("July 9" in c.text for c in chunks)
    assert store.bm25_search("210-BLYZ") == []  # BM25 view refreshed after the change


def test_index_survives_reopen(store, bid, tmp_path):
    store.index_folder(bid)
    reopened = IndexStore(tmp_path / "index", embedder=HashEmbedder())
    assert reopened.count() == 3
    assert reopened.known_hashes("BidZ").keys() == {"RFP.pdf", "Specs.pdf"}
