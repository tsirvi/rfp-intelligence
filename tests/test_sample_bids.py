"""Checks against the provided sample bids. Skipped when data/ does not contain them."""

import shutil
from pathlib import Path

import pymupdf
import pytest

from ingestion.pdf_parser import parse_pdf
from ingestion.pipeline import ingest_folder
from schemas.documents import DocType

DATA = Path(__file__).resolve().parent.parent / "data"
BID1, BID2 = DATA / "Bid1", DATA / "Bid2"
RFP1 = BID1 / "JA-207652 Student and Staff Computing Devices FINAL.pdf"
needs_samples = pytest.mark.skipif(not RFP1.exists(), reason="sample bids not in data/")
# The saved BidNet pages are not in the repository (they hold the downloader's session details).
needs_bid_pages = pytest.mark.skipif(
    not (any(BID1.glob("*.html")) and any(BID2.glob("*.html"))), reason="BidNet pages not in data/"
)


@pytest.fixture(scope="module")
def bid1():
    return {d.file_name: d for d in ingest_folder(BID1, cfg={"ocr_enabled": False}).documents}


@pytest.fixture(scope="module")
def bid2():
    return {d.file_name: d for d in ingest_folder(BID2, cfg={"ocr_enabled": False}).documents}


@needs_samples
@needs_bid_pages
def test_bid1_document_types(bid1):
    types = {name: (d.doc_type, d.addendum_number) for name, d in bid1.items()}
    assert sorted(types.values(), key=str) == sorted(
        [(DocType.ADDENDUM, 1), (DocType.ADDENDUM, 2), (DocType.RFP, None), (DocType.BID_PAGE, None)], key=str
    )


@needs_samples
@needs_bid_pages
def test_bid2_document_types(bid2):
    assert sorted(d.doc_type.value for d in bid2.values()) == [
        "affidavit",
        "affidavit",
        "bid_page",
        "rfp",
        "specs",
    ]


@needs_samples
def test_bid1_rfp_dates_page_is_clean(bid1):
    rfp = bid1[RFP1.name]
    page2 = rfp.pages[1].text
    assert "Solicitation Due | 27-JUN-2024 14:00:00" in page2
    assert "Page 2 of 40" not in page2 and "Dallas ISD rev" not in page2
    assert rfp.running_header.startswith("Request For Proposal 168884")


@needs_samples
def test_bid1_addendum2_new_due_date(bid1):
    add2 = next(d for d in bid1.values() if d.addendum_number == 2)
    assert "July 9, 2024 at 2:00 PM CST" in add2.full_text


@needs_samples
@needs_bid_pages
def test_bid_pages_have_fields_but_no_session_data(bid1, bid2):
    page1 = next(d for d in bid1.values() if d.doc_type == DocType.BID_PAGE)
    page2 = next(d for d in bid2.values() if d.doc_type == DocType.BID_PAGE)
    assert page1.fields["Closing Date"] == "07/09/2024 03:00 PM EDT"
    assert page2.fields["Issuing Organization"] == "State of Maryland Treasurer's Office"
    for text in (page1.full_text, page2.full_text):
        assert "csrf" not in text.lower() and "session_username" not in text


@needs_samples
def test_specs_keep_sku_with_description(bid2):
    specs = bid2["Dell_Laptop_Specs.pdf"].full_text
    assert "Dell Latitude 5550 XCTO Base | 210-BLYZ" in specs
    assert "65W AC adapter, USB Type-C, EcoDesign | 492-BDMN" in specs


@needs_samples
def test_price_schedule_table_keeps_columns(bid1):
    table = bid1[RFP1.name].pages[33].tables[0]
    assert table.rows[0][:4] == ["Line", "Item, Rev", "Target Quantity", "Unit"]
    assert table.rows[1][2:4] == ["1", "Each"]


@needs_samples
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_ocr_recovers_w9_page_without_text_layer(tmp_path):
    single = pymupdf.open()
    single.insert_pdf(pymupdf.open(RFP1), from_page=53, to_page=53)  # W-9: vector outlines, no text
    single.save(tmp_path / "w9.pdf")
    parsed = parse_pdf(tmp_path / "w9.pdf", {"ocr_enabled": True})
    assert parsed.pages[0].ocr
    assert "Taxpayer" in parsed.pages[0].text


@needs_samples
def test_sample_chunks_keep_answers_together(bid1, bid2):
    from search.chunking import chunk_document

    rfp = chunk_document(bid1[RFP1.name])
    due = next(c for c in rfp if "Solicitation Due | 27-JUN-2024 14:00:00" in c.text)
    assert due.page_start <= 2 <= due.page_end
    assert any(c.section == "PRICING RESPONSE" for c in rfp)
    specs = chunk_document(bid2["Dell_Laptop_Specs.pdf"])
    assert any("Dell Latitude 5550 XCTO Base | 210-BLYZ" in c.text for c in specs)
