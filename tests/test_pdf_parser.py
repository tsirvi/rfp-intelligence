from pathlib import Path

import pymupdf

from ingestion.pdf_parser import clean_table, layout_lines, parse_pdf
from ingestion.pipeline import file_hash, ingest_folder
from schemas.documents import DocType

NO_OCR = {"ocr_enabled": False}


def make_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> Path:
    """pages: list of (x, y, text) placements."""
    doc = pymupdf.open()
    for placements in pages:
        page = doc.new_page()
        for x, y, text in placements:
            page.insert_text((x, y), text, fontsize=10)
    doc.save(path)
    return path


def rfp_pages(n: int = 3) -> list[list[tuple[float, float, str]]]:
    return [
        [
            (72, 40, "Request For Proposal 77 Widgets"),
            (72, 100, "Solicitation Due"),
            (300, 100, "27-JUN-2024 14:00:00"),
            (72, 115, f"Body paragraph on page {i}."),
            (72, 800, f"Page {i} of {n}"),
        ]
        for i in range(1, n + 1)
    ]


def test_layout_lines_keeps_label_and_value_on_one_row(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", rfp_pages(1))
    lines = layout_lines(pymupdf.open(pdf)[0])
    assert "Solicitation Due | 27-JUN-2024 14:00:00" in lines


def test_layout_lines_does_not_split_justified_text_into_columns(tmp_path):
    words = ["in", "to", "submit", "a", "bid"]
    pdf = make_pdf(tmp_path / "j.pdf", [[(72 + 60 * i, 100, w) for i, w in enumerate(words)]])
    assert layout_lines(pymupdf.open(pdf)[0]) == ["in to submit a bid"]


def test_parse_pdf_strips_running_header_and_footer(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "rfp.pdf", rfp_pages(3)), NO_OCR)
    assert parsed.running_header == "Request For Proposal 77 Widgets"
    for page in parsed.pages:
        assert "Request For Proposal" not in page.text
        assert "of 3" not in page.text
        assert f"Body paragraph on page {page.page_number}." in page.text


def test_parse_pdf_reports_empty_page(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "blank.pdf", [[]]), NO_OCR)
    assert parsed.pages[0].text == ""
    assert any("empty page" in w for w in parsed.warnings)


def test_clean_table_drops_layout_boxes_and_keeps_grid_of_real_tables():
    assert clean_table([["LENGTH OF CONTRACT", ""], ["Three years.", None]]) is None
    price = clean_table([["Line", "Qty", "Unit Price"], ["1 Laptop", "1", None], ["2 Dock", "1", ""]])
    assert price == [["Line", "Qty", "Unit Price"], ["1 Laptop", "1", ""], ["2 Dock", "1", ""]]
    # Merged-cell form grid: the first row itself is mostly empty, so cells are compacted.
    form = clean_table([["Name:", None, "", None, "Jo"], [None, "Phone:", "", "123", ""], ["", "", "Fax", "", ""]])
    assert form == [["Name:", "Jo"], ["Phone:", "123"], ["Fax"]]


def test_ingest_folder_handles_mixed_good_bad_and_unsupported_files(tmp_path):
    bid = tmp_path / "BidX"
    bid.mkdir()
    make_pdf(bid / "Main RFP.pdf", rfp_pages(3))
    make_pdf(bid / "Addendum 1.pdf", [[(72, 100, "The due date is extended.")]])
    (bid / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (bid / "notes.docx").write_bytes(b"ignored")
    (bid / ".DS_Store").write_bytes(b"ignored")

    result = ingest_folder(bid, cfg=NO_OCR)

    assert result.bid_id == "BidX"
    by_name = {d.file_name: d for d in result.documents}
    assert by_name["Main RFP.pdf"].doc_type == DocType.RFP
    assert by_name["Addendum 1.pdf"].doc_type == DocType.ADDENDUM
    assert by_name["Addendum 1.pdf"].addendum_number == 1
    assert [f.file_name for f in result.failures] == ["broken.pdf"]
    assert result.skipped == ["notes.docx"]


def test_ingest_folder_skips_files_already_indexed(tmp_path):
    bid = tmp_path / "BidY"
    bid.mkdir()
    old = make_pdf(bid / "RFP.pdf", rfp_pages(1))
    make_pdf(bid / "Addendum 1.pdf", [[(72, 100, "New date.")]])

    result = ingest_folder(bid, known_files={old.name: file_hash(old)}, cfg=NO_OCR)

    assert result.unchanged == ["RFP.pdf"]
    assert [d.file_name for d in result.documents] == ["Addendum 1.pdf"]
