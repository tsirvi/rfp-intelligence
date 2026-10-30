from schemas.documents import DocType, Document, Page, Table
from search.chunking import chunk_document, count_tokens, is_heading

CFG = {"max_tokens": 60, "overlap_tokens": 15, "min_tokens": 10}


def make_doc(pages: list[str], tables: dict[int, list[Table]] | None = None, **kw) -> Document:
    return Document(
        doc_id="BidX/rfp.pdf",
        bid_id="BidX",
        file_name="rfp.pdf",
        file_hash="h",
        doc_type=kw.pop("doc_type", DocType.RFP),
        pages=[Page(page_number=i + 1, text=t, tables=(tables or {}).get(i + 1, [])) for i, t in enumerate(pages)],
        **kw,
    )


def sentences(n: int, word: str = "delivery") -> str:
    return " ".join(f"Sentence {i} about {word} terms." for i in range(n))


def test_is_heading():
    assert is_heading("PRICING INFORMATION")
    assert is_heading("3.2 Delivery Requirements")
    assert is_heading("Section 4 – Scope of Work")
    assert not is_heading("Section 4 of the contract applies")
    assert not is_heading("Solicitation Due | 27-JUN-2024")
    assert not is_heading("BIDDERS ARE REQUIRED TO ACKNOWLEDGE. THIS ADDENDUM")
    assert not is_heading("The vendor shall deliver all devices.")


def test_sections_start_new_chunks_and_label_them():
    doc = make_doc([f"SCOPE OF WORK\n{sentences(3)}\n\nPRICING INFORMATION\n{sentences(3, 'price')}"])
    chunks = chunk_document(doc, CFG)
    assert [c.section for c in chunks] == ["SCOPE OF WORK", "PRICING INFORMATION"]
    assert chunks[1].text.startswith("PRICING INFORMATION")
    assert "price" not in chunks[0].text


def test_wrapped_heading_is_joined():
    doc = make_doc(["SCOPE AND SPECIFICATIONS OF THE\nPROPOSAL\n" + sentences(2)])
    assert chunk_document(doc, CFG)[0].section == "SCOPE AND SPECIFICATIONS OF THE PROPOSAL"


def test_chunks_respect_budget_overlap_and_pages():
    paragraphs = [sentences(2, f"topic{i}") for i in range(6)]  # ~12 tokens each, 72 in total
    doc = make_doc(["\n\n".join(paragraphs), sentences(2)])
    chunks = chunk_document(doc, CFG)
    assert all(count_tokens(c.text) <= CFG["max_tokens"] + CFG["overlap_tokens"] for c in chunks)
    assert len(chunks) > 1
    # The last paragraph of one chunk is repeated at the start of the next.
    assert chunks[1].text.split("\n")[0] == chunks[0].text.split("\n")[-1]
    assert chunks[-1].page_end == 2
    assert chunks[0].citation == "rfp.pdf, p. 1"


def test_oversized_paragraph_is_split_at_sentences():
    doc = make_doc([sentences(40)])  # one ~240-token paragraph
    chunks = chunk_document(doc, CFG)
    assert len(chunks) >= 4
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_heading_at_chunk_end_moves_to_next_chunk():
    # Below min_tokens the heading does not flush, so it is appended; the next paragraph then overflows.
    doc = make_doc([sentences(7) + "\nPRICING RESPONSE\n" + sentences(4, "price")])
    chunks = chunk_document(doc, {**CFG, "min_tokens": 60})
    assert len(chunks) == 2
    assert not chunks[0].text.endswith("PRICING RESPONSE")
    assert any(c.text.startswith("PRICING RESPONSE") for c in chunks)


def test_tables_become_markdown_chunks_with_header_repeated():
    rows = [["Line", "Item", "Qty"]] + [[str(i), f"Laptop model {i}", "10"] for i in range(30)]
    doc = make_doc(["PRICE SCHEDULE\nSee table."], tables={1: [Table(page_number=1, rows=rows)]})
    tables = [c for c in chunk_document(doc, CFG) if c.kind == "table"]
    assert len(tables) > 1
    assert all(t.text.startswith("| Line | Item | Qty |") for t in tables)
    assert all(t.section == "PRICE SCHEDULE" for t in tables)


def test_context_carries_bid_type_and_section():
    doc = make_doc([sentences(1)], doc_type=DocType.ADDENDUM, addendum_number=2, title="Addendum No. 2")
    chunk = chunk_document(doc, CFG)[0]
    assert chunk.context == "BidX | addendum 2 | Addendum No. 2"
    assert chunk.indexed_text.startswith(chunk.context)
    assert chunk.to_metadata()["addendum_number"] == 2


def test_empty_document_has_no_chunks():
    assert chunk_document(make_doc(["", "  "]), CFG) == []


def test_fact_chunks_pick_out_dates_and_contacts_with_their_labels():
    page = (
        "PURCHASING DEPARTMENT\n9400 North Central Expressway, Dallas\n"
        "Buyer ALZATE, JASMINE | Email JALZATE@dallasisd.org\n"
        "Questions Due (Closing) Date and Time:\n06/01/2024 at 2:00 PM EDT\n\n" + sentences(3)
    )
    facts = [c for c in chunk_document(make_doc([page]), CFG) if c.kind == "fact"]
    expected = [
        "9400 North Central Expressway, Dallas",
        "Buyer ALZATE, JASMINE | Email JALZATE@dallasisd.org",
        "Questions Due (Closing) Date and Time:",
        "06/01/2024 at 2:00 PM EDT",
    ]
    assert [f.text for f in facts] == ["\n".join(expected)]
    assert facts[0].section == "PURCHASING DEPARTMENT"
