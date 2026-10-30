from ingestion.cleaning import (
    merge_lone_bullets,
    normalise_whitespace,
    strip_repeated_edges,
    unwrap_lines,
)


def test_normalise_whitespace_collapses_spaces_ligatures_and_blank_lines():
    assert normalise_whitespace("ofﬁce  hours  \n\n\n\nnext ") == "office hours\n\nnext"


def test_merge_lone_bullets():
    assert merge_lone_bullets(["•", "Storage: 64 GB", "plain"]) == ["• Storage: 64 GB", "plain"]


def test_unwrap_joins_broken_sentences_and_keeps_hyphens():
    lines = ["It is highly encouraged to attend the pre-", "proposal meeting and", "ask questions."]
    assert unwrap_lines(lines) == ["It is highly encouraged to attend the pre-proposal meeting and ask questions."]


def test_unwrap_leaves_column_rows_bullets_and_sentence_ends_alone():
    lines = ["Solicitation Due | 27-JUN-2024", "next line", "Done.", "lowercase start", "• item"]
    assert unwrap_lines(lines) == ["Solicitation Due | 27-JUN-2024", "next line", "Done.", "lowercase start", "• item"]


def test_strip_repeated_edges_removes_headers_footers_with_varying_page_numbers():
    body = [[f"Question {q} about item {n}?" for q in range(1, 7)] for n in range(1, 4)]
    pages = [["ADDENDUM No. 1", "RFP X-1 Devices", *body[n - 1], f"Page {n} | 3"] for n in range(1, 4)]
    cleaned, header = strip_repeated_edges(pages)
    assert cleaned == body
    assert header == "ADDENDUM No. 1 RFP X-1 Devices"


def test_strip_repeated_edges_keeps_body_lines_that_differ_only_by_numbers():
    pages = [["Title", f"Answer: {n} years", f"Item {n}", f"Page {n}"] for n in range(1, 4)]
    cleaned, _ = strip_repeated_edges(pages)
    assert [p[0] for p in cleaned] == ["Answer: 1 years", "Answer: 2 years", "Answer: 3 years"]


def test_strip_repeated_edges_ignores_short_documents():
    pages = [["Header", "a"], ["Header", "b"]]
    assert strip_repeated_edges(pages) == (pages, None)
