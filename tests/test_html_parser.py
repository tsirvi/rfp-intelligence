from ingestion.html_parser import parse_html

PAGE = """<html><head><title>Bid X | Portal</title><meta name="_csrf" content="SECRET-TOKEN"/>
<script>var user = "SECRET-USER";</script></head>
<body>
<header><ul class="main menu"><li>Search</li></ul></header>
<main>
  <h1>RFP-42 - Rugged Tablets</h1>
  <div class="toolbarActions"><button>Print</button>Share</div>
  <div class="dialog hidden">Are you sure you want to delete this note?</div>
  <h3>Dates</h3>
  <div class="field"><span class="field-label">Closing Date</span><div class="body"><p>07/09/2024 03:00 PM EDT</p></div></div>
  <div class="field no-label"><span class="field-label filler"></span><div class="body"><p>Procurement Office</p></div></div>
  <h3>Contact Information</h3>
  <div class="field no-label"><span class="field-label"></span><div class="body"><p>buyer@example.org</p></div></div>
  <dl><dt>Bid Bond</dt><dd>5% of bid amount</dd></dl>
  <p>For details see <a href="https://example.org/bids/42/full-notice">https://example.org/bids/42/fu...</a></p>
  <input type="hidden" name="_csrf" value="SECRET-TOKEN"/>
</main>
<div class="cookie-banner">We use cookies</div>
</body></html>"""


def test_parse_html_extracts_fields_in_reading_order_and_drops_boilerplate(tmp_path):
    path = tmp_path / "bid.html"
    path.write_text(PAGE, encoding="utf-8")
    result = parse_html(path)

    assert result.title == "RFP-42 - Rugged Tablets"
    assert result.fields["Closing Date"] == "07/09/2024 03:00 PM EDT"
    assert result.fields["Dates"] == "Procurement Office"  # blank label takes the section heading
    assert result.fields["Contact Information"] == "buyer@example.org"
    assert result.fields["Bid Bond"] == "5% of bid amount"
    assert "Closing Date: 07/09/2024 03:00 PM EDT" in result.text
    assert "https://example.org/bids/42/full-notice" in result.text  # truncated link text replaced

    for noise in ("SECRET", "Print", "Share", "cookies", "delete this note", "Search"):
        assert noise not in result.text


def test_parse_html_empty_page_warns(tmp_path):
    path = tmp_path / "empty.html"
    path.write_text("<html><body><script>x()</script></body></html>", encoding="utf-8")
    result = parse_html(path)
    assert result.text == "" and result.warnings
