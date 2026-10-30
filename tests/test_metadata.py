from datetime import date

import pytest

from ingestion.metadata import addendum_number, classify, issue_date_from_fields, issue_date_from_text
from schemas.documents import DocType


@pytest.mark.parametrize(
    ("file_name", "opening", "expected"),
    [
        ("portal page.html", "", DocType.BID_PAGE),
        ("Addendum 1 RFP JA-1 Devices.pdf", "", DocType.ADDENDUM),
        ("Contract_Affidavit.pdf", "", DocType.AFFIDAVIT),
        ("Dell_Laptop_Specs.pdf", "", DocType.SPECS),
        ("PORFP_-_Dell_Laptop_Final.pdf", "", DocType.RFP),
        # Name says nothing: fall back to the opening text.
        ("JA-1 Devices FINAL.pdf", "Request For Proposal 168884 Devices", DocType.RFP),
        ("scan001.pdf", "MERCURY AFFIDAVIT\nI am the authorised representative", DocType.AFFIDAVIT),
        ("doc.pdf", "ADDENDUM No. 3\nThe purpose of this addendum", DocType.ADDENDUM),
        # RFP wording outranks "specifications" in body text.
        ("doc.pdf", "Invitation to Bid. Scope and specifications follow.", DocType.RFP),
        ("notes.pdf", "Meeting minutes", DocType.OTHER),
    ],
)
def test_classify(file_name, opening, expected):
    assert classify(file_name, opening) == expected


@pytest.mark.parametrize(
    ("file_name", "opening", "expected"),
    [
        ("Addendum 2 RFP JA-1.pdf", "", 2),
        ("amendment_no_4.pdf", "", 4),
        ("update.pdf", "ADDENDUM No. 1\nRFP JA-1", 1),
        ("update.pdf", "Addendum Three to the solicitation", 3),
        ("update.pdf", "no number here", None),
    ],
)
def test_addendum_number(file_name, opening, expected):
    assert addendum_number(file_name, opening) == expected


def test_issue_date_from_text_and_fields():
    assert issue_date_from_text("First Advertisement Date/Issue Date | 26-MAY-2024 08:00:00") == date(2024, 5, 26)
    assert issue_date_from_text("PORFP Issue Date: | 05/24/2024 | PROPOSAL DUE") == date(2024, 5, 24)
    assert issue_date_from_text("Date: ________") is None
    assert issue_date_from_fields({"Closing Date": "07/09/2024", "Publication": "05/29/2024 11:16 AM EDT"}) == date(
        2024, 5, 29
    )
