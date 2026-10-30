"""Generate a fictional bid folder (samples/Bid3) with a known answer key, to test the pipeline on an unseen bid.

python eval/make_sample_bid.py [--out samples/Bid3]

Everything here is invented: the district, people, brand ("Lumetra"), addresses (.example) and phone numbers
(555). The folder deliberately differs from the two sample bids: sealed paper submission, a 5% bid bond, an
addendum that changes both the due date and a quantity, a price table, repeated headers/footers, and one
scanned (image-only) page that needs OCR. The expected values are in eval/bid3_answer_key.md.
"""

from __future__ import annotations

import argparse
import io
import textwrap
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BANNER = "SAMPLE DOCUMENT - FICTIONAL ORGANISATION, CREATED FOR SOFTWARE TESTING"
WIDTH, HEIGHT, LEFT, TOP, BOTTOM = 612, 792, 60, 90, 730


class Writer:
    """Flows headings, paragraphs and tables onto pages, with a running header and a page footer."""

    def __init__(self, header: str):
        self.doc = pymupdf.open()
        self.header = header
        self.y = BOTTOM  # forces a new page on first use

    def _page(self) -> None:
        self.page = self.doc.new_page(width=WIDTH, height=HEIGHT)
        self.page.insert_text((LEFT, 40), self.header, fontsize=8, fontname="helv")
        self.page.insert_text((LEFT, 52), BANNER, fontsize=7, fontname="helv")
        self.y = TOP

    def _room(self, height: float) -> None:
        if self.y + height > BOTTOM:
            self._page()

    def heading(self, text: str) -> None:
        self._room(40)
        self.y += 10
        self.page.insert_text((LEFT, self.y), text, fontsize=12, fontname="hebo")
        self.y += 20

    def para(self, text: str, bold_label: str | None = None) -> None:
        lines = textwrap.wrap(((bold_label + " ") if bold_label else "") + text, 98)
        self._room(14 * len(lines) + 6)
        for line in lines:
            self.page.insert_text((LEFT, self.y), line, fontsize=10, fontname="helv")
            self.y += 14
        self.y += 6

    def table(self, rows: list[list[str]], widths: list[float]) -> None:
        self._room(20 * len(rows) + 10)
        for r, row in enumerate(rows):
            x = LEFT
            for cell, w in zip(row, widths, strict=True):
                self.page.draw_rect(pymupdf.Rect(x, self.y, x + w, self.y + 20), color=(0, 0, 0), width=0.6)
                self.page.insert_text((x + 4, self.y + 14), cell, fontsize=9, fontname="hebo" if r == 0 else "helv")
                x += w
            self.y += 20
        self.y += 12

    def scanned_page(self, lines: list[str]) -> None:
        """A page that is only a picture of text (as from a scanner): the parser must OCR it."""
        img = Image.new("RGB", (1275, 1650), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
        y = 120
        for line in lines:
            for part in textwrap.wrap(line, 70) or [""]:
                draw.text((110, y), part, fill="black", font=font)
                y += 46
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page = self.doc.new_page(width=WIDTH, height=HEIGHT)
        page.insert_image(page.rect, stream=buf.getvalue())

    def save(self, path: Path) -> None:
        total = len(self.doc)
        for i, page in enumerate(self.doc, start=1):
            page.insert_text((WIDTH / 2 - 30, 770), f"Page {i} of {total}", fontsize=8, fontname="helv")
        self.doc.save(path)


def rfp(path: Path) -> None:
    w = Writer("Riverton Unified School District | RFP RUSD-2025-014")
    w._page()
    w.page.insert_text((LEFT, 140), "REQUEST FOR PROPOSAL RUSD-2025-014", fontsize=18, fontname="hebo")
    w.page.insert_text((LEFT, 170), "Classroom Interactive Projectors and Installation", fontsize=14, fontname="hebo")
    w.y = 210
    w.para("Riverton Unified School District (the District), Purchasing Department, 400 Mill Road, Riverton.")
    w.para("Issue Date: February 10, 2025")
    w.para("Proposal Due Date: March 14, 2025 at 3:00 PM PST")
    w.para("Procurement Contact: Maria Chen, Senior Buyer, mchen@riverton-usd.example, (555) 014-2290")

    w.heading("1. INTRODUCTION")
    w.para(
        "The District invites sealed proposals from qualified vendors to supply and install interactive laser "
        "projectors, wall mounts and document cameras for classrooms at its middle and high schools. The "
        "District will award one contract to the responsive, responsible vendor whose proposal offers the best "
        "value to the District."
    )

    w.heading("2. SCHEDULE OF EVENTS")
    w.table(
        [
            ["Event", "Date and time"],
            ["RFP issued", "February 10, 2025"],
            ["Pre-proposal conference (non-mandatory)", "February 25, 2025, 10:00 AM PST"],
            ["Deadline for written questions", "March 3, 2025, 5:00 PM PST"],
            ["Proposals due", "March 14, 2025, 3:00 PM PST"],
            ["Anticipated award", "April 8, 2025"],
        ],
        [300, 192],
    )
    w.para(
        "A non-mandatory pre-proposal conference will be held on February 25, 2025 at 10:00 AM PST in Room 204 "
        "of the District Office, 400 Mill Road, Riverton. Vendors may also join by video link; the link will be "
        "emailed to vendors who register with the Procurement Contact. Attendance is encouraged but not required."
    )

    w.heading("3. SUBMISSION INSTRUCTIONS")
    w.para(
        "Proposals must be submitted as sealed hard copies: one signed original, two copies and one USB drive "
        "containing a PDF of the complete proposal, in a sealed envelope marked RUSD-2025-014. Proposals must be "
        "delivered by hand or courier to the Purchasing Department, 400 Mill Road, Riverton. Proposals sent by "
        "email or fax will not be accepted. Late proposals will be returned unopened."
    )

    w.heading("4. SCOPE OF WORK")
    w.para(
        "The vendor shall supply the following products, deliver them to the District warehouse and install the "
        "projectors and mounts in 40 classrooms, including mounting, cabling, configuration and removal of "
        "packaging. The vendor shall provide two hours of on-site teacher training at each of the five schools."
    )
    w.table(
        [
            ["Line", "Description", "Part number", "Quantity"],
            ["1", "Lumetra LX-770i interactive laser projector", "LMT-770I-NA", "120"],
            ["2", "Lumetra WM-200 ultra short throw wall mount", "LMT-WM200", "120"],
            ["3", "Lumetra DC-40 document camera", "LMT-DC40", "40"],
        ],
        [40, 250, 120, 82],
    )
    w.para("Brand: Lumetra, model LX-770i, or an approved equal meeting all specifications in Section 5.")

    w.heading("5. TECHNICAL SPECIFICATIONS")
    w.para("Brightness: minimum 5,000 lumens, laser light source.")
    w.para("Resolution: native WUXGA, 1920 x 1200.")
    w.para("Interactivity: finger touch and pen interaction, at least 10 simultaneous touch points.")
    w.para("Connectivity: two HDMI inputs, one USB-C input with 65 W power delivery, wired LAN for remote management.")
    w.para(
        "Warranty: three (3) years parts and labour on the projector, and five (5) years or 20,000 hours, "
        "whichever comes first, on the laser light source."
    )

    w.heading("6. DELIVERY")
    w.para(
        "All products shall be delivered and installed within sixty (60) calendar days after the vendor receives "
        "the District's purchase order. Installation work in classrooms must take place outside teaching hours."
    )

    w.heading("7. CONTRACT TERM")
    w.para(
        "The contract term is two (2) years from the date of award, with the option to renew for two (2) "
        "additional one (1) year periods by mutual written agreement."
    )

    w.heading("8. PAYMENT TERMS")
    w.para(
        "The District will pay undisputed invoices Net 45 days after receipt of a correct invoice and acceptance "
        "of the delivered and installed products. Invoices must quote the purchase order number."
    )

    w.heading("9. BID BOND")
    w.para(
        "Each proposal must be accompanied by a bid bond, certified check or cashier's check in the amount of "
        "five percent (5%) of the total proposal amount, payable to Riverton Unified School District. The bid "
        "bond will be returned to unsuccessful vendors after award."
    )

    w.heading("10. COOPERATIVE PURCHASING")
    w.para(
        "Under the Valley Schools Purchasing Cooperative agreement, other member districts may purchase from the "
        "resulting contract at the same prices and terms, with the vendor's consent."
    )

    w.heading("11. MANUFACTURER AUTHORIZATION")
    w.para(
        "The vendor must be an authorized reseller of Lumetra, or of the manufacturer of the approved equal, and "
        "must include a letter of authorization from the manufacturer with the proposal."
    )

    w.heading("12. REQUIRED FORMS")
    w.para("The following documents must be submitted with the proposal:")
    for item in (
        "Form A - Proposal Signature Page",
        "Form B - Non-Collusion Affidavit (notarized)",
        "IRS Form W-9",
        "Certificate of Insurance showing general liability coverage of at least $1,000,000 per occurrence",
        "Three customer references from K-12 school districts",
        "Signed acknowledgement of every addendum",
    ):
        w.para(f"- {item}")
    w.scanned_page(
        [
            "FORM B - NON-COLLUSION AFFIDAVIT",
            "RFP RUSD-2025-014 Classroom Interactive Projectors and Installation",
            "",
            (
                "The undersigned, being duly sworn, states that this proposal is made without collusion with any "
                "other person, firm or corporation submitting a proposal for the same work."
            ),
            "",
            "Company name: ______________________",
            "Signature: __________________________",
            "Subscribed and sworn before me this ____ day of __________, 2025.",
            "Notary Public: ______________________",
        ]
    )
    w.save(path)


def addendum(path: Path) -> None:
    w = Writer("Riverton Unified School District | RFP RUSD-2025-014 | Addendum No. 1")
    w._page()
    w.page.insert_text((LEFT, 130), "ADDENDUM No. 1", fontsize=16, fontname="hebo")
    w.page.insert_text((LEFT, 155), "RFP RUSD-2025-014 Classroom Interactive Projectors", fontsize=12, fontname="hebo")
    w.y = 190
    w.para("Date of Addendum: March 5, 2025")
    w.para("This addendum modifies the RFP as follows. All other terms and conditions remain unchanged.")
    w.para(
        "1. Due date: the proposal due date is extended. Proposals are now due March 21, 2025 at 3:00 PM PST.",
    )
    w.para(
        "2. Quantity: the quantity of Lumetra LX-770i interactive laser projectors (Line 1, part number "
        "LMT-770I-NA) is increased from 120 to 135. The quantity of wall mounts (Line 2) is increased from 120 "
        "to 135."
    )
    w.heading("QUESTIONS AND ANSWERS")
    w.para("Q1: Must the document cameras be the same brand as the projectors?")
    w.para("A1: No. Any document camera meeting the specifications is acceptable.")
    w.para("Q2: Is the pre-proposal conference recorded?")
    w.para("A2: Yes. The recording is available on request from the Procurement Contact.")
    w.para(
        "Vendors must sign this addendum and include it with their proposal. Failure to acknowledge this "
        "addendum may result in the proposal being considered non-responsive."
    )
    w.para("Acknowledged by (company): __________________   Signature: __________________")
    w.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "samples" / "Bid3"))
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rfp(out / "RFP RUSD-2025-014 Classroom Interactive Projectors.pdf")
    addendum(out / "Addendum 1 RFP RUSD-2025-014.pdf")
    print(f"wrote {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
