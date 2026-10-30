# Bid3 answer key (fictional test bid)

`python eval/make_sample_bid.py` writes `samples/Bid3` (outside `data/`, so the API does not index it at start-up; add it from the UI sidebar or with `python -m search index samples/Bid3`). Every value below is stated in those documents, so an
extraction of Bid3 can be checked field by field. The organisation, people, brand and contact details are
invented.

| Field | Expected value | Where |
|---|---|---|
| Bid Number | RUSD-2025-014 | RFP p. 1 |
| Title | Classroom Interactive Projectors and Installation | RFP p. 1 |
| company_name | Riverton Unified School District | RFP p. 1 |
| contact_info | Maria Chen, Senior Buyer, mchen@riverton-usd.example, (555) 014-2290 | RFP p. 1 |
| Due Date | **2025-03-21 15:00 PST** (Addendum 1; the RFP said 2025-03-14) | Addendum 1 p. 1 |
| Pre Bid Meeting | 2025-02-25 10:00 AM PST, District Office Room 204 (video link available); non-mandatory | RFP p. 1–2 |
| Delivery Date | Within 60 calendar days after the purchase order (delivered and installed) | RFP, section 6 |
| Bid Submission Type | Sealed hard copy: 1 original, 2 copies and a USB drive, by hand or courier; no email or fax | RFP, section 3 |
| Term of Bid | 2 years, plus two optional 1-year renewals | RFP, section 7 |
| Payment Terms | Net 45 after receipt of a correct invoice and acceptance | RFP, section 8 |
| Bid Bond Requirement | 5% of the total proposal amount (bid bond, certified or cashier's check) | RFP, section 9 |
| Any Additional Documentation Required | Form A signature page; Form B Non-Collusion Affidavit (notarized); W-9; Certificate of Insurance ($1,000,000 per occurrence); 3 K-12 references; signed addendum acknowledgements | RFP, section 12 |
| Contract or Cooperative to use | Valley Schools Purchasing Cooperative | RFP, section 10 |
| MFG for Registration | Lumetra (authorized reseller, with a manufacturer's letter of authorization) | RFP, section 11 |
| Product | **135** interactive laser projectors and **135** wall mounts (Addendum 1; RFP said 120), 40 document cameras | RFP section 4 table; Addendum 1 |
| Model_no | Lumetra LX-770i (or approved equal); WM-200 mount; DC-40 document camera | RFP, section 4 |
| Part_no | LMT-770I-NA; LMT-WM200; LMT-DC40 | RFP, section 4 table |
| Product Specification | 5,000 lumens laser; WUXGA 1920 x 1200; 10-point touch and pen; 2 HDMI, USB-C 65 W, LAN; 3-year warranty, 5 years/20,000 h on the light source | RFP, section 5 |
| Installation | Yes: install in 40 classrooms (mounting, cabling, configuration), outside teaching hours, plus 2 h teacher training at each of 5 schools | RFP, sections 4 and 6 |
| Bid Summary | (generated) should mention the projectors, the district, the 21 March 2025 due date and installation | — |

## What this bid tests

- **Addendum precedence:** the due date (14 → 21 March) and the projector and mount quantities (120 → 135).
- **OCR:** Form B (the last RFP page) is an image only, so its text is read by OCR.
- **Tables:** the schedule of events and the price list are ruled tables.
- **Headers and footers:** every page repeats a header, a sample-data banner and "Page N of M"; these should be removed.
- **Traps:** the deadline for questions (3 March) is not the due date; the anticipated award (8 April) is not the delivery date; "encouraged but not required" means the pre-proposal conference is non-mandatory.

## Questions to try in the Ask tab

| Question | Expected answer |
|---|---|
| When is Bid3 due? | 21 March 2025, 3:00 PM PST (Addendum 1) |
| Is a bid bond required for Bid3? | Yes, 5% of the total proposal amount |
| What changed in Addendum 1 of Bid3? | Due date extended to 21 March; projectors and mounts increased from 120 to 135 |
| How many projectors does Bid3 need? | 135 (after Addendum 1) |
| What warranty does Bid3 require? | 3 years on the projector; 5 years or 20,000 hours on the light source |
| Can I email my proposal for Bid3? | No: sealed hard copies only; email and fax are not accepted |
| What is the deadline for questions in Bid3? | 3 March 2025, 5:00 PM PST |
| Compare all three bids | A table of the key fields for Bid1, Bid2 and Bid3 |
