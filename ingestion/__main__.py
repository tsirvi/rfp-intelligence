"""Inspect ingestion output: python -m ingestion data/Bid1 [--dump outputs/Bid1.ingested.json]"""

import argparse
import logging
from pathlib import Path

from ingestion.pipeline import ingest_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("--dump", help="write the full IngestionResult as JSON to this path")
args = parser.parse_args()

result = ingest_folder(args.folder)
for doc in result.documents:
    tables = sum(len(p.tables) for p in doc.pages)
    print(
        f"{doc.doc_type.value:9} add#={doc.addendum_number or '-':<2} pages={len(doc.pages):<3} tables={tables:<3} "
        f"date={doc.document_date} ({doc.document_date_source})  {doc.file_name}\n          title={doc.title!r}"
    )
    for w in doc.warnings:
        print(f"          ! {w}")
for f in result.failures:
    print(f"FAILED   {f.file_name}: {f.error}")
for name in result.skipped:
    print(f"SKIPPED  {name}")
if args.dump:
    Path(args.dump).write_text(result.model_dump_json(indent=2), encoding="utf-8")
