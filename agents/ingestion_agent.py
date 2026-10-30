"""Ingestion agent: parses a bid folder and (re)indexes it, reporting what it found."""

from __future__ import annotations

from pathlib import Path

from agents.tracing import Tracer
from schemas.documents import Document
from search.store import IndexReport, IndexStore


class IngestionAgent:
    def __init__(self, store: IndexStore, tracer: Tracer):
        self.store = store
        self.tracer = tracer

    def run(self, bid_id: str, folder: str | Path | None = None) -> tuple[list[Document], IndexReport | None]:
        """Index the folder when given (incremental), then return the bid's documents from the index."""
        with self.tracer.step(
            "ingestion", "index", {"bid_id": bid_id, "folder": str(folder) if folder else None}
        ) as step:
            report = None
            if folder is not None:
                report = self.store.index_folder(folder, bid_id)
                step.tool(
                    "index_folder", {"folder": str(folder)}, report.model_dump(exclude={"seconds"}), report.seconds
                )
            documents = self.store.documents(bid_id)
            step.output(
                [
                    {
                        "file": d.file_name,
                        "type": d.doc_type.value,
                        "addendum": d.addendum_number,
                        "pages": len(d.pages),
                    }
                    for d in documents
                ]
            )
        return documents, report
