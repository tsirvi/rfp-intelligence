"""REST API.

uvicorn api.app:app --port 8000

GET  /health                 index status
GET  /bids                   indexed bids and their documents
POST /index                  index a bid folder (runs in the background; poll GET /index/{job_id})
GET  /search?q=...&bid_id=   hybrid search with citations
POST /ask                    question answering with citations (multi-agent)
POST /extract                extract the 20-field record for an indexed bid (background; poll GET /extract/{job_id})
GET  /records/{bid_id}       the latest extracted record (JSON); /records/{bid_id}/report for Markdown
POST /compare                side-by-side comparison report of extracted bids
GET  /go-no-go/{bid_id}      bid / no-bid recommendation against the company profile
GET  /usage                  cost and latency per kind of run, Q&A cache hit rate
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from search.hybrid import HybridSearcher
from search.store import IndexReport, IndexStore
from settings import ROOT, section

log = logging.getLogger(__name__)


class IndexRequest(BaseModel):
    path: str = Field(description="Bid folder on the server's filesystem")
    bid_id: str | None = Field(default=None, description="Defaults to the folder name")
    force: bool = Field(default=False, description="Re-parse files even if unchanged")


class IndexJob(BaseModel):
    job_id: str
    bid_id: str
    path: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    report: IndexReport | None = None
    error: str | None = None


class SearchHitOut(BaseModel):
    rank: int
    score: float
    citation: str
    bid_id: str
    file_name: str
    doc_type: str
    addendum_number: int | None
    page_start: int
    page_end: int
    section: str | None
    kind: str
    text: str
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None


class SearchResponse(BaseModel):
    query: str
    variants: list[str]
    filters: dict
    timings: dict[str, float]
    hits: list[SearchHitOut]


class DocumentOut(BaseModel):
    file_name: str
    doc_type: str
    addendum_number: int | None
    title: str | None
    document_date: str | None
    document_date_source: str | None
    pages: int
    tables: int
    ocr_pages: int
    warnings: list[str]


class BidOut(BaseModel):
    bid_id: str
    documents: list[DocumentOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    bids: list[str] | None = Field(default=None, description="Restrict to these bids; default: inferred")


class CompareRequest(BaseModel):
    bids: list[str] = Field(min_length=2, description="Extracted bids to compare")
    summary: bool = Field(default=True, description="Add a model-written summary (slower on CPU)")


class ExtractRequest(BaseModel):
    bid_id: str


class ExtractJob(BaseModel):
    job_id: str
    bid_id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    validation: dict | None = None
    error: str | None = None


class State:
    """Process-wide singletons. Models load lazily on first search, so the API starts fast."""

    def __init__(self) -> None:
        self.store = IndexStore()
        self.searcher = HybridSearcher(self.store)
        self.jobs: dict[str, IndexJob] = {}
        self.extract_jobs: dict[str, ExtractJob] = {}
        self.index_lock = threading.Lock()  # one indexing job at a time; searches keep working
        self.extract_lock = threading.Lock()  # one extraction at a time: the local LLM is the bottleneck
        self._orchestrator = None

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from agents.orchestrator import Orchestrator

            self._orchestrator = Orchestrator(store=self.store, searcher=self.searcher)
        return self._orchestrator


state: State | None = None


def get_state() -> State:
    global state
    if state is None:
        state = State()
    return state


def run_index_job(job: IndexJob, force: bool) -> None:
    s = get_state()
    with s.index_lock:
        job.status, job.started_at = "running", datetime.now(UTC)
        try:
            job.report = s.store.index_folder(job.path, job.bid_id, force=force)
            job.status = "done"
        except Exception as exc:
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("index job %s failed", job.job_id)
        job.finished_at = datetime.now(UTC)


def submit_index_job(path: Path, bid_id: str | None, force: bool) -> IndexJob:
    job = IndexJob(job_id=uuid.uuid4().hex[:12], bid_id=bid_id or path.name, path=str(path))
    get_state().jobs[job.job_id] = job
    return job


def startup_folders() -> list[Path]:
    """Bid folders under DATA_DIR are indexed when the API starts (incremental, so cheap after the first run)."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if section("api").get("index_on_startup", True):
        jobs = [submit_index_job(folder, None, False) for folder in startup_folders()]

        def index_all() -> None:
            for job in jobs:
                run_index_job(job, force=False)
            try:  # load the models now, so the first question is not slowed by loading them
                get_state().orchestrator.preload()
            except Exception:
                log.exception("model preload failed")

        threading.Thread(target=index_all, name="startup-index", daemon=True).start()
    yield


app = FastAPI(title="RFP Intelligence API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    s = get_state()
    return {
        "status": "ok",
        "bids": s.store.bids(),
        "chunks": s.store.count(),
        "indexing": [j.job_id for j in s.jobs.values() if j.status in ("queued", "running")],
    }


@app.get("/bids", response_model=list[BidOut])
def bids() -> list[BidOut]:
    s = get_state()
    out = []
    for bid_id in s.store.bids():
        docs = [
            DocumentOut(
                file_name=d.file_name,
                doc_type=d.doc_type.value,
                addendum_number=d.addendum_number,
                title=d.title,
                document_date=d.document_date.isoformat() if d.document_date else None,
                document_date_source=d.document_date_source,
                pages=len(d.pages),
                tables=sum(len(p.tables) for p in d.pages),
                ocr_pages=sum(p.ocr for p in d.pages),
                warnings=d.warnings,
            )
            for d in s.store.documents(bid_id)
        ]
        out.append(BidOut(bid_id=bid_id, documents=docs))
    return out


def index_roots() -> list[Path]:
    roots = section("api").get("index_roots") or ["data"]
    return [(r if (r := Path(root)).is_absolute() else ROOT / r).resolve() for root in roots]


@app.post("/index", response_model=IndexJob, status_code=202)
def index(req: IndexRequest, background: BackgroundTasks) -> IndexJob:
    path = Path(req.path).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    # Only folders under the configured roots: the API must not read arbitrary paths on the server.
    if not any(path.is_relative_to(root) for root in index_roots()):
        allowed = ", ".join(str(r.relative_to(ROOT)) if r.is_relative_to(ROOT) else str(r) for r in index_roots())
        raise HTTPException(status_code=403, detail=f"Bid folders must be inside: {allowed}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a folder: {req.path}")
    job = submit_index_job(path, req.bid_id, req.force)
    background.add_task(run_index_job, job, req.force)
    return job


@app.get("/index/{job_id}", response_model=IndexJob)
def index_status(job_id: str) -> IndexJob:
    job = get_state().jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


@app.get("/search", response_model=SearchResponse)
def search(
    q: Annotated[str, Query(min_length=2, description="Search query")],
    bid_id: Annotated[list[str] | None, Query()] = None,
    doc_type: Annotated[list[str] | None, Query()] = None,
    addendum_number: Annotated[list[int] | None, Query()] = None,
    file_name: Annotated[list[str] | None, Query()] = None,
    top_k: Annotated[int | None, Query(ge=1, le=50)] = None,
    k: Annotated[int | None, Query(ge=1, le=50, description="Alias of top_k")] = None,
    rerank: bool | None = None,
    expand: bool | None = None,
) -> SearchResponse:
    """Filters are combined with AND; repeating a parameter means any of its values."""
    filters = {"bid_id": bid_id, "doc_type": doc_type, "addendum_number": addendum_number, "file_name": file_name}
    searcher = get_state().searcher
    overrides = {k_: v for k_, v in {"rerank": rerank, "expand": expand}.items() if v is not None}  # else config
    result = searcher.search(q, filters, top_k=top_k or k or searcher.config.top_k, **overrides)
    hits = [
        SearchHitOut(
            rank=h.rank,
            score=h.score,
            citation=h.citation,
            bid_id=h.chunk.bid_id,
            file_name=h.chunk.file_name,
            doc_type=h.chunk.doc_type.value,
            addendum_number=h.chunk.addendum_number,
            page_start=h.chunk.page_start,
            page_end=h.chunk.page_end,
            section=h.chunk.section,
            kind=h.chunk.kind,
            text=h.chunk.text,
            dense_rank=h.dense_rank,
            bm25_rank=h.bm25_rank,
            rerank_score=h.rerank_score,
        )
        for h in result.hits
    ]
    filters_out = {key: v for key, v in filters.items() if v}
    return SearchResponse(query=q, variants=result.variants, filters=filters_out, timings=result.timings, hits=hits)


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Runs the Q&A graph (plan -> retrieve per bid -> cited answer). Takes about a minute on a CPU-only LLM."""
    s = get_state()
    if not s.store.bids():
        raise HTTPException(status_code=409, detail="No bids indexed yet")
    unknown = sorted(set(req.bids or []) - set(s.store.bids()))
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown bid(s): {', '.join(unknown)}")
    answer = s.orchestrator.ask(req.question, req.bids)
    return answer.model_dump(mode="json") | {"citation_labels": [c.label() for c in answer.citations]}


def run_extract_job(job: ExtractJob) -> None:
    s = get_state()
    with s.extract_lock:
        job.status, job.started_at = "running", datetime.now(UTC)
        try:
            record = s.orchestrator.extract_bid(job.bid_id)
            job.validation = record.validation.model_dump(exclude={"issues"})
            job.status = "done"
        except Exception as exc:
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("extract job %s failed", job.job_id)
        job.finished_at = datetime.now(UTC)


@app.post("/extract", response_model=ExtractJob, status_code=202)
def extract(req: ExtractRequest, background: BackgroundTasks) -> ExtractJob:
    s = get_state()
    if req.bid_id not in s.store.bids():
        raise HTTPException(status_code=404, detail=f"Bid not indexed: {req.bid_id}")
    # A repeated click must not queue another 15-30 minute run of the same bid.
    for job in s.extract_jobs.values():
        if job.bid_id == req.bid_id and job.status in ("queued", "running"):
            return job
    job = ExtractJob(job_id=uuid.uuid4().hex[:12], bid_id=req.bid_id)
    s.extract_jobs[job.job_id] = job
    background.add_task(run_extract_job, job)
    return job


@app.get("/extract/{job_id}", response_model=ExtractJob)
def extract_status(job_id: str) -> ExtractJob:
    job = get_state().extract_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


@app.get("/records/{bid_id}/report", response_class=PlainTextResponse)
def record_report(bid_id: str) -> str:
    """The human-readable Markdown report written next to the JSON record."""
    path = get_state().orchestrator.outputs_dir / f"{bid_id}.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No report for {bid_id} yet")
    return path.read_text("utf-8")


@app.get("/records/{bid_id}")
def record(bid_id: str) -> dict:
    path = get_state().orchestrator.outputs_dir / f"{bid_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No extracted record for {bid_id} yet")
    return json.loads(path.read_text("utf-8"))


def _extracted(bids: list[str]) -> None:
    s = get_state()
    missing = [b for b in bids if not (s.orchestrator.outputs_dir / f"{b}.json").is_file()]
    if missing:
        raise HTTPException(status_code=404, detail=f"No extracted record for {', '.join(missing)} yet")


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    """Side-by-side report; the model summary takes about a minute on CPU (summary=false skips it)."""
    _extracted(req.bids)
    result = get_state().orchestrator.compare(req.bids, summary=req.summary)
    from agents.comparison_agent import to_markdown

    return result.model_dump(mode="json") | {"markdown": to_markdown(result)}


@app.get("/go-no-go/{bid_id}")
def go_no_go(bid_id: str) -> dict:
    _extracted([bid_id])
    from agents.go_no_go_agent import to_markdown

    decision = get_state().orchestrator.go_no_go(bid_id)
    return decision.model_dump(mode="json") | {"markdown": to_markdown(decision)}


@app.get("/usage")
def usage() -> dict:
    from agents.usage import summary

    return summary(get_state().orchestrator.outputs_dir)
