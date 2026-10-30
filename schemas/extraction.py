"""Agent-to-agent messages, shared state and the per-bid output record (Part D)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

NOT_FOUND = "Not found in documents"


class Source(BaseModel):
    file: str
    page: int
    page_end: int | None = None
    chunk_id: str | None = None

    def label(self) -> str:
        pages = (
            f"p. {self.page}" if not self.page_end or self.page_end == self.page else f"pp. {self.page}-{self.page_end}"
        )
        return f"{self.file}, {pages}"


class Evidence(BaseModel):
    """A retrieved passage, as handed from the retrieval agent to the extraction agents."""

    chunk_id: str
    text: str
    source: Source
    doc_type: str
    addendum_number: int | None = None
    section: str | None = None
    score: float = 0.0


class FieldResult(BaseModel):
    name: str
    value: str | None = None
    sources: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    attempts: int = 1
    status: Literal["draft", "passed", "failed", "not_found"] = "draft"


# ---- LLM output schemas (structured outputs) ------------------------------------------------------


def _as_fraction(value: object) -> object:
    # Small models sometimes give a percentage (e.g. 100) despite the schema's 0-1 range.
    return value / 100 if isinstance(value, int | float) and 1 < value <= 100 else value


Confidence = Annotated[float, BeforeValidator(_as_fraction), Field(ge=0, le=1)]


class ExtractedField(BaseModel):
    # Field order matters: the model fills them in this order, so it picks evidence before the value.
    name: str
    evidence_ids: list[str] = Field(description='Ids of the passages that state the value, e.g. ["E2"]')
    value: str | None = Field(description="The value, or null if the evidence does not state it")
    confidence: Confidence
    notes: str = Field(description="At most 20 words: source of conflicts, or why not found")


class ExtractionOutput(BaseModel):
    fields: list[ExtractedField]


class ProposedChange(BaseModel):
    field: str = Field(description="One of the listed field names, or Other")
    change_type: Literal["modifies", "clarifies"] = Field(
        description="modifies: replaces a stated value (e.g. new due date); clarifies: adds detail only"
    )
    description: str
    old_value: str | None = None
    new_value: str | None = None
    evidence_ids: list[str] = Field(description='Ids of the passages stating the change, e.g. ["E1"]')


class ChangeLogOutput(BaseModel):
    changes: list[ProposedChange]


class AnswerOutput(BaseModel):
    evidence_ids: list[str] = Field(description='Ids of the passages the answer relies on, e.g. ["E1", "E4"]')
    answer: str = Field(description=f'The answer, or "{NOT_FOUND}" when the evidence does not contain it')
    confidence: Confidence


class SummaryOutput(BaseModel):
    summary: str = Field(description="3-6 sentences")


# ---- Reconciliation, validation, final record ------------------------------------------------------


class AddendumChange(BaseModel):
    addendum: str  # file name
    addendum_number: int | None = None
    field: str
    change_type: Literal["modifies", "clarifies"]
    description: str
    old_value: str | None = None
    new_value: str | None = None
    sources: list[Source] = Field(default_factory=list)
    applied: bool = False


class ValidationIssue(BaseModel):
    field: str
    check: str  # citation | grounding | format | consistency | not_found
    message: str


class ValidationSummary(BaseModel):
    passed: int = 0
    failed: int = 0
    not_found: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)


class FieldOut(BaseModel):
    value: str | None
    sources: list[dict]
    confidence: float
    notes: str
    status: Literal["passed", "failed", "not_found"] = "passed"  # the validator's verdict


class BidRecord(BaseModel):
    """The JSON file written per bid."""

    bid_id: str
    generated_at: str
    model: str
    fields: dict[str, FieldOut]
    addendum_changes: list[AddendumChange]
    validation: ValidationSummary
    run: dict = Field(default_factory=dict)  # run id, timings, token counts
