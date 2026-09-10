"""
shared.models: Canonical Pydantic schemas and immutable contracts for Project LEDGER.
These schemas serve as the single source of truth across all 7 microservices.
"""

import math
from enum import Enum
from typing import Any, Literal

import simpleeval
from pydantic import BaseModel, Field, model_validator

# ==============================================================================
# 1. Strict Answer Schema & Evidence Models
# ==============================================================================


class AnswerType(str, Enum):
    DIRECT = "direct"
    CALCULATED = "calculated"
    MULTI_SPAN = "multi_span"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Citation(BaseModel):
    """
    Mandatory citation grounding the answer back to the source PDF report.
    """

    document_id: str = Field(
        ...,
        description="Unique filename or UID of source document (e.g., 'doc_017.pdf' or 'cts-corporation_2019.pdf')",
    )
    page: int = Field(..., ge=0, description="0-indexed or 1-indexed page number in the source PDF")
    section: str = Field(
        default="",
        description="Optional heading or section name (e.g., 'Income Statement', 'Note 4')",
    )
    bbox: list[float] | None = Field(
        default=None,
        description="Optional bounding box [x0, y0, x1, y1] for UI visual grounding",
    )


class DirectParams(BaseModel):
    """Parameters for 'direct' fact lookup answers."""

    value: str | float | int = Field(..., description="The direct factual value extracted from the document")


class CalculatedParams(BaseModel):
    """Parameters for 'calculated' arithmetic answers evaluated deterministically."""

    value: float | int = Field(..., description="Final computed numerical result")
    formula: str = Field(
        ...,
        min_length=1,
        description="Arithmetic expression executed (e.g., '(3875-3410)/3410*100')",
    )


class MultiSpanParams(BaseModel):
    """Parameters for 'multi_span' answers containing two or more distinct entities or metrics."""

    values: list[str | float | int] = Field(
        ...,
        min_length=2,
        description="List of at least two items that together answer the question",
    )


class InsufficientParams(BaseModel):
    """Parameters for 'insufficient_evidence' abstention answers."""

    reason: str = Field(
        ...,
        min_length=1,
        description="Explanation of why grounding was insufficient to prevent hallucination",
    )


class StrictAnswer(BaseModel):
    """
    The canonical output contract produced by the LangGraph agent and
    validated by answer-validator-api before reaching the user.
    """

    answer_type: Literal["direct", "calculated", "multi_span", "insufficient_evidence"]
    evidence: list[Citation] = Field(default_factory=list, description="Grounding evidence citations")
    params: dict[str, Any] = Field(..., description="Type-specific answer parameters")

    @model_validator(mode="after")
    def validate_answer_rules(self) -> "StrictAnswer":
        a_type = self.answer_type
        params = self.params or {}
        evidence = self.evidence or []

        if a_type == AnswerType.DIRECT.value:
            DirectParams(**params)
            if len(evidence) < 1:
                raise ValueError("Direct answer requires at least 1 evidence citation.")

        elif a_type == AnswerType.CALCULATED.value:
            calc = CalculatedParams(**params)
            # Ensure value is actually a numeric float or int
            if not isinstance(calc.value, (int, float)):
                raise ValueError(f"Calculated answer value must be a number, got {type(calc.value).__name__}.")
            # Reject non-finite values outright
            if not math.isfinite(calc.value):
                raise ValueError(f"Calculated answer value must be finite, got {calc.value}.")

            if len(evidence) < 1:
                raise ValueError("Calculated answer requires evidence citations for the operands.")
            # Recompute the formula independently and compare to the reported value
            try:
                SAFE_FUNCTIONS = {
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "pow": pow,
                }
                recomputed = simpleeval.simple_eval(calc.formula, functions=SAFE_FUNCTIONS)
            except (
                simpleeval.InvalidExpression,
                SyntaxError,
                ZeroDivisionError,
                TypeError,
            ) as exc:
                raise ValueError(f"Formula '{calc.formula}' could not be evaluated: {exc}")

            if not math.isfinite(recomputed):
                raise ValueError(f"Formula '{calc.formula}' evaluates to a non-finite value.")

            if not math.isclose(recomputed, calc.value, rel_tol=1e-3, abs_tol=1e-6):
                raise ValueError(
                    f"Reported value {calc.value} does not match formula result {recomputed} for '{calc.formula}'."
                )

        elif a_type == AnswerType.MULTI_SPAN.value:
            ms = MultiSpanParams(**params)
            if len(ms.values) < 2:
                raise ValueError("Multi-span answer must contain at least 2 distinct values.")
            if len(evidence) < 1:
                raise ValueError("Multi-span answer requires at least 1 evidence citation.")

        elif a_type == AnswerType.INSUFFICIENT_EVIDENCE.value:
            InsufficientParams(**params)
            # evidence can be empty list for unanswerable questions

        else:
            raise ValueError(f"Unknown answer_type '{a_type}'. Must be one of {list(AnswerType)}")

        return self


class DecompositionResult(BaseModel):
    sub_questions: list[str]


# ==============================================================================
# 2. Document Processing & Ingestion Contracts (doc-processor-api)
# ==============================================================================


class DocumentBlock(BaseModel):
    """
    Structured block emitted by doc-processor-api layout OCR engine.
    """

    block_id: str = Field(..., description="Unique UUID for this text or table block")
    document_id: str = Field(..., description="Source PDF filename or document identifier")
    page: int = Field(..., ge=0, description="Source page number")
    content_type: Literal["text", "table", "header", "footnote"] = Field(..., description="Semantic type of block")
    markdown_content: str = Field(..., description="Text content formatted in Markdown (tables as Markdown grids)")
    table_rows: list[list[str]] | None = Field(default=None, description="Raw 2D cell grid if content_type is table")
    bbox: list[float] | None = Field(default=None, description="Coordinates [x0, y0, x1, y1] on the PDF page")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (scale: thousand, year: 2019, etc.)",
    )


class ProcessPdfRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute path or filename of raw PDF to process")
    document_id: str | None = Field(default=None, description="Optional custom document id")


class ProcessPdfResponse(BaseModel):
    document_id: str
    total_pages: int
    total_blocks: int
    blocks: list[DocumentBlock]
    processing_time_s: float = 0.0


# ==============================================================================
# 3. Retrieval & Hybrid Search Contracts (retrieval-api)
# ==============================================================================


class RetrievedChunk(BaseModel):
    """
    A single ranked chunk returned from hybrid search and reranking.
    """

    chunk_id: str
    document_id: str
    page: int
    section: str = ""
    content_type: str = "text"
    content: str
    score: float = Field(..., description="Final fused / reranked relevance score")
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Over-retrieval pool size for hybrid fusion",
    )
    top_n: int = Field(default=5, ge=1, le=20, description="Final number of reranked results to return")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata filters (e.g., {'document_id': 'doc_017.pdf'})",
    )
    use_reranking: bool = Field(default=True, description="Whether to apply cross-encoder reranker")


class SearchQueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    total_found: int
    execution_time_ms: float = 0.0


# ==============================================================================
# 4. Orchestrator, UI & Agent Communication Contracts
# ==============================================================================


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User or benchmark question")
    document_id: str | None = Field(default=None, description="Optional scope filter (None = corpus-wide)")
    session_id: str | None = Field(default=None, description="Conversation session ID for tracing")


class AskResponse(BaseModel):
    query: str
    answer: StrictAnswer
    latency_ms: float = 0.0
    trace_id: str | None = None


# ==============================================================================
# 5. Answer Validator Contracts (answer-validator-api)
# ==============================================================================


class ValidationRequest(BaseModel):
    answer: dict[str, Any] = Field(..., description="Raw dictionary representation of candidate answer")


class ValidationResponse(BaseModel):
    is_valid: bool
    answer_type: str | None = None
    error: str | None = None
    log_message: str
