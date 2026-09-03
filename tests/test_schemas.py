"""
tests/test_schemas.py: Automated tests for strict answer schemas and microservice contracts.
Ensures full conformance with Final_Project.md and LEDGER_HANDBOOK.md specifications.
"""

import pytest
from pydantic import ValidationError

from shared.config import get_service_url
from shared.models import (
    AnswerType,
    AskRequest,
    AskResponse,
    Citation,
    DocumentBlock,
    RetrievedChunk,
    SearchQueryRequest,
    SearchQueryResponse,
    StrictAnswer,
)

# ==============================================================================
# 1. Tests for 'direct' answers
# ==============================================================================


def test_valid_direct_answer_string():
    """Valid fact lookup with string value and citation."""
    data = {
        "answer_type": "direct",
        "evidence": [
            {"document_id": "doc_017.pdf", "page": 1, "section": "Income Statement"}
        ],
        "params": {"value": "$142.5M"},
    }
    answer = StrictAnswer(**data)
    assert answer.answer_type == AnswerType.DIRECT.value
    assert len(answer.evidence) == 1
    assert answer.evidence[0].document_id == "doc_017.pdf"
    assert answer.evidence[0].page == 1
    assert answer.params["value"] == "$142.5M"


def test_valid_direct_answer_numeric():
    """Valid fact lookup with numeric value."""
    data = {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017.pdf", "page": 2}],
        "params": {"value": 142.5},
    }
    answer = StrictAnswer(**data)
    assert answer.params["value"] == 142.5


def test_invalid_direct_answer_missing_evidence():
    """Direct answers must reject empty evidence lists."""
    data = {
        "answer_type": "direct",
        "evidence": [],
        "params": {"value": "$142.5M"},
    }
    with pytest.raises(ValidationError) as excinfo:
        StrictAnswer(**data)
    assert "Direct answer requires at least 1 evidence citation" in str(excinfo.value)


def test_invalid_direct_answer_missing_value():
    """Direct answers must contain a 'value' parameter."""
    data = {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017.pdf", "page": 1}],
        "params": {},
    }
    with pytest.raises(ValidationError):
        StrictAnswer(**data)


# ==============================================================================
# 2. Tests for 'calculated' answers
# ==============================================================================


def test_valid_calculated_answer():
    """Valid calculated answer with arithmetic formula, numeric result, and citations."""
    data = {
        "answer_type": "calculated",
        "evidence": [
            {"document_id": "doc_041.pdf", "page": 2, "section": "Operating Expenses"},
            {"document_id": "doc_041.pdf", "page": 1, "section": "Operating Expenses"},
        ],
        "params": {
            "value": 13.64,
            "formula": "(3875 - 3410) / 3410 * 100",
        },
    }
    answer = StrictAnswer(**data)
    assert answer.answer_type == AnswerType.CALCULATED.value
    assert answer.params["value"] == 13.64
    assert "(3875 - 3410)" in answer.params["formula"]
    assert len(answer.evidence) == 2


def test_invalid_calculated_answer_missing_formula():
    """Calculated answers must reject payloads without an explicit formula."""
    data = {
        "answer_type": "calculated",
        "evidence": [{"document_id": "doc_041.pdf", "page": 1}],
        "params": {"value": 13.64},
    }
    with pytest.raises(ValidationError):
        StrictAnswer(**data)


def test_invalid_calculated_answer_missing_evidence():
    """Calculated answers must reject empty evidence citations for operands."""
    data = {
        "answer_type": "calculated",
        "evidence": [],
        "params": {"value": 13.64, "formula": "3875 - 3410"},
    }
    with pytest.raises(ValidationError) as excinfo:
        StrictAnswer(**data)
    assert "Calculated answer requires evidence citations for the operands" in str(
        excinfo.value
    )


# ==============================================================================
# 3. Tests for 'multi_span' answers
# ==============================================================================


def test_valid_multi_span_answer():
    """Valid multi-span answer with a list of at least 2 values."""
    data = {
        "answer_type": "multi_span",
        "evidence": [
            {"document_id": "doc_022.pdf", "page": 3, "section": "Operating Expenses"}
        ],
        "params": {"values": ["Marketing", "R&D", "Logistics"]},
    }
    answer = StrictAnswer(**data)
    assert answer.answer_type == AnswerType.MULTI_SPAN.value
    assert len(answer.params["values"]) == 3


def test_invalid_multi_span_single_item():
    """Multi-span must reject single items (those belong in 'direct')."""
    data = {
        "answer_type": "multi_span",
        "evidence": [{"document_id": "doc_022.pdf", "page": 3}],
        "params": {"values": ["Marketing"]},
    }
    with pytest.raises(ValidationError):
        StrictAnswer(**data)


# ==============================================================================
# 4. Tests for 'insufficient_evidence' answers
# ==============================================================================


def test_valid_insufficient_evidence():
    """Valid abstention answer with reason and empty evidence."""
    data = {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {
            "reason": "No document in the indexed corpus reports restructuring expenses."
        },
    }
    answer = StrictAnswer(**data)
    assert answer.answer_type == AnswerType.INSUFFICIENT_EVIDENCE.value
    assert len(answer.evidence) == 0
    assert "restructuring expenses" in answer.params["reason"]


def test_invalid_insufficient_evidence_missing_reason():
    """Abstention must include an explanatory reason."""
    data = {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {},
    }
    with pytest.raises(ValidationError):
        StrictAnswer(**data)


# ==============================================================================
# 5. Tests for Service Contracts & Networking
# ==============================================================================


def test_document_block_schema():
    """Verify doc-processor-api DocumentBlock schema."""
    block = DocumentBlock(
        block_id="blk-001",
        document_id="cts-corporation_2019.pdf",
        page=1,
        content_type="table",
        markdown_content="| Finished Goods | 9,447 |",
        bbox=[10.0, 20.0, 200.0, 150.0],
        metadata={"scale": "thousand", "period": "2019"},
    )
    assert block.document_id == "cts-corporation_2019.pdf"
    assert block.content_type == "table"


def test_search_query_roundtrip():
    """Verify retrieval-api request and response schemas."""
    req = SearchQueryRequest(
        query="What was the amount of Finished Goods in 2019?", top_k=30, top_n=5
    )
    assert req.top_k == 30

    chunk = RetrievedChunk(
        chunk_id="chk-001",
        document_id="cts-corporation_2019.pdf",
        page=1,
        content="| Finished Goods | 9,447 |",
        score=0.92,
        dense_score=0.88,
        sparse_score=14.5,
    )
    resp = SearchQueryResponse(
        query=req.query, results=[chunk], total_found=1, execution_time_ms=12.5
    )
    assert resp.total_found == 1
    assert resp.results[0].document_id == "cts-corporation_2019.pdf"


def test_ask_request_and_response():
    """Verify orchestrator-api AskRequest and AskResponse schemas."""
    req = AskRequest(query="What was operating income in 2020?")
    assert req.document_id is None

    ans = StrictAnswer(
        answer_type="direct",
        evidence=[Citation(document_id="doc_01.pdf", page=0)],
        params={"value": "$100M"},
    )
    resp = AskResponse(query=req.query, answer=ans, latency_ms=145.2)
    assert resp.answer.params["value"] == "$100M"


def test_service_url_resolution():
    """Verify service URL resolving helper."""
    assert get_service_url("orchestrator-api") == "http://localhost:8001"
    assert get_service_url("retrieval-api") == "http://localhost:8003"
    assert get_service_url("agent-service") == "http://localhost:8004"
    assert get_service_url("answer-validator-api") == "http://localhost:8005"
    assert get_service_url("eval-service") == "http://localhost:8006"
    assert get_service_url("ui-service") == "http://localhost:8000"
