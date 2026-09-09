"""
HTTP-level tests for answer-validator-api.

Unlike test_schemas.py (which validates the StrictAnswer Pydantic model in
isolation), these tests exercise the actual running FastAPI endpoints via
TestClient — covering request/response wrapping, exception handling and
ordering, console log output format (per the grading spec's exact log
strings), and the /calculate endpoint's safe arithmetic sandboxing.
"""

import logging

import pytest


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "answer-validator-api"


VALID_CASES = [
    (
        "direct",
        {
            "answer_type": "direct",
            "evidence": [{"document_id": "doc_017.pdf", "page": 1}],
            "params": {"value": "$142.5M"},
        },
    ),
    (
        "calculated",
        {
            "answer_type": "calculated",
            "evidence": [
                {"document_id": "doc_041.pdf", "page": 2},
                {"document_id": "doc_041.pdf", "page": 1},
            ],
            "params": {"value": 13.64, "formula": "(3875-3410)/3410*100"},
        },
    ),
    (
        "multi_span",
        {
            "answer_type": "multi_span",
            "evidence": [{"document_id": "doc_022.pdf", "page": 3}],
            "params": {"values": ["Marketing", "R&D"]},
        },
    ),
    (
        "insufficient_evidence",
        {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": "No document reports restructuring expenses."},
        },
    ),
]


@pytest.mark.parametrize("expected_type,payload", VALID_CASES)
def test_validate_answer_valid_all_types(client, caplog, expected_type, payload):
    with caplog.at_level(logging.INFO, logger="AnswerValidator"):
        resp = client.post("/validate_answer", json={"answer": payload})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_valid"] is True
    assert body["answer_type"] == expected_type
    assert "[ANSWER-VALIDATOR-SUCCESS]" in caplog.text


INVALID_CASES = [
    (
        "calculated",
        {
            "answer_type": "calculated",
            "evidence": [{"document_id": "doc_041.pdf", "page": 2}],
            "params": {"value": 13.4},
        },
        "formula",
    ),
    (
        "direct",
        {"answer_type": "direct", "evidence": [], "params": {"value": "$100M"}},
        "evidence",
    ),
    (
        "multi_span",
        {
            "answer_type": "multi_span",
            "evidence": [{"document_id": "doc_1.pdf", "page": 1}],
            "params": {"values": ["only one"]},
        },
        "at least 2 items",
    ),
    (
        "insufficient_evidence",
        {"answer_type": "insufficient_evidence", "evidence": [], "params": {}},
        "reason",
    ),
]


@pytest.mark.parametrize("a_type,payload,expected_snippet", INVALID_CASES)
def test_validate_answer_invalid_all_types(
    client, caplog, a_type, payload, expected_snippet
):
    with caplog.at_level(logging.ERROR, logger="AnswerValidator"):
        resp = client.post("/validate_answer", json={"answer": payload})
    body = resp.json()
    assert body["is_valid"] is False
    assert "[ANSWER-VALIDATOR-ERROR]" in caplog.text
    # If this line fails, it tells you exactly which type's error message
    # isn't being cleaned up correctly yet — that's the point of the test.
    assert expected_snippet.lower() in body["log_message"].lower()


def test_validate_answer_malformed_envelope(client):
    resp = client.post("/validate_answer", json={"wrong_key": {}})
    assert resp.status_code == 422


def test_validate_answer_reports_clean_missing_key(client):
    resp = client.post(
        "/validate_answer",
        json={
            "answer": {
                "answer_type": "calculated",
                "evidence": [{"document_id": "doc_041.pdf", "page": 2}],
                "params": {"value": 13.4},
            }
        },
    )
    assert resp.json()["error"] == "Missing required key 'formula'"


def test_validate_answer_rejects_formula_value_mismatch(client):
    resp = client.post(
        "/validate_answer",
        json={
            "answer": {
                "answer_type": "calculated",
                "evidence": [{"document_id": "doc_041.pdf", "page": 2}],
                "params": {"value": 13.64, "formula": "3875 - 3410"},
            }
        },
    )
    body = resp.json()
    assert body["is_valid"] is False
    assert "does not match formula result" in body["error"].lower()


def test_validate_answer_rejects_invalid_formula(client):
    resp = client.post(
        "/validate_answer",
        json={
            "answer": {
                "answer_type": "calculated",
                "evidence": [{"document_id": "doc_041.pdf", "page": 2}],
                "params": {"value": 13.64, "formula": "__import__('os')"},
            }
        },
    )
    body = resp.json()
    assert body["is_valid"] is False
    assert "could not be evaluated" in body["error"].lower()


def test_calculate_basic(client):
    resp = client.post("/calculate", json={"expression": "(3875-3410)/3410*100"})
    assert resp.json()["status"] == "success"
    assert round(resp.json()["result"], 2) == 13.64


def test_calculate_division_by_zero(client):
    resp = client.post("/calculate", json={"expression": "5/0"})
    assert resp.json()["status"] == "error"


def test_calculate_rejects_non_finite_result(client):
    resp = client.post("/calculate", json={"expression": "1e309"})
    assert resp.json()["status"] == "error"


@pytest.mark.parametrize(
    "bad_expr",
    [
        "import os",
        "__import__('os').system('ls')",
        "().__class__.__bases__",
    ],
)
def test_calculate_rejects_unsafe(client, bad_expr):
    resp = client.post("/calculate", json={"expression": bad_expr})
    assert resp.json()["status"] == "error"
