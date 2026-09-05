from fastapi.testclient import TestClient
from services.answer_validator_api.src.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "answer-validator-api"

def test_validate_answer_success_logs(caplog):
    payload = {"answer": {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017.pdf", "page": 1}],
        "params": {"value": "$142.5M"}
    }}
    resp = client.post("/validate_answer", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_valid"] is True
    assert "[ANSWER-VALIDATOR-SUCCESS]" in body["log_message"]

def test_validate_answer_error_message_format():
    payload = {"answer": {
        "answer_type": "calculated",
        "evidence": [{"document_id": "doc_041.pdf", "page": 2}],
        "params": {"value": 13.4}  # missing formula
    }}
    resp = client.post("/validate_answer", json=payload)
    body = resp.json()
    assert body["is_valid"] is False
    assert "[ANSWER-VALIDATOR-ERROR]" in body["log_message"]

def test_calculate_basic():
    resp = client.post("/calculate", json={"expression": "(3875-3410)/3410*100"})
    assert resp.json()["status"] == "success"
    assert round(resp.json()["result"], 2) == 13.64

def test_calculate_division_by_zero():
    resp = client.post("/calculate", json={"expression": "5/0"})
    assert resp.json()["status"] == "error"

def test_calculate_malformed_expression():
    resp = client.post("/calculate", json={"expression": "import os"})
    assert resp.json()["status"] == "error"