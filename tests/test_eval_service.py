"""
tests/test_eval_service.py: Functional tests for eval_service FastAPI endpoints and benchmark harness.
"""

from fastapi.testclient import TestClient

from services.eval_service.src.main import app

client = TestClient(app)


def test_eval_service_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8006
    assert data["service"] == "eval-service"


def test_eval_service_run_benchmark_mock():
    payload = {"sample_size": 5, "use_mock": True}
    response = client.post("/run_benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 5
    assert data["status"] == "completed"
    assert "exact_match" in data
    assert "numerical_accuracy" in data
    assert "f1_score" in data
    assert "retrieval_recall" in data
    assert 0.0 <= data["pass_rate"] <= 1.0


def test_eval_service_metrics_and_failure_analysis():
    # First trigger benchmark to populate latest run
    client.post("/run_benchmark", json={"sample_size": 10, "use_mock": True})

    # Check /metrics
    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200
    metrics_data = resp_metrics.json()
    assert metrics_data["benchmark_file"] == "questions_setA_practice.json"
    assert "latest_run" in metrics_data
    assert metrics_data["latest_run"]["total_evaluated"] == 10

    # Check /failure_analysis
    resp_failures = client.get("/failure_analysis?top_n=5")
    assert resp_failures.status_code == 200
    failures_data = resp_failures.json()
    assert "presented_failures_count" in failures_data
    if failures_data["presented_failures_count"] > 0:
        ex = failures_data["examples"][0]
        assert "question_id" in ex
        assert "root_cause_stage" in ex
        assert "diagnosis" in ex
