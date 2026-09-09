from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_search_endpoint_with_reranking():
    response = client.post(
        "/search",
        json={
            "query": "Finished goods balance",
            "top_k": 30,
            "top_n": 5,
            "filters": {
                "company": "CTS Corporation",
                "year": 2019,
                "content_type": "table",
            },
            "use_reranking": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Finished goods balance"
    assert "results" in data
    assert "total_found" in data
    assert "execution_time_ms" in data

    assert len(data["results"]) > 0
    assert len(data["results"]) <= 5

    result = data["results"][0]

    assert result["chunk_id"] == "test-cts-2019-001"
    assert result["document_id"] == "cts-corporation_2019.pdf"
    assert result["content_type"] == "table"

    assert result["rerank_score"] is not None
    assert result["dense_score"] is not None
    assert result["sparse_score"] is not None


def test_search_endpoint_without_reranking():
    response = client.post(
        "/search",
        json={
            "query": "Finished goods balance",
            "top_k": 30,
            "top_n": 5,
            "filters": {
                "company": "CTS Corporation",
                "year": 2019,
                "content_type": "table",
            },
            "use_reranking": False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Finished goods balance"
    assert len(data["results"]) > 0
    assert len(data["results"]) <= 5

    result = data["results"][0]

    assert result["chunk_id"] == "test-cts-2019-001"
    assert result["rerank_score"] is None
    assert result["dense_score"] is not None
    assert result["sparse_score"] is not None
    assert result["score"] is not None