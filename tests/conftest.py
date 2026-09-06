"""
Shared pytest fixtures for the answer-validator-api test suite.

Provides the `client` fixture (a FastAPI TestClient wrapping the live app) so
test files can send real HTTP requests to /validate_answer and /calculate
without each file re-instantiating its own client. Auto-discovered by pytest;
no imports needed elsewhere.
"""



import pytest
from fastapi.testclient import TestClient
from services.answer_validator_api.src.main import app


@pytest.fixture
def client():
    return TestClient(app)