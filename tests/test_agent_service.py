import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from services.agent_service.src.graph import (
    decompose,
    reason,
    route_after_validate,
    validate,
)
from services.agent_service.src.state import AgentState
from services.agent_service.src.tools import calculate, decompose_question, search_documents, search_tables
from shared.models import Citation, StrictAnswer


def test_decompose_produces_user_turn():
    """Verify decompose produces a user role message to avoid Gemini 400 model turn error."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="What was the revenue?"),
            ToolMessage(content='["What was Q1 revenue?", "What was Q2 revenue?"]', tool_call_id="call_1"),
        ]
    }
    result = decompose(state)
    assert "messages" in result
    msg = result["messages"][0]
    assert msg["role"] == "user"
    assert "What was Q1 revenue?" in msg["content"]


def test_reason_ensures_no_trailing_model_turn(monkeypatch):
    """Verify reason() adds a user turn if the last message in history was from model/assistant."""
    recorded_messages = []

    class MockRunnable:
        def invoke(self, messages, *args, **kwargs):
            recorded_messages.extend(messages)
            return AIMessage(content="test answer")

    monkeypatch.setattr("services.agent_service.src.graph.llm_with_tools", MockRunnable())
    monkeypatch.setattr("services.agent_service.src.graph.invoke_with_retry", lambda r, m, *a, **kw: r.invoke(m))

    state: AgentState = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Previous assistant turn"),
        ]
    }
    reason(state)
    assert len(recorded_messages) > 0
    # The last message before sending to model should be user, not assistant
    last_msg = recorded_messages[-1]
    last_role = getattr(last_msg, "type", None) or (last_msg.get("role") if isinstance(last_msg, dict) else None)
    assert last_role == "user"


def test_route_after_validate_caps_repair_attempts():
    """Verify repair cycles terminate after 2 attempts to prevent infinite loops."""
    # Attempt 0 invalid -> repair
    assert route_after_validate({"validation": {"is_valid": False}, "repair_attempts": 0}) == "repair"
    # Attempt 1 invalid -> repair
    assert route_after_validate({"validation": {"is_valid": False}, "repair_attempts": 1}) == "repair"
    # Attempt 2 invalid -> end
    assert route_after_validate({"validation": {"is_valid": False}, "repair_attempts": 2}) == "end"
    # Valid -> end
    assert route_after_validate({"validation": {"is_valid": True}, "repair_attempts": 0}) == "end"


def test_validate_sends_wrapped_answer(monkeypatch):
    """Verify validate() wraps answer in {'answer': answer_dict} for ValidationRequest."""
    captured_json = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"is_valid": True, "error": None}

    def mock_post(url, json, timeout):
        captured_json.update(json)
        return MockResponse()

    monkeypatch.setattr("requests.post", mock_post)

    answer = StrictAnswer(
        answer_type="direct",
        params={"value": "100 million"},
        evidence=[Citation(document_id="doc1.pdf", page=1)],
    )
    result = validate({"answer": answer})
    assert result["validation"]["is_valid"] is True
    assert "answer" in captured_json
    assert captured_json["answer"]["answer_type"] == "direct"
    assert captured_json["answer"]["params"]["value"] == "100 million"


def test_calculate_tool_error_resilience():
    """Verify calculate tool does not crash on invalid input or division by zero."""
    result = calculate.invoke({"expression": "10 / 0"})
    assert "error" in str(result).lower()

    syntax_err = calculate.invoke({"expression": "10 ++"})
    assert "error" in str(syntax_err).lower()

    valid_res = calculate.invoke({"expression": "10 + 5 * 2"})
    assert valid_res == 20.0


def test_search_tools_error_resilience(monkeypatch):
    """Verify search tools return empty result dict on network error instead of raising."""

    def mock_post_err(*args, **kwargs):
        raise ConnectionError("Network down")

    monkeypatch.setattr("requests.post", mock_post_err)

    docs_res = search_documents.invoke({"query": "revenue"})
    assert docs_res["results"] == []
    assert "error" in docs_res

    tables_res = search_tables.invoke({"query": "balance sheet"})
    assert tables_res["results"] == []
    assert "error" in tables_res
