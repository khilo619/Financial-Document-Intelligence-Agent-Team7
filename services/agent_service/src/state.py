from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from shared.models import StrictAnswer


class AgentState(TypedDict, total=False):
    query: str
    document_id: str | None
    messages: Annotated[list[Any], add_messages]
    evidence: list[dict[str, Any]]
    answer: StrictAnswer
    validation: dict[str, Any]
