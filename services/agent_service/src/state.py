from typing import TypedDict, Any
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from shared.models import StrictAnswer


class AgentState(TypedDict, total=False):
    query: str
    messages: Annotated[list[Any], add_messages]
    evidence: list[dict[str, Any]]
    answer: StrictAnswer
    validation: dict[str, Any]