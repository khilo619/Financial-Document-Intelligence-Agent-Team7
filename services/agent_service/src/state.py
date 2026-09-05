from typing import TypedDict, Any
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class AgentState(TypedDict, total=False):
    query: str
    messages: Annotated[list, add_messages]
    evidence: list[dict[str, Any]]
    answer: str