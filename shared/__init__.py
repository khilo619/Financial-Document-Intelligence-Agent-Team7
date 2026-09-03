"""
shared package: Central contracts, immutable schemas, configuration, and telemetry
for Project LEDGER (Financial Document Intelligence Agent - MIA Team 7).
"""

from shared.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RERANKER_MODEL,
    SERVICE_HOSTS,
    SERVICE_PORTS,
    get_service_url,
)
from shared.models import (
    AnswerType,
    AskRequest,
    AskResponse,
    CalculatedParams,
    Citation,
    DirectParams,
    DocumentBlock,
    InsufficientParams,
    MultiSpanParams,
    RetrievedChunk,
    SearchQueryRequest,
    SearchQueryResponse,
    StrictAnswer,
    ValidationRequest,
    ValidationResponse,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "SERVICE_HOSTS",
    "SERVICE_PORTS",
    "AnswerType",
    "AskRequest",
    "AskResponse",
    "CalculatedParams",
    "Citation",
    "DirectParams",
    "DocumentBlock",
    "InsufficientParams",
    "MultiSpanParams",
    "RetrievedChunk",
    "SearchQueryRequest",
    "SearchQueryResponse",
    "StrictAnswer",
    "ValidationRequest",
    "ValidationResponse",
    "get_service_url",
]
