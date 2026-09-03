"""
shared.config: Global system constants, service registry, and networking resolvers.
Follows the Twelve-Factor App specification.
"""

import os
from enum import Enum


class ServiceName(str, Enum):
    UI = "ui-service"
    ORCHESTRATOR = "orchestrator-api"
    DOC_PROCESSOR = "doc-processor-api"
    RETRIEVAL = "retrieval-api"
    AGENT = "agent-service"
    ANSWER_VALIDATOR = "answer-validator-api"
    EVAL = "eval-service"
    QDRANT = "qdrant"
    LANGFUSE = "langfuse"


# Standard Port Registry
SERVICE_PORTS: dict[str, int] = {
    ServiceName.UI.value: int(os.getenv("UI_PORT", "8000")),
    ServiceName.ORCHESTRATOR.value: int(os.getenv("ORCHESTRATOR_PORT", "8001")),
    ServiceName.DOC_PROCESSOR.value: int(os.getenv("DOC_PROCESSOR_PORT", "8002")),
    ServiceName.RETRIEVAL.value: int(os.getenv("RETRIEVAL_PORT", "8003")),
    ServiceName.AGENT.value: int(os.getenv("AGENT_PORT", "8004")),
    ServiceName.ANSWER_VALIDATOR.value: int(os.getenv("ANSWER_VALIDATOR_PORT", "8005")),
    ServiceName.EVAL.value: int(os.getenv("EVAL_PORT", "8006")),
    ServiceName.QDRANT.value: int(os.getenv("QDRANT_PORT", "6333")),
    ServiceName.LANGFUSE.value: int(os.getenv("LANGFUSE_PORT", "3000")),
}

# Standard Hostnames (localhost for standalone runs, service DNS in Docker network)
SERVICE_HOSTS: dict[str, str] = {
    ServiceName.UI.value: os.getenv("UI_HOST", "localhost"),
    ServiceName.ORCHESTRATOR.value: os.getenv("ORCHESTRATOR_HOST", "localhost"),
    ServiceName.DOC_PROCESSOR.value: os.getenv("DOC_PROCESSOR_HOST", "localhost"),
    ServiceName.RETRIEVAL.value: os.getenv("RETRIEVAL_HOST", "localhost"),
    ServiceName.AGENT.value: os.getenv("AGENT_HOST", "localhost"),
    ServiceName.ANSWER_VALIDATOR.value: os.getenv("ANSWER_VALIDATOR_HOST", "localhost"),
    ServiceName.EVAL.value: os.getenv("EVAL_HOST", "localhost"),
    ServiceName.QDRANT.value: os.getenv("QDRANT_HOST", "localhost"),
    ServiceName.LANGFUSE.value: os.getenv("LANGFUSE_HOST", "localhost"),
}


def get_service_url(service_name: str, host: str | None = None) -> str:
    """
    Returns the complete HTTP base URL for a given microservice.
    Example: get_service_url('retrieval-api') -> 'http://localhost:8003'
    """
    resolved_host = host or SERVICE_HOSTS.get(service_name, "localhost")
    port = SERVICE_PORTS.get(service_name, 8000)
    return f"http://{resolved_host}:{port}"


# Model Configurations & Defaults
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Retrieval Hyperparameters
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
DEFAULT_TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "30"))
DEFAULT_TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "5"))
DEFAULT_RRF_K_CONSTANT = int(os.getenv("RRF_K_CONSTANT", "60"))

# Logging Prefixes (Mandatory for grading & verification)
VALIDATOR_SUCCESS_PREFIX = "[ANSWER-VALIDATOR-SUCCESS]"
VALIDATOR_ERROR_PREFIX = "[ANSWER-VALIDATOR-ERROR]"
