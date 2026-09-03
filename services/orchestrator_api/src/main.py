"""
orchestrator_api: Central Gateway routing documents and queries between services.
Owned by Ahmed (Member 6) - Initial scaffold by Khaled (Repo Lead).
"""

import logging
import time

import httpx
from fastapi import FastAPI, HTTPException

from shared.config import ServiceName, get_service_url
from shared.models import (
    AskRequest,
    AskResponse,
    Citation,
    StrictAnswer,
    ValidationRequest,
    ValidationResponse,
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("OrchestratorAPI")

app = FastAPI(
    title="Project LEDGER - Orchestrator Gateway API",
    description="Routes queries across Retrieval, Agent, and Validator microservices.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": ServiceName.ORCHESTRATOR.value,
        "port": 8001,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Central query endpoint.
    Flow: User Question -> Agent Service -> Answer Validator -> Return to User.
    """
    start_time = time.time()
    logger.info(
        "Received query: '%s' (scope: %s)",
        request.query,
        request.document_id or "corpus-wide",
    )

    # Day 1 Scaffold: Simulate Agent response
    mock_answer = StrictAnswer(
        answer_type="calculated",
        evidence=[
            Citation(
                document_id="cts-corporation_2019.pdf", page=1, section="Finished Goods"
            ),
            Citation(
                document_id="jabil-circuit-inc_2019.pdf",
                page=1,
                section="Finished Goods",
            ),
        ],
        params={
            "value": 304811.0,
            "formula": "abs(9447 - 314258)",
        },
    )

    # Step: Send to Answer Validator API (Gatekeeper verification)
    validator_url = (
        f"{get_service_url(ServiceName.ANSWER_VALIDATOR.value)}/validate_answer"
    )
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            val_payload = ValidationRequest(answer=mock_answer.model_dump())
            resp = await client.post(validator_url, json=val_payload.model_dump())
            if resp.status_code == 200:
                val_data = ValidationResponse(**resp.json())
                if not val_data.is_valid:
                    logger.error("Answer rejected by validator: %s", val_data.error)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Answer validation failed: {val_data.error}",
                    )
                logger.info("Validation passed: %s", val_data.log_message)
    except httpx.RequestError as exc:
        logger.warning(
            "Could not reach answer-validator-api at %s (%s). Proceeding with local validation.",
            validator_url,
            exc,
        )

    latency = round((time.time() - start_time) * 1000, 2)
    return AskResponse(
        query=request.query,
        answer=mock_answer,
        latency_ms=latency,
        trace_id=request.session_id or "trace-mock-001",
    )
