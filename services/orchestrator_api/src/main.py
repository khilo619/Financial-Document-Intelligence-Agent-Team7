"""
orchestrator_api: Central API Gateway for Project LEDGER.

Responsibilities:
- Receive user questions from the UI / evaluation service.
- Forward questions to the reasoning agent.
- Send every candidate answer to the answer validator.
- Return only validated answers to the caller.

Owned by Ahmed (Member 6).
"""

import logging
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from shared.config import ServiceName, get_service_url
from shared.models import (
    AskRequest,
    AskResponse,
    StrictAnswer,
    ValidationRequest,
    ValidationResponse,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)

logger = logging.getLogger("OrchestratorAPI")


# ---------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------

AGENT_URL = f"{get_service_url(ServiceName.AGENT.value)}/solve"

VALIDATOR_URL = (
    f"{get_service_url(ServiceName.ANSWER_VALIDATOR.value)}/validate_answer"
)


# ---------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Project LEDGER - Orchestrator Gateway API",
    description=(
        "Central API gateway coordinating questions between the UI, "
        "reasoning agent, and answer validator."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Simple health endpoint used by Docker and the UI dashboard.
    """

    return {
        "status": "healthy",
        "service": ServiceName.ORCHESTRATOR.value,
        "port": 8001,
    }


# ---------------------------------------------------------------------
# Main Question Endpoint
# ---------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Main corpus-wide question endpoint.

    Flow:

        User / UI
            ↓
        Orchestrator
            ↓
        Agent Service
            ↓
        Answer Validator
            ↓
        Orchestrator
            ↓
        Validated Answer

    The orchestrator fails closed:
    an answer is NEVER returned if the validator cannot validate it.
    """

    start_time = time.perf_counter()

    trace_id = request.session_id or str(uuid.uuid4())

    scope = request.document_id or "corpus-wide"

    logger.info(
        "Received query='%s' scope='%s' trace_id='%s'",
        request.query,
        scope,
        trace_id,
    )

    # ================================================================
    # STEP 1 — Send question to Agent Service
    # ================================================================

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:

            agent_response = await client.post(
                AGENT_URL,
                json=request.model_dump(),
            )

    except httpx.TimeoutException as exc:

        logger.error("Agent service timed out: %s", exc)

        raise HTTPException(
            status_code=504,
            detail="Agent service timed out while processing the question.",
        ) from exc

    except httpx.RequestError as exc:

        logger.error(
            "Could not connect to agent service at %s: %s",
            AGENT_URL,
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="Agent service is currently unavailable.",
        ) from exc

    if agent_response.status_code != 200:

        logger.error(
            "Agent returned HTTP %s: %s",
            agent_response.status_code,
            agent_response.text,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Agent service failed while generating a candidate answer."
            ),
        )

    # ================================================================
    # STEP 2 — Read raw candidate answer
    # ================================================================

    try:
        candidate_answer = agent_response.json()

    except ValueError as exc:

        logger.error(
            "Agent returned invalid JSON: %s",
            agent_response.text,
        )

        raise HTTPException(
            status_code=502,
            detail="Agent service returned malformed JSON.",
        ) from exc

    if not isinstance(candidate_answer, dict):

        logger.error(
            "Agent response must be an object, received: %s",
            type(candidate_answer).__name__,
        )

        raise HTTPException(
            status_code=502,
            detail="Agent service returned an invalid answer structure.",
        )

    logger.info(
        "Agent produced candidate answer type='%s'",
        candidate_answer.get("answer_type", "unknown"),
    )

    # ================================================================
    # STEP 3 — Mandatory Answer Validator Gate
    # ================================================================

    validation_payload = ValidationRequest(
        answer=candidate_answer,
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:

            validator_response = await client.post(
                VALIDATOR_URL,
                json=validation_payload.model_dump(),
            )

    except httpx.TimeoutException as exc:

        logger.error("Answer validator timed out: %s", exc)

        raise HTTPException(
            status_code=504,
            detail="Answer validator timed out.",
        ) from exc

    except httpx.RequestError as exc:

        logger.error(
            "Could not reach answer validator at %s: %s",
            VALIDATOR_URL,
            exc,
        )

        # IMPORTANT:
        # Do NOT return an unvalidated answer.
        raise HTTPException(
            status_code=503,
            detail=(
                "Answer validator is unavailable. "
                "The answer cannot safely be returned."
            ),
        ) from exc

    if validator_response.status_code != 200:

        logger.error(
            "Validator returned HTTP %s: %s",
            validator_response.status_code,
            validator_response.text,
        )

        raise HTTPException(
            status_code=502,
            detail="Answer validator service failed.",
        )

    # ================================================================
    # STEP 4 — Parse Validator Result
    # ================================================================

    try:

        validation_result = ValidationResponse(
            **validator_response.json()
        )

    except Exception as exc:  # noqa: BLE001

        logger.error(
            "Malformed validator response: %s",
            validator_response.text,
        )

        raise HTTPException(
            status_code=502,
            detail="Validator returned an invalid response.",
        ) from exc

    # ================================================================
    # STEP 5 — Reject Invalid Answer
    # ================================================================

    if not validation_result.is_valid:

        logger.error(
            "Candidate answer rejected by validator: %s",
            validation_result.error,
        )

        raise HTTPException(
            status_code=422,
            detail={
                "message": "Generated answer failed validation.",
                "reason": validation_result.error,
            },
        )

    logger.info(
        "Answer validation successful: %s",
        validation_result.log_message,
    )

    # ================================================================
    # STEP 6 — Convert validated result to canonical schema
    # ================================================================

    try:

        final_answer = StrictAnswer(
            **candidate_answer
        )

    except Exception as exc:  # noqa: BLE001

        logger.error(
            "Validated answer could not be parsed locally: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Validated answer could not be parsed.",
        ) from exc

    # ================================================================
    # STEP 7 — Final Response
    # ================================================================

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    logger.info(
        "Query completed successfully in %.2f ms trace_id='%s'",
        latency_ms,
        trace_id,
    )

    return AskResponse(
        query=request.query,
        answer=final_answer,
        latency_ms=latency_ms,
        trace_id=trace_id,
    )