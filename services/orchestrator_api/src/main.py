"""
orchestrator_api: Central API Gateway for Project LEDGER.

Responsibilities:
- Receive user questions from the UI / evaluation service.
- Forward questions to the reasoning agent.
- Send every candidate answer to the answer validator.
- Return only validated answers to the caller.
- Monitor backend service health for the UI dashboard.
- Track recent queries and their latency.

Owned by Ahmed (Member 6).
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

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


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)

logger = logging.getLogger("OrchestratorAPI")


# =============================================================================
# Service URLs
# =============================================================================

AGENT_URL = f"{get_service_url(ServiceName.AGENT.value)}/solve"

VALIDATOR_URL = (
    f"{get_service_url(ServiceName.ANSWER_VALIDATOR.value)}/validate_answer"
)


# Backend services checked by the dashboard.
SERVICE_HEALTH_TARGETS = {
    ServiceName.DOC_PROCESSOR.value: (
        get_service_url(ServiceName.DOC_PROCESSOR.value),
        "/health",
    ),
    ServiceName.RETRIEVAL.value: (
        get_service_url(ServiceName.RETRIEVAL.value),
        "/health",
    ),
    ServiceName.AGENT.value: (
        get_service_url(ServiceName.AGENT.value),
        "/health",
    ),
    ServiceName.ANSWER_VALIDATOR.value: (
        get_service_url(ServiceName.ANSWER_VALIDATOR.value),
        "/health",
    ),
    ServiceName.EVAL.value: (
        get_service_url(ServiceName.EVAL.value),
        "/health",
    ),
}


# =============================================================================
# Runtime State
# =============================================================================

# Small in-memory history used by the Gradio dashboard.
#
# This is intentionally lightweight:
# - no external database is required
# - newest query appears first
# - only the latest 50 requests are retained
#
# It resets when the orchestrator process restarts.
RECENT_QUERIES: deque[dict[str, Any]] = deque(maxlen=50)


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Project LEDGER - Orchestrator Gateway API",
    description=(
        "Central API gateway coordinating questions between the UI, "
        "reasoning agent, answer validator, and supporting services."
    ),
    version="1.1.0",
)


# =============================================================================
# Helpers
# =============================================================================

def _utc_timestamp() -> str:
    """
    Return a timezone-aware UTC timestamp suitable for the dashboard.
    """

    return datetime.now(timezone.utc).isoformat()


def _record_query(
    *,
    query: str,
    document_id: str | None,
    trace_id: str,
    answer_type: str | None,
    latency_ms: float,
    status: str,
    error: str | None = None,
) -> None:
    """
    Store one query execution summary for the UI dashboard.
    """

    RECENT_QUERIES.appendleft(
        {
            "timestamp": _utc_timestamp(),
            "query": query,
            "document_id": document_id,
            "scope": document_id or "corpus-wide",
            "answer_type": answer_type,
            "latency_ms": latency_ms,
            "status": status,
            "trace_id": trace_id,
            "error": error,
        }
    )


async def _check_service_health(
    client: httpx.AsyncClient,
    service_name: str,
    base_url: str,
    health_path: str,
) -> tuple[str, dict[str, Any]]:
    """
    Check one backend service without failing the whole health request.
    """

    url = f"{base_url}{health_path}"
    started = time.perf_counter()

    try:
        response = await client.get(url)

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {}

            return service_name, {
                "status": "healthy",
                "url": base_url,
                "latency_ms": latency_ms,
                "details": payload,
            }

        return service_name, {
            "status": "unhealthy",
            "url": base_url,
            "latency_ms": latency_ms,
            "http_status": response.status_code,
        }

    except httpx.TimeoutException:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return service_name, {
            "status": "timeout",
            "url": base_url,
            "latency_ms": latency_ms,
        }

    except httpx.RequestError:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return service_name, {
            "status": "offline",
            "url": base_url,
            "latency_ms": latency_ms,
        }


def _raise_and_record(
    *,
    status_code: int,
    detail: Any,
    request: AskRequest,
    trace_id: str,
    start_time: float,
    answer_type: str | None = None,
) -> None:
    """
    Record a failed query before raising an HTTPException.
    """

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    error_text = (
        detail
        if isinstance(detail, str)
        else str(detail)
    )

    _record_query(
        query=request.query,
        document_id=request.document_id,
        trace_id=trace_id,
        answer_type=answer_type,
        latency_ms=latency_ms,
        status="error",
        error=error_text,
    )

    raise HTTPException(
        status_code=status_code,
        detail=detail,
    )


# =============================================================================
# Basic Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """
    Health endpoint for the orchestrator itself.
    """

    return {
        "status": "healthy",
        "service": ServiceName.ORCHESTRATOR.value,
        "port": 8001,
    }


# =============================================================================
# Backend Services Health
# =============================================================================

@app.get("/services/health")
async def services_health():
    """
    Check all backend LEDGER services concurrently.

    This endpoint is consumed by the Gradio dashboard.
    One unavailable service does not cause this endpoint itself to fail.
    """

    async with httpx.AsyncClient(timeout=2.5) as client:
        tasks = [
            _check_service_health(
                client=client,
                service_name=service_name,
                base_url=base_url,
                health_path=health_path,
            )
            for service_name, (
                base_url,
                health_path,
            ) in SERVICE_HEALTH_TARGETS.items()
        ]

        service_results = await asyncio.gather(*tasks)

    services = {
        service_name: result
        for service_name, result in service_results
    }

    healthy_count = sum(
        1
        for result in services.values()
        if result["status"] == "healthy"
    )

    total_count = len(services)

    if healthy_count == total_count:
        overall_status = "healthy"
    elif healthy_count == 0:
        overall_status = "unavailable"
    else:
        overall_status = "degraded"

    return {
        "overall_status": overall_status,
        "orchestrator": {
            "status": "healthy",
            "url": get_service_url(
                ServiceName.ORCHESTRATOR.value
            ),
        },
        "healthy_services": healthy_count,
        "total_services": total_count,
        "services": services,
    }


# =============================================================================
# Recent Queries
# =============================================================================

@app.get("/recent-queries")
def recent_queries(limit: int = 10):
    """
    Return recent orchestrated requests for the UI dashboard.

    Maximum returned records: 50.
    """

    safe_limit = max(
        1,
        min(limit, 50),
    )

    records = list(RECENT_QUERIES)[:safe_limit]

    successful_queries = [
        item
        for item in RECENT_QUERIES
        if item["status"] == "success"
    ]

    if successful_queries:
        average_latency_ms = round(
            sum(
                item["latency_ms"]
                for item in successful_queries
            )
            / len(successful_queries),
            2,
        )
    else:
        average_latency_ms = 0.0

    return {
        "total_recorded": len(RECENT_QUERIES),
        "successful_queries": len(successful_queries),
        "average_latency_ms": average_latency_ms,
        "queries": records,
    }


# =============================================================================
# Main Question Endpoint
# =============================================================================

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

    trace_id = str(uuid.uuid4())

    scope = request.document_id or "corpus-wide"

    logger.info(
        "Received query='%s' scope='%s' trace_id='%s'",
        request.query,
        scope,
        trace_id,
    )

    candidate_answer_type: str | None = None

    # =========================================================================
    # STEP 1 — Send Question to Agent Service
    # =========================================================================

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            agent_response = await client.post(
                AGENT_URL,
                json=request.model_dump(),
            )

    except httpx.TimeoutException as exc:
        logger.error(
            "Agent service timed out: %s",
            exc,
        )

        _raise_and_record(
            status_code=504,
            detail=(
                "Agent service timed out while processing "
                "the question."
            ),
            request=request,
            trace_id=trace_id,
            start_time=start_time,
        )

    except httpx.RequestError as exc:
        logger.error(
            "Could not connect to agent service at %s: %s",
            AGENT_URL,
            exc,
        )

        _raise_and_record(
            status_code=503,
            detail="Agent service is currently unavailable.",
            request=request,
            trace_id=trace_id,
            start_time=start_time,
        )

    if agent_response.status_code != 200:
        logger.error(
            "Agent returned HTTP %s: %s",
            agent_response.status_code,
            agent_response.text,
        )

        _raise_and_record(
            status_code=502,
            detail=(
                "Agent service failed while generating "
                "a candidate answer."
            ),
            request=request,
            trace_id=trace_id,
            start_time=start_time,
        )

    # =========================================================================
    # STEP 2 — Read Raw Candidate Answer
    # =========================================================================

    try:
        candidate_answer = agent_response.json()

    except ValueError:
        logger.error(
            "Agent returned invalid JSON: %s",
            agent_response.text,
        )

        _raise_and_record(
            status_code=502,
            detail="Agent service returned malformed JSON.",
            request=request,
            trace_id=trace_id,
            start_time=start_time,
        )

    if not isinstance(candidate_answer, dict):
        logger.error(
            "Agent response must be an object, received: %s",
            type(candidate_answer).__name__,
        )

        _raise_and_record(
            status_code=502,
            detail=(
                "Agent service returned an invalid "
                "answer structure."
            ),
            request=request,
            trace_id=trace_id,
            start_time=start_time,
        )

    candidate_answer_type = candidate_answer.get(
        "answer_type"
    )

    logger.info(
        "Agent produced candidate answer type='%s'",
        candidate_answer_type or "unknown",
    )

    # =========================================================================
    # STEP 3 — Mandatory Answer Validator Gate
    # =========================================================================

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
        logger.error(
            "Answer validator timed out: %s",
            exc,
        )

        _raise_and_record(
            status_code=504,
            detail="Answer validator timed out.",
            request=request,
            trace_id=trace_id,
            start_time=start_time,
            answer_type=candidate_answer_type,
        )

    except httpx.RequestError as exc:
        logger.error(
            "Could not reach answer validator at %s: %s",
            VALIDATOR_URL,
            exc,
        )

        # Fail closed:
        # NEVER expose an unvalidated financial answer.
        _raise_and_record(
            status_code=503,
            detail=(
                "Answer validator is unavailable. "
                "The answer cannot safely be returned."
            ),
            request=request,
            trace_id=trace_id,
            start_time=start_time,
            answer_type=candidate_answer_type,
        )

    if validator_response.status_code != 200:
        logger.error(
            "Validator returned HTTP %s: %s",
            validator_response.status_code,
            validator_response.text,
        )

        _raise_and_record(
            status_code=502,
            detail="Answer validator service failed.",
            request=request,
            trace_id=trace_id,
            start_time=start_time,
            answer_type=candidate_answer_type,
        )

    # =========================================================================
    # STEP 4 — Parse Validator Result
    # =========================================================================

    try:
        validation_result = ValidationResponse(
            **validator_response.json()
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Malformed validator response: %s",
            validator_response.text,
        )

        _raise_and_record(
            status_code=502,
            detail="Validator returned an invalid response.",
            request=request,
            trace_id=trace_id,
            start_time=start_time,
            answer_type=candidate_answer_type,
        )

    # =========================================================================
    # STEP 5 — Reject Invalid Answer
    # =========================================================================

    if not validation_result.is_valid:
        logger.error(
            "Candidate answer rejected by validator: %s",
            validation_result.error,
        )

        _raise_and_record(
            status_code=422,
            detail={
                "message": (
                    "Generated answer failed validation."
                ),
                "reason": validation_result.error,
            },
            request=request,
            trace_id=trace_id,
            start_time=start_time,
            answer_type=candidate_answer_type,
        )

    logger.info(
        "Answer validation successful: %s",
        validation_result.log_message,
    )

    # =========================================================================
    # STEP 6 — Convert to Canonical Strict Answer
    # =========================================================================

    try:
        final_answer = StrictAnswer(
            **candidate_answer
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Validated answer could not be parsed locally: %s",
            exc,
        )

        _raise_and_record(
            status_code=500,
            detail="Validated answer could not be parsed.",
            request=request,
            trace_id=trace_id,
            start_time=start_time,
            answer_type=candidate_answer_type,
        )

    # =========================================================================
    # STEP 7 — Final Response
    # =========================================================================

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    logger.info(
        "Query completed successfully in %.2f ms "
        "trace_id='%s'",
        latency_ms,
        trace_id,
    )

    _record_query(
        query=request.query,
        document_id=request.document_id,
        trace_id=trace_id,
        answer_type=final_answer.answer_type,
        latency_ms=latency_ms,
        status="success",
    )

    return AskResponse(
        query=request.query,
        answer=final_answer,
        latency_ms=latency_ms,
        trace_id=trace_id,
    )