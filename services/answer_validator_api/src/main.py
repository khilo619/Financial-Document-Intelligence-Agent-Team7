"""
answer_validator_api: Strict Schema Gatekeeper & Safe Python Calculator.
Owned by Omar (Member 4) - Initial implementation by Khaled (Repo Lead).
"""

import logging

import simpleeval
from fastapi import FastAPI
from pydantic import ValidationError

from shared.config import VALIDATOR_ERROR_PREFIX, VALIDATOR_SUCCESS_PREFIX, ServiceName
from shared.models import StrictAnswer, ValidationRequest, ValidationResponse

# Configure root logger with strict formatting
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AnswerValidator")

app = FastAPI(
    title="Project LEDGER - Answer Validator API",
    description="The single source of truth for schema compliance, evidence grounding, and deterministic arithmetic.",
    version="0.1.0",
)


def _clean_pydantic_error(exc: ValidationError) -> str:
    """Convert Pydantic's verbose error into a short, spec-style message."""
    first = exc.errors()[0]
    field = first["loc"][-1] if first["loc"] else "field"
    err_type = first["type"]

    if err_type == "missing":
        return f"Missing required key '{field}'"
    if "type" in err_type:  # e.g. int_type, string_type, float_type
        return f"Invalid type for key '{field}'"
    return first["msg"]  # fallback to Pydantic's own short message


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.ANSWER_VALIDATOR.value,
        "port": 8005,
    }


@app.post("/validate_answer", response_model=ValidationResponse)
def validate_answer(request: ValidationRequest):
    """
    Strictly validates incoming answer payload against the Project LEDGER schema.
    Emits required grading logs on stdout.
    """
    raw_payload = request.answer
    a_type = raw_payload.get("answer_type", "unknown")



    try:
        validated_answer = StrictAnswer(**raw_payload)
        evidence_dicts = [e.model_dump() for e in validated_answer.evidence]
        log_msg = (
            f"{VALIDATOR_SUCCESS_PREFIX} Received and validated answer of type "
            f"'{validated_answer.answer_type}' with evidence {evidence_dicts}"
        )
        logger.info(log_msg)
        return ValidationResponse(
            is_valid=True,
            answer_type=validated_answer.answer_type,
            error=None,
            log_message=log_msg,
        )
    except (ValueError) as exc:
        error_str = str(exc)
        log_msg = f"{VALIDATOR_ERROR_PREFIX} Invalid answer for '{a_type}': {error_str}"
        logger.error(log_msg)
        return ValidationResponse(
            is_valid=False,
            answer_type=a_type,
            error=error_str,
            log_message=log_msg,
        )
    except (ValidationError) as exc:
            error_str = _clean_pydantic_error(exc)
            log_msg = f"{VALIDATOR_ERROR_PREFIX} Invalid answer for '{a_type}': {error_str}"
            logger.error(log_msg)
            return ValidationResponse(
                is_valid=False,
                answer_type=a_type,
                error=error_str,
                log_message=log_msg,
            )


@app.post("/calculate")
def safe_calculate(payload: dict):
    """
    Deterministically computes mathematical expressions.
    NEVER uses unsafe eval(). Uses simpleeval with strict math functions.
    """
    expression = payload.get("expression", "")
    safe_functions = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}

    try:
        result = simpleeval.simple_eval(expression, functions=safe_functions)
        return {"expression": expression, "result": float(result), "status": "success"}
    except (
        simpleeval.InvalidExpression,
        ValueError,
        TypeError,
        ZeroDivisionError,
    ) as exc:
        logger.error("Calculation error on '%s': %s", expression, exc)
        return {"expression": expression, "error": str(exc), "status": "error"}
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected calculation error on '%s': %s", expression, exc)
        return {"expression": expression, "error": str(exc), "status": "error"}
