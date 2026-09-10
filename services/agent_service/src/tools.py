import ast
import operator

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from shared.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    ServiceName,
    get_service_url,
)
from shared.models import DecompositionResult

from .llm import get_llm, invoke_with_retry
from .prompts import DECOMPOSE_PROMPT

RETRIEVAL_API_URL = f"{get_service_url(ServiceName.RETRIEVAL.value)}/search"


# -----------------------------------------
# decompose tool -
# --------------------------------------------
@tool(
    "decompose_question",
    description="Break a complex financial question into smaller sub-questions.",
)
def decompose_question(query: str) -> list[str]:
    decomposition_llm = get_llm().with_structured_output(DecompositionResult)

    result = invoke_with_retry(
        decomposition_llm,
        [
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    return result.sub_questions


# ---------------------------------------------------------
# Serach Document tool-
# ----------------------------------------------------------
@tool(
    "search_documents",
    description="Search relevant information from financial documents",
)
def search_documents(query: str, filters: dict | None = None):
    response = requests.post(
        RETRIEVAL_API_URL,
        json={
            "query": query,
            "top_k": 30,
            "top_n": 5,
            "filters": filters,
            "use_reranking": True,
        },
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


# ------------------------------------------------------------------------
# Serach tables tool
# ------------------------------------------------------------------------


@tool(
    "search_tables",
    description="Search the financial document corpus for relevant table data.",
)
def search_tables(
    query: str,
    filters: dict | None = None,
):

    table_filters = filters.copy() if filters else {}
    table_filters["content_type"] = "table"

    response = requests.post(
        RETRIEVAL_API_URL,
        json={
            "query": query,
            "top_k": 30,
            "top_n": 5,
            "filters": table_filters,
            "use_reranking": True,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ------------------------------------------------------------------------
# Calculation  tool
# ------------------------------------------------------------------------
## Safe Calculation (save eval)

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_OPERATORS:
            raise ValueError("Operator is not allowed.")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_OPERATORS:
            raise ValueError("Operator is not allowed.")

        operand = _safe_eval(node.operand)

        return _ALLOWED_OPERATORS[type(node.op)](operand)

    raise ValueError("Invalid mathematical expression.")


# ------------------------------------
# calculate tool
@tool(
    "calculate",
    description="Safely evaluate a mathematical expression.Only basic arithmetic operations are allowed.",
)
def calculate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)

        return float(result)

    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid calculation: {e}")


# -------------------------------------------------
# -------------------------------------------------

tools = [decompose_question, search_documents, search_tables, calculate]
